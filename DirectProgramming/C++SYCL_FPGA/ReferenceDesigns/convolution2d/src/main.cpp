//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// main.cpp

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#define NUM_FRAMES 5
#include <stdlib.h>  // malloc, free

#include <fstream>  // ofstream
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>

#include "bmp_tools.hpp"
#include "convolution_kernel.hpp"
#include "data_bundle.hpp"
#include "exception_handler.hpp"
#include "vvp_stream_adapters.hpp"

#ifndef DEFAULT_EXTENSION
#define DEFAULT_EXTENSION ".bmp"
#endif

#ifndef DEFAULT_INPUT
#define DEFAULT_INPUT "../test_bitmaps/test"
#endif

#ifndef DEFAULT_OUTPUT
#define DEFAULT_OUTPUT "./output"
#endif

#ifndef DEFAULT_EXPECTED
#define DEFAULT_EXPECTED "../test_bitmaps/expected_sobel"
#endif

#define M_DEFAULT_INPUT DEFAULT_INPUT
#define M_DEFAULT_OUTPUT DEFAULT_OUTPUT
#define M_DEFAULT_EXPECTED DEFAULT_EXPECTED

#define ERR_MSG_BUF_SIZE 256

/////////////////////
// Test subroutines
/////////////////////

/// @brief Initialize a buffer meant to store an image. This helps with
/// debugging, because you can see if an image was only partly written. The
/// buffer is initialized with an incrementing pattern, ranging from `0` to
/// `(1 << 24)`.
/// @param[out] buf Buffer to initialize
/// @param[in] size Number of pixels to initialize in the buffer
void initializeBuffer(conv2d::Pixel_rgb *buf, size_t size) {
  uint16_t pixel = 0;
  for (size_t i = 0; i < size; i++) {
    pixel++;
    buf[i] = conv2d::Pixel_rgb{pixel, pixel, pixel};
  }
}

/// @brief Convert pixels read from a bmp image using the bmptools functions to
/// pixels that can be parsed by our 2D convolution IP.
/// @param[in] bmp_buf pixels read by bmptools
/// @param[out] vvp_buf pixels to be consumed by 2D convolution IP
/// @param[in] pixel_count (input) number of pixels in input image and output
/// image
void convertToVvpRgb(unsigned int *bmp_buf, conv2d::Pixel_rgb *vvp_buf,
                     size_t pixel_count) {
  std::cout << "INFO: convert to vvp type." << std::endl;
  for (int idx = 0; idx < pixel_count; idx++) {
    uint32_t pixel_int = bmp_buf[idx];
    BmpTools::PixelRGB bmp_rgb(pixel_int);

    // convert from 8-bit to whatever the VVP IP expects
    conv2d::Pixel_rgb pixel_vvp{
        (uint16_t)(bmp_rgb.b << (conv2d::kBitsPerChannel - 8)),   //
        (uint16_t)(bmp_rgb.g << (conv2d::kBitsPerChannel - 8)),   //
        (uint16_t)(bmp_rgb.r << (conv2d::kBitsPerChannel - 8))};  //

    vvp_buf[idx] = pixel_vvp;
  }
}

/// @brief Convert pixels read from the 2D convolution IP to a format that can
/// be read by the bmptools functions.
/// @param[in] vvp_buf pixels produced by 2D convolution IP
/// @param[out] bmp_buf pixels to send to bmptools
/// @param[in] pixel_count number of pixels in input image and output image
void convertToBmpRgb(conv2d::Pixel_rgb *vvp_buf, unsigned int *bmp_buf,
                     size_t pixel_count) {
  std::cout << "INFO: convert to bmp type." << std::endl;
  for (int idx = 0; idx < pixel_count; idx++) {
    conv2d::Pixel_rgb pixelConv = vvp_buf[idx];

    // convert the VVP IP back to 8-bit
    BmpTools::PixelRGB bmp_rgb(
        (uint8_t)(pixelConv.r >> (conv2d::kBitsPerChannel - 8)),   //
        (uint8_t)(pixelConv.g >> (conv2d::kBitsPerChannel - 8)),   //
        (uint8_t)(pixelConv.b >> (conv2d::kBitsPerChannel - 8)));  //

    uint32_t pixel_int = bmp_rgb.getImgPixel();
    bmp_buf[idx] = pixel_int;
  }
}

/// @brief Verify image dimensions from a just-read image and compare with
/// previous image dimensions if appropriate.
/// @param rows previous image rows (0 if no previous dimensions)
/// @param cols previous image cols (0 if no previous dimensions)
/// @param rows_new new image rows
/// @param cols_new new image columns
/// @return `true` if new dimensions are acceptable, and `rows` and `cols` have
/// been successfully updated to `rows_new` and `cols_new` respectively.
bool updateAndCheckImageDimensions(size_t &rows, size_t &cols, size_t rows_new,
                                   size_t cols_new) {
  // sanity check: all images should be the same size
  if (rows == 0)
    rows = rows_new;
  else if (rows != rows_new) {
    std::cerr << "ERROR: dimensions of images must match. Expected " << rows
              << " but saw " << rows_new << "!" << std::endl;
    return false;
  }

  if (cols == 0)
    cols = cols_new;
  else if (cols != cols_new) {
    std::cerr << "ERROR: dimensions of images must match. Expected " << cols
              << " but saw " << cols_new << "!" << std::endl;
    return false;
  }

  // max allowable value for rows * cols must be less than the max value of a
  // signed 32-bit integer.
  constexpr int ROWS_COLS_MAX = 1 << 29;

  bool imageSizeOk =
      (rows_new > 0) && (cols_new > 0) && (rows_new * cols_new < ROWS_COLS_MAX);

  // sanity check; this design assumes that the number of columns in the input
  // image is a multiple of kParallelPixels.
  if (cols % conv2d::kParallelPixels != 0) {
    std::cerr << "ERROR: image cols = " << cols
              << " not compatible with kernel compiled for "
              << conv2d::kParallelPixels
              << " pixels in parallel. Please choose a different image, or "
                 "recompile "
                 "with a different value of the PARALLEL_PIXELS "
                 "pre-processor macro."
              << std::endl;
    return false;
  }

  return imageSizeOk;
}

#if TEST_CONV2D_ISOLATED

constexpr std::array<float, 9> identity_coeffs = {
    0.0f, 0.0f, 0.0f,  //
    0.0f, 1.0f, 0.0f,  //
    0.0f, 0.0f, 0.0f   //
};

/// @brief Trivial test that exercises Convolution2d on its own, using an
/// extremely simple image. This is useful for debugging that data is flowing
/// through the line buffer properly.
/// @param q The SYCL queue to assign work to
/// @return `true` if successful, `false` otherwise
bool testTinyFrameOnStencil(sycl::queue q) {
  constexpr int rowsSmall = 3;
  constexpr int colsSmall = 8;

  constexpr int pixelsCount = rowsSmall * colsSmall;

  conv2d::PixelType greyPixelsIn[] = {
      101, 201, 301, 401, 501, 601, 701, 801,  //
      102, 202, 302, 402, 502, 602, 702, 802,  //
      103, 203, 303, 403, 503, 603, 703, 803};

  vvp_stream_adapters::writeFrameToPipe<InputImageStreamGrey>(
      q, rowsSmall, colsSmall, greyPixelsIn);

  // extra pixels to flush out the FIFO
  int dummyPixels = colsSmall * conv2d::kWindowSize;
  vvp_stream_adapters::writeDummyPixelsToPipe<InputImageStreamGrey>(
      q, dummyPixels, (uint16_t)69);

  // disable bypass, since it's on by default
  BypassCSR::write(q, false);

  sycl::event frameEvent = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rowsSmall, (int)colsSmall, identity_coeffs});

  conv2d::PixelType greyPixelsOut[pixelsCount];
  bool sidebandsOk;
  int defectiveFrames;
  vvp_stream_adapters::readFrameFromPipe<OutputImageStreamGrey>(
      q, rowsSmall, colsSmall, greyPixelsOut, sidebandsOk, defectiveFrames);

  bool pixelsMatch = true;
  for (int i = 0; i < pixelsCount; i++) {
    constexpr float kOutputOffset = ((1 << conv2d::kBitsPerChannel) / 2);
    constexpr float kNormalizationFactor = (1 << conv2d::kBitsPerChannel);
    conv2d::PixelType greyPixelExpected =
        ((float)greyPixelsIn[i] / kNormalizationFactor) * kOutputOffset +
        kOutputOffset;
    pixelsMatch &= (greyPixelExpected == greyPixelsOut[i]);
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  frameEvent.wait();

  return sidebandsOk & pixelsMatch;
}

#else

constexpr std::array<float, 9> sobel_coeffs = {
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f,  //
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f,  //
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f   //
};

/// @brief 'Happy Path' test that repeatedly passes a known good frame through
/// the IP.
/// @param q SYCL queue
/// @param num_frames Number of times to pass the image frame through the IP
/// @param input_bmp_filename base filename to use for reading input frames
/// @param output_bmp_filename_base base filename to use for writing output
/// files
/// @param canonical_expected_bmp_path 'known good' file to compare IP output
/// against
/// @return `true` if all frames emitted by the IP match the `known good` file,
/// `false` otherwise.
bool testGoodFramesSequence(sycl::queue q, size_t num_frames,
                            std::string input_bmp_filename_base,
                            std::string output_bmp_filename_base,
                            std::string expected_bmp_filename_base) {
  std::cout << "\n**********************************\n"
            << "Check a sequence of good frames... "
            << "\n**********************************\n"
            << std::endl;

  sycl::event frameEvent;
  bool allPassed = true;

  size_t rows = 0, cols = 0;

  for (int itr = 0; itr < num_frames; itr++) {
    // load image
    unsigned int *in_img = 0;
    int rows_new, cols_new;

    std::string canonical_input_bmp_path =  //
        input_bmp_filename_base + "_" + std::to_string(itr) + DEFAULT_EXTENSION;
    std::string canonical_expected_bmp_path =  //
        expected_bmp_filename_base + "_" + std::to_string(itr) +
        DEFAULT_EXTENSION;

    std::cout << "INFO: Load image " << canonical_input_bmp_path << std::endl;
    if (!BmpTools::read_bmp(canonical_input_bmp_path, &in_img, rows_new,
                            cols_new)) {
      std::cerr << "ERROR: Could not read image from "
                << canonical_input_bmp_path << std::endl;
      return false;
    }

    bool imageOk =
        updateAndCheckImageDimensions(rows, cols, rows_new, cols_new);

    if (!imageOk) {
      std::cerr << "ERROR: invalid image size " << rows << " x " << cols
                << std::endl;
      continue;
    }

    conv2d::Pixel_rgb *in_img_vvp = new conv2d::Pixel_rgb[rows * cols];

    convertToVvpRgb(in_img, in_img_vvp, rows * cols);

    // don't need in_img anymore
    free(in_img);

    vvp_stream_adapters::writeFrameToPipe<InputImageStream>(q, rows, cols,
                                                            in_img_vvp);

    // don't need in_img_vvp anymore
    delete[] in_img_vvp;
  }

  // extra pixels to flush out the FIFO
  int dummyPixels = cols * conv2d::kWindowSize;
  constexpr auto kDummyVal = conv2d::Pixel_rgb{69, 69, 69};
  vvp_stream_adapters::writeDummyPixelsToPipe<InputImageStream,
                                              conv2d::Pixel_rgb>(q, dummyPixels,
                                                                 kDummyVal);

  std::cout << "Launch kernels! " << std::endl;

  q.single_task<ID_RGB2Grey>(
      RGB2Grey<InputImageStream, InputImageStreamGrey>{});
  frameEvent = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, sobel_coeffs});
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  for (int itr = 0; itr < num_frames; itr++) {
    std::cout << "\n*********************\n"  //
              << "Reading out frame " << itr  //
              << std::endl;

    if (itr == 0) {
      std::cout << "Expect some unexpected start-of-packets as the kernel "
                   "flushes its initial state."
                << std::endl;
    }
    std::string absolute_output_bmp_path = output_bmp_filename_base + "_" +
                                           std::to_string(itr) +
                                           DEFAULT_EXTENSION;

    conv2d::Pixel_rgb *out_img_vvp = new conv2d::Pixel_rgb[rows * cols];
    unsigned int *out_img = new unsigned int[rows * cols];
    initializeBuffer(out_img_vvp, rows * cols);

    int defectiveFrames = 0;
    bool sidebands_ok = false;
    if (out_img_vvp) {
      vvp_stream_adapters::readFrameFromPipe<OutputImageStream>(
          q, rows, cols, out_img_vvp, sidebands_ok, defectiveFrames);
    }

    if (out_img) {
      convertToBmpRgb(out_img_vvp, out_img, rows * cols);
      delete[] out_img_vvp;

      BmpTools::write_bmp(absolute_output_bmp_path, out_img, rows, cols);
      std::cout << "Wrote convolved image " << absolute_output_bmp_path
                << std::endl;
    } else {
      std::cerr << "ERROR: could not write output image: out_img=null."
                << std::endl;
    }

    std::string absolute_expected_bmp_path = expected_bmp_filename_base + "_" +
                                             std::to_string(itr) +
                                             DEFAULT_EXTENSION;

    std::cout << "Compare with " << absolute_expected_bmp_path << ". "
              << std::endl;
    bool passed = BmpTools::compareFrames(out_img, rows, cols,
                                          absolute_expected_bmp_path);

    delete[] out_img;
    allPassed &= passed & sidebands_ok;
    printf("frame %d %s\n", itr,
           (passed && sidebands_ok) ? "passed" : "failed");
  }

  int detected_version = VersionCSR::read(q);
  std::cout << "\nKernel version = " << detected_version << " (Expected "
            << kKernelVersion << ")" << std::endl;

  if (detected_version != kKernelVersion) {
    std::cerr << "ERROR: kernel version did not match!" << std::endl;
    allPassed = false;
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  frameEvent.wait();

  std::cout << "\nFinished checking a sequence of good frames.\n\n"
            << std::endl;

  return allPassed;
}

/// @brief Test how the IP handles a defective frame by passing a defective
/// frame (which should emit a partial image) followed by a complete frame
/// which should match the 'known good' file.
/// @param q SYCL queue
/// @param rows Number of rows in the image frame
/// @param cols Number of columns in the image frame
/// @param in_img Buffer containing the image frame to process
/// @param output_bmp_filename_base File to output the processed frame to
/// @param canonical_expected_bmp_path 'known good' file to compare IP output
/// against
/// @return `true` if the second frame emitted by the IP matches the `known
/// good` file, `false` otherwise.
bool testDefectiveFrame(sycl::queue q, std::string input_bmp_filename,
                        std::string output_bmp_filename_base,
                        std::string expected_bmp_filename) {
  std::cout << "\n******************************************************\n"
            << "Check a defective frame followed by a good frame... "
            << "\n******************************************************\n"
            << std::endl;

  // load image
  unsigned int *in_img = 0;
  int rows_new, cols_new;

  std::string canonical_input_bmp_path =  //
      input_bmp_filename + DEFAULT_EXTENSION;
  std::string canonical_expected_bmp_path =  //
      expected_bmp_filename + DEFAULT_EXTENSION;

  std::cout << "Reading input image " << canonical_input_bmp_path << std::endl;
  if (!BmpTools::read_bmp(canonical_input_bmp_path, &in_img, rows_new,
                          cols_new)) {
    std::cerr << "ERROR: Could not read image from " << canonical_input_bmp_path
              << std::endl;
    return false;
  }

  size_t rows = 0;
  size_t cols = 0;
  bool imageOk = updateAndCheckImageDimensions(rows, cols, rows_new, cols_new);

  if (!imageOk) {
    std::cerr << "ERROR: invalid image size " << rows << " x " << cols
              << std::endl;
  }

  int endPixel = rows * cols / 2;

  // Enqueue a defective frame that ends after `endPixel` pixels.
  conv2d::Pixel_rgb *in_img_vvp = new conv2d::Pixel_rgb[rows * cols];
  conv2d::Pixel_rgb *out_img_vvp = new conv2d::Pixel_rgb[rows * cols];
  unsigned int *out_img = new unsigned int[rows * cols];

  convertToVvpRgb(in_img, in_img_vvp, rows * cols);

  vvp_stream_adapters::writeFrameToPipe<InputImageStream>(q, rows, cols,
                                                          in_img_vvp, endPixel);

  // Now enqueue a good frame.
  vvp_stream_adapters::writeFrameToPipe<InputImageStream>(q, rows, cols,
                                                          in_img_vvp);

  int dummyPixels = cols * conv2d::kWindowSize;
  vvp_stream_adapters::writeDummyPixelsToPipe<InputImageStream>(
      q, dummyPixels, conv2d::Pixel_rgb{32, 32, 32});

  // Enqueue the kernel. Run it until we have read out the partial frame and
  // good frame, then stop.
  sycl::event frameEvent;
  q.single_task<ID_RGB2Grey>(
      RGB2Grey<InputImageStream, InputImageStreamGrey>{});
  frameEvent = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, sobel_coeffs});
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  std::string defect_output_bmp_path =
      output_bmp_filename_base + "_defect" + DEFAULT_EXTENSION;

  initializeBuffer(out_img_vvp, rows * cols);

  std::cout << "\n****************************\n"  //
            << "Read out good frame "              //
            << "(defective frame overwritten)" << std::endl;

  std::cout << "Expect some unexpected start-of-packets as the kernel "
               "flushes its initial state, and the defective frame."
            << std::endl;

  bool sidebands_ok = false;
  int badFrames = 0;
  if (out_img_vvp) {
    vvp_stream_adapters::readFrameFromPipe<OutputImageStream>(
        q, rows, cols, out_img_vvp, sidebands_ok, badFrames);
  }
  if (out_img) {
    convertToBmpRgb(out_img_vvp, out_img, rows * cols);
    BmpTools::write_bmp(defect_output_bmp_path, out_img, rows, cols);
    std::cout << "Wrote convolved image " << defect_output_bmp_path
              << std::endl;
  } else {
    std::cerr << "ERROR: could not write output image: out_img=null."
              << std::endl;
  }

  // This should succeed since the defective pixels were overwritten by the
  // subsequent good frame.
  bool passed =
      BmpTools::compareFrames(out_img, rows, cols, canonical_expected_bmp_path);

  bool allPassed = passed & sidebands_ok;
  printf("frame 'defect' %s\n", (passed && sidebands_ok) ? "passed" : "failed");

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  frameEvent.wait();

  delete[] out_img;
  delete[] out_img_vvp;
  delete[] in_img_vvp;

  return allPassed;
}

#endif

int main(int argc, char **argv) {
  try {
    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    // make sure the device supports USM host allocations
    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // image files
    std::string input_bmp_filename = M_DEFAULT_INPUT;
    std::string output_bmp_filename = M_DEFAULT_OUTPUT;
    std::string expected_bmp_filename = M_DEFAULT_EXPECTED;

    bool allPassed = true;

#if TEST_CONV2D_ISOLATED
    allPassed &= testTinyFrameOnStencil(q);
#else
    BypassCSR::write(q, false);
    allPassed &=
        testGoodFramesSequence(q, NUM_FRAMES, input_bmp_filename,
                               output_bmp_filename, expected_bmp_filename);

    BypassCSR::write(q, false);
    allPassed &=
        testDefectiveFrame(q, input_bmp_filename + "_0", output_bmp_filename,
                           expected_bmp_filename + "_0");
#endif

    std::cout << "\nOverall result:\t" << (allPassed ? "PASSED" : "FAILED")
              << std::endl;
    return EXIT_SUCCESS;

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code.
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}