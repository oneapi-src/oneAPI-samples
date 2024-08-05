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

#ifndef TEST_CONV2D_ISOLATED
#define TEST_CONV2D_ISOLATED 0
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
void InitializeBuffer(conv2d::PixelRGB *buf, size_t size) {
  uint16_t pixel = 0;
  for (size_t i = 0; i < size; i++) {
    pixel++;
    buf[i] = conv2d::PixelRGB{pixel, pixel, pixel};
  }
}

/// @brief Convert pixels read from a bmp image using the bmptools functions to
/// pixels that can be parsed by our 2D convolution IP.
/// @param[in] bmp_buf pixels read by bmptools
/// @param[out] vvp_buf pixels to be consumed by 2D convolution IP
/// @param[in] pixel_count (input) number of pixels in input image and output
/// image
void ConvertToVvpRgb(unsigned int *bmp_buf, conv2d::PixelRGB *vvp_buf,
                     size_t pixel_count) {
  std::cout << "INFO: convert to vvp type." << std::endl;
  for (size_t idx = 0; idx < pixel_count; idx++) {
    uint32_t pixel_int = bmp_buf[idx];
    bmp_tools::PixelRGB bmp_rgb(pixel_int);

    // convert from 8-bit to whatever the VVP IP expects
    conv2d::PixelRGB pixel_vvp{
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
void ConvertToBmpRgb(conv2d::PixelRGB *vvp_buf, unsigned int *bmp_buf,
                     size_t pixel_count) {
  std::cout << "INFO: convert to bmp type." << std::endl;
  for (size_t idx = 0; idx < pixel_count; idx++) {
    conv2d::PixelRGB pixel_conv = vvp_buf[idx];

    // convert the VVP IP back to 8-bit
    bmp_tools::PixelRGB bmp_rgb(
        (uint8_t)(pixel_conv.r >> (conv2d::kBitsPerChannel - 8)),   //
        (uint8_t)(pixel_conv.g >> (conv2d::kBitsPerChannel - 8)),   //
        (uint8_t)(pixel_conv.b >> (conv2d::kBitsPerChannel - 8)));  //

    uint32_t pixel_int = bmp_rgb.GetImgPixel();
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
bool UpdateAndCheckImageDimensions(size_t &rows, size_t &cols, size_t rows_new,
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

  // Max allowable value for rows * cols must be less than the max value of a
  // signed 32-bit integer.
  constexpr int kRowsColsMax = 1 << 29;

  bool image_size_ok =
      (rows_new > 0) && (cols_new > 0) && (rows_new * cols_new < kRowsColsMax);

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

  return image_size_ok;
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
/// @param print_debug_info Print additional debug information when reading from
/// pipe
/// @return `true` if successful, `false` otherwise
bool TestTinyFrameOnStencil(sycl::queue q, bool print_debug_info) {
  std::cout << "\n**********************************\n"
            << "Check Tiny frame... "
            << "\n**********************************\n"
            << std::endl;
  constexpr int rows_small = 3;
  constexpr int cols_small = 8;

  constexpr int pixels_count = rows_small * cols_small;

  conv2d::PixelType grey_pixels_in[] = {
      101, 201, 301, 401, 501, 601, 701, 801,  //
      102, 202, 302, 402, 502, 602, 702, 802,  //
      103, 203, 303, 403, 503, 603, 703, 803};

  vvp_stream_adapters::WriteFrameToPipe<InputImageStreamGrey>(
      q, rows_small, cols_small, grey_pixels_in);

  // add extra pixels to flush out the FIFO after all image frames
  // have been added
  int dummy_pixels = cols_small * conv2d::kWindowSize;
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStreamGrey>(
      q, dummy_pixels, (uint16_t)15);

  sycl::event e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows_small, (int)cols_small, identity_coeffs});

  conv2d::PixelType grey_pixels_out[pixels_count];
  bool sidebands_ok;
  int parsed_frames;
  vvp_stream_adapters::ReadFrameFromPipe<OutputImageStreamGrey>(
      q, rows_small, cols_small, grey_pixels_out, sidebands_ok, parsed_frames,
      print_debug_info);

  bool pixels_match = true;
  for (int i = 0; i < pixels_count; i++) {
    constexpr float kOutputOffset = ((1 << conv2d::kBitsPerChannel) / 2);
    constexpr float kNormalizationFactor = (1 << conv2d::kBitsPerChannel);
    conv2d::PixelType grey_pixel_expected =
        ((float)grey_pixels_in[i] / kNormalizationFactor) * kOutputOffset +
        kOutputOffset;
    pixels_match &= (grey_pixel_expected == grey_pixels_out[i]);
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  e.wait();

  return sidebands_ok & pixels_match;
}

/// @brief Test that the 'bypass' control works correctly.
/// @param q The SYCL queue to assign work to
/// @param[in] print_debug_info Print additional debug information when reading
/// from pipe
/// @return `true` if input image matches output image
bool TestBypass(sycl::queue q, bool print_debug_info) {
  std::cout << "\n**********************************\n"
            << "Check bypass... "
            << "\n**********************************\n"
            << std::endl;

  constexpr int rows_small = 3;
  constexpr int cols_small = 8;

  constexpr int pixels_count = rows_small * cols_small;

  conv2d::PixelType grey_pixels_in[] = {
      101, 201, 301, 401, 501, 601, 701, 801,  //
      102, 202, 302, 402, 502, 602, 702, 802,  //
      103, 203, 303, 403, 503, 603, 703, 803};

  vvp_stream_adapters::WriteFrameToPipe<InputImageStreamGrey>(
      q, rows_small, cols_small, grey_pixels_in);

  // add extra pixels to flush out the FIFO after all image frames
  // have been added
  int dummy_pixels = cols_small * conv2d::kWindowSize;
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStreamGrey>(
      q, dummy_pixels, (uint16_t)15);

  // Enable 'bypass' mode by writing to CSR.
  BypassCSR::write(q, true);

  sycl::event e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows_small, (int)cols_small, identity_coeffs});

  conv2d::PixelType grey_pixels_out[pixels_count];
  bool sidebands_ok;
  int parsed_frames;
  vvp_stream_adapters::ReadFrameFromPipe<OutputImageStreamGrey>(
      q, rows_small, cols_small, grey_pixels_out, sidebands_ok, parsed_frames,
      print_debug_info);

  bool pixels_match = true;
  for (int i = 0; i < pixels_count; i++) {
    conv2d::PixelType grey_pixel_expected = grey_pixels_in[i];
    pixels_match &= (grey_pixel_expected == grey_pixels_out[i]);
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  e.wait();

  return sidebands_ok & pixels_match;
}

#else

constexpr std::array<float, 9> sobel_coeffs = {
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f,  //
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f,  //
    -1.0f / 6.0f, 0.0f, 1.0f / 6.0f   //
};

/// @brief 'Happy Path' test that repeatedly passes a known good frame through
/// the IP.
/// @param[in] q SYCL queue
/// @param[in] num_frames Number of times to pass the image frame through the IP
/// @param[in] input_bmp_filename_base base filename to use for reading input
/// frames
/// @param[in] output_bmp_filename_base base filename to use for writing output
/// files
/// @param[in] expected_bmp_filename_base 'known good' file to compare IP output
/// against
/// @param[in] print_debug_messages Pass to the `vvp_stream_adapters`
/// functions to print debug information.
/// @return `true` if all frames emitted by the IP match the `known good` file,
/// `false` otherwise.
bool TestGoodFramesSequence(sycl::queue q, size_t num_frames,
                            std::string input_bmp_filename_base,
                            std::string output_bmp_filename_base,
                            std::string expected_bmp_filename_base,
                            bool print_debug_messages) {
  std::cout << "\n**********************************\n"
            << "Check a sequence of good frames... "
            << "\n**********************************\n"
            << std::endl;

  sycl::event e;
  bool all_passed = true;

  size_t rows = 0, cols = 0;

  for (size_t itr = 0; itr < num_frames; itr++) {
    // load image
    unsigned int *in_img = 0;
    int rows_new, cols_new;

    std::string canonical_input_bmp_path =  //
        input_bmp_filename_base + "_" + std::to_string(itr) + DEFAULT_EXTENSION;
    std::string canonical_expected_bmp_path =  //
        expected_bmp_filename_base + "_" + std::to_string(itr) +
        DEFAULT_EXTENSION;

    std::cout << "INFO: Load image " << canonical_input_bmp_path << std::endl;
    if (!bmp_tools::ReadBmp(canonical_input_bmp_path, &in_img, rows_new,
                            cols_new)) {
      std::cerr << "ERROR: Could not read image from "
                << canonical_input_bmp_path << std::endl;
      return false;
    }

    bool image_ok =
        UpdateAndCheckImageDimensions(rows, cols, rows_new, cols_new);

    if (!image_ok) {
      std::cerr << "ERROR: invalid image size " << rows << " x " << cols
                << std::endl;
      continue;
    }

    conv2d::PixelRGB *in_img_vvp = new conv2d::PixelRGB[rows * cols];

    ConvertToVvpRgb(in_img, in_img_vvp, rows * cols);

    // don't need in_img anymore
    free(in_img);

    vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(q, rows, cols,
                                                            in_img_vvp);

    // don't need in_img_vvp anymore
    delete[] in_img_vvp;
  }

  // extra pixels to flush out the FIFO
  int dummy_pixels = cols * (conv2d::kWindowSize - 1);
  constexpr auto kDummyVal = conv2d::PixelRGB{100, 100, 100};
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStream,
                                              conv2d::PixelRGB>(q, dummy_pixels,
                                                                kDummyVal);

  std::cout << "\n*********************" << std::endl;
  std::cout << "Launch RGB2Grey kernel" << std::endl;
  q.single_task<ID_RGB2Grey>(
      RGB2Grey<InputImageStream, InputImageStreamGrey>{});

  std::cout << "Launch Convolution2d kernel" << std::endl;
  e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, sobel_coeffs});

  std::cout << "Launch Grey2RGB kernel" << std::endl;
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  for (size_t itr = 0; itr < num_frames; itr++) {
    std::cout << "\n*********************\n"  //
              << "Reading out frame " << itr  //
              << std::endl;

    std::string absolute_output_bmp_path = output_bmp_filename_base + "_" +
                                           std::to_string(itr) +
                                           DEFAULT_EXTENSION;

    conv2d::PixelRGB *out_img_vvp = new conv2d::PixelRGB[rows * cols];
    unsigned int *out_img = new unsigned int[rows * cols];
    InitializeBuffer(out_img_vvp, rows * cols);

    int parsed_frames = 0;
    bool sidebands_ok = false;
    if (out_img_vvp) {
      vvp_stream_adapters::ReadFrameFromPipe<OutputImageStream>(
          q, rows, cols, out_img_vvp, sidebands_ok, parsed_frames,
          print_debug_messages);
    }

    if (1 != parsed_frames) {
      std::cerr << "ERROR: saw " << parsed_frames
                << " parsed frames (expected 1)." << std::endl;
    }

    if (out_img) {
      ConvertToBmpRgb(out_img_vvp, out_img, rows * cols);
      delete[] out_img_vvp;

      bmp_tools::WriteBmp(absolute_output_bmp_path, out_img, rows, cols);
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
    bool passed = bmp_tools::CompareFrames(out_img, rows, cols,
                                           absolute_expected_bmp_path);

    delete[] out_img;
    all_passed &= passed & sidebands_ok & (1 == parsed_frames);
    printf("frame %zu %s\n", itr,
           (passed && sidebands_ok) ? "passed" : "failed");
  }

  int detected_version = VersionCSR::read(q);
  std::cout << "\nKernel version = " << detected_version << " (Expected "
            << kKernelVersion << ")" << std::endl;

  if (detected_version != kKernelVersion) {
    std::cerr << "ERROR: kernel version did not match!" << std::endl;
    all_passed = false;
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  e.wait();

  std::cout << "\nFinished checking a sequence of good frames.\n\n"
            << std::endl;

  return all_passed;
}

/// @brief Test how the IP handles a defective frame by passing a defective
/// frame (which should emit a partial image) followed by a complete frame
/// which should match the 'known good' file.
/// @param[in] q SYCL queue
/// @param[in] input_bmp_filename Buffer containing the image frame to process
/// @param[in] output_bmp_filename_base File to output the processed frame to
/// @param[in] expected_bmp_filename 'known good' file to compare IP output
/// against
/// @param[in] print_debug_messages Pass to the `vvp_stream_adapters`
/// functions to print debug information.
/// @return `true` if the second frame emitted by the IP matches the `known
/// good` file, `false` otherwise.
bool TestDefectiveFrame(sycl::queue q, std::string input_bmp_filename,
                        std::string output_bmp_filename_base,
                        std::string expected_bmp_filename,
                        bool print_debug_messages = false) {
  std::cout << "\n******************************************************\n"
            << "Check a defective frame followed by a good frame... "
            << "\n******************************************************\n"
            << std::endl;

  // load image
  unsigned int *in_img = nullptr;
  int rows_new, cols_new;

  std::string canonical_input_bmp_path =  //
      input_bmp_filename + DEFAULT_EXTENSION;
  std::string canonical_expected_bmp_path =  //
      expected_bmp_filename + DEFAULT_EXTENSION;

  std::cout << "Reading input image " << canonical_input_bmp_path << std::endl;
  if (!bmp_tools::ReadBmp(canonical_input_bmp_path, &in_img, rows_new,
                          cols_new)) {
    std::cerr << "ERROR: Could not read image from " << canonical_input_bmp_path
              << std::endl;
    return false;
  }

  size_t rows = 0;
  size_t cols = 0;
  bool image_ok = UpdateAndCheckImageDimensions(rows, cols, rows_new, cols_new);

  if (!image_ok) {
    std::cerr << "ERROR: invalid image size " << rows << " x " << cols
              << std::endl;
  }

  int end_pixel = rows * cols / 2;

  // Enqueue a defective frame that ends after `end_pixel` pixels.
  conv2d::PixelRGB *in_img_vvp = new conv2d::PixelRGB[rows * cols];
  conv2d::PixelRGB *out_img_vvp = new conv2d::PixelRGB[rows * cols];
  unsigned int *out_img = new unsigned int[rows * cols];

  ConvertToVvpRgb(in_img, in_img_vvp, rows * cols);

  vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(
      q, rows, cols, in_img_vvp, end_pixel);

  // Now enqueue a good frame.
  vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(q, rows, cols,
                                                          in_img_vvp);

  int dummy_pixels = cols * conv2d::kWindowSize;
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStream>(
      q, dummy_pixels, conv2d::PixelRGB{32, 32, 32});

  // Enqueue the kernel. Run it until we have read out the partial frame and
  // good frame, then stop.
  sycl::event e;

  std::cout << "\n*********************" << std::endl;
  std::cout << "Launch RGB2Grey kernel" << std::endl;
  q.single_task<ID_RGB2Grey>(
      RGB2Grey<InputImageStream, InputImageStreamGrey>{});

  std::cout << "Launch Convolution2d kernel" << std::endl;
  e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, sobel_coeffs});

  std::cout << "Launch Grey2RGB kernel" << std::endl;
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  std::string defect_output_bmp_path =
      output_bmp_filename_base + "_defect" + DEFAULT_EXTENSION;

  InitializeBuffer(out_img_vvp, rows * cols);

  std::cout << "\n****************************\n"  //
            << "Read out defective frame, and overwrite with good frame."
            << std::endl;

  bool sidebands_ok = false;
  int parsed_frames = 0;
  if (out_img_vvp) {
    vvp_stream_adapters::ReadFrameFromPipe<OutputImageStream>(
        q, rows, cols, out_img_vvp, sidebands_ok, parsed_frames,
        print_debug_messages);
  }
  bool passed = true;

  // expect the defective frame + the good frame
  if (2 != parsed_frames) {
    std::cerr << "ERROR: saw " << parsed_frames
              << " parsed frames (expected 2)." << std::endl;
    passed = false;
  }

  if (out_img) {
    ConvertToBmpRgb(out_img_vvp, out_img, rows * cols);
    bmp_tools::WriteBmp(defect_output_bmp_path, out_img, rows, cols);
    std::cout << "Wrote convolved image " << defect_output_bmp_path
              << std::endl;
  } else {
    std::cerr << "ERROR: could not write output image: out_img=null."
              << std::endl;
  }

  // This should succeed since the defective pixels were overwritten by the
  // subsequent good frame.
  passed &= bmp_tools::CompareFrames(out_img, rows, cols,
                                     canonical_expected_bmp_path);

  bool all_passed = passed & sidebands_ok;
  printf("frame 'defect' %s\n", (passed && sidebands_ok) ? "passed" : "failed");

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  e.wait();

  delete[] out_img;
  delete[] out_img_vvp;
  delete[] in_img_vvp;

  return all_passed;
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

    bool all_passed = true;

#if TEST_CONV2D_ISOLATED
    all_passed &= TestTinyFrameOnStencil(q, false);
    all_passed &= TestBypass(q, false);
#else
    all_passed &= TestGoodFramesSequence(q, NUM_FRAMES, input_bmp_filename,
                                         output_bmp_filename,
                                         expected_bmp_filename, false);
    all_passed &=
        TestDefectiveFrame(q, input_bmp_filename + "_0", output_bmp_filename,
                           expected_bmp_filename + "_0", false);
#endif

    std::cout << "\nOverall result:\t" << (all_passed ? "PASSED" : "FAILED")
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