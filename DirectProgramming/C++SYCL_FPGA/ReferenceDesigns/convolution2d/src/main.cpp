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
#include "exception_handler.hpp"
#include "image_buffer_adapters.hpp"
#include "vvp_stream_adapters.hpp"

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

#if TEST_CONV2D_ISOLATED
#include "run_convolution_kernel.hpp"
#else
#include "run_kernel_system.hpp"
#endif

#define M_DEFAULT_INPUT DEFAULT_INPUT
#define M_DEFAULT_OUTPUT DEFAULT_OUTPUT
#define M_DEFAULT_EXPECTED DEFAULT_EXPECTED

#define ERR_MSG_BUF_SIZE 256

/////////////////////
// Test subroutines
/////////////////////
#if TEST_CONV2D_ISOLATED

constexpr std::array<float, 9> identity_coeffs = {
    0.0f, 0.0f, 0.0f,  //
    0.0f, 1.0f, 0.0f,  //
    0.0f, 0.0f, 0.0f   //
};

/// @brief Trivial test that exercises Convolution2d on its own, using an
/// extremely simple image. This is useful for debugging how data flows through
/// the line buffer.
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

  vvp_gray::PixelGray grey_pixels[] = {
      101, 201, 301, 401, 501, 601, 701, 801,  //
      102, 202, 302, 402, 502, 602, 702, 802,  //
      103, 203, 303, 403, 503, 603, 703, 803};

  vvp_gray::ImageGrey frame_in(rows_small, cols_small);
  for (int idx = 0; idx < pixels_count; idx++) {
    frame_in(idx) = grey_pixels[idx];
  }

  bool sidebands_ok;
  int parsed_frames;

  auto frame_out = single_kernel::KernelProcessSingleFrame(
      q, frame_in, identity_coeffs, sidebands_ok, parsed_frames);

  bool pixels_match = true;
  for (int itr = 0; itr < pixels_count; itr++) {
    constexpr float kOutputOffset = ((1 << vvp_rgb::kBitsPerChannel) / 2);
    constexpr float kNormalizationFactor = (1 << vvp_rgb::kBitsPerChannel);
    vvp_gray::PixelGray grey_pixel_expected =
        ((float)grey_pixels[itr] / kNormalizationFactor) * kOutputOffset +
        kOutputOffset;
    pixels_match &= (grey_pixel_expected == frame_out(itr));
  }

  return sidebands_ok & pixels_match & (parsed_frames == 1);
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

  vvp_gray::PixelGray grey_pixels[] = {
      101, 201, 301, 401, 501, 601, 701, 801,  //
      102, 202, 302, 402, 502, 602, 702, 802,  //
      103, 203, 303, 403, 503, 603, 703, 803};

  vvp_gray::ImageGrey frame_in(rows_small, cols_small);
  for (int idx = 0; idx < pixels_count; idx++) {
    frame_in(idx) = grey_pixels[idx];
  }

  // Enable 'bypass' mode by writing to CSR.
  BypassCSR::write(q, true);

  bool sidebands_ok;
  int parsed_frames;

  vvp_gray::ImageGrey frame_out = single_kernel::KernelProcessSingleFrame(
      q, frame_in, identity_coeffs, sidebands_ok, parsed_frames);

  bool pixels_match = true;
  for (int i = 0; i < pixels_count; i++) {
    vvp_gray::PixelGray grey_pixel_expected = grey_pixels[i];
    pixels_match &= (grey_pixel_expected == frame_out(i));
  }

  return sidebands_ok & pixels_match & (parsed_frames == 1);
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
/// @param[in] input_bmp_filenames Bitmap files containing image frames to
/// process
/// @param[in] expected_bmp_filenames 'known good' files to compare output
/// against
/// @param[in] output_path Path where result bitmaps can be saved
/// @param[in] print_debug_messages Pass to the `vvp_stream_adapters`
/// functions to print debug information.
/// @return `true` if all frames emitted by the IP match the `known good` file,
/// `false` otherwise.
/// @throw This function throws an `std::invalid_argument` exception if the
/// sizes of `input_bmp_filenames` and expected_bmp_filenames` do not match.
bool TestGoodFramesSequence(sycl::queue q,
                            std::vector<std::string> input_bmp_filenames,
                            std::vector<std::string> expected_bmp_filenames,
                            std::string output_path,
                            bool print_debug_messages = false) {
  std::cout << "\n**********************************\n"
            << "Check a sequence of good frames... "
            << "\n**********************************\n"
            << std::endl;

  if (input_bmp_filenames.size() != expected_bmp_filenames.size()) {
    throw std::invalid_argument(
        "input_bmp_filenames and expected_bmp_filenames must be the same "
        "size!");
  }

  int num_frames = input_bmp_filenames.size();

  std::vector<vvp_rgb::ImageRGB> input_frames;
  for (size_t itr = 0; itr < num_frames; itr++) {
    // load image
    std::string input_bmp_path = input_bmp_filenames[itr];

    std::cout << "INFO: Load image " << input_bmp_path << std::endl;
    unsigned int error;
    bmp_tools::BitmapRGB input_bitmap =
        bmp_tools::ReadBmp(input_bmp_path, error);
    if (bmp_tools::BmpError::OK != error) {
      std::cerr << "ERROR: Could not read image from " << input_bmp_path
                << std::endl;
      return false;
    }

    input_frames.push_back(ConvertToVvpRgb(input_bitmap));
  }

  std::vector<vvp_rgb::ImageRGB> output_frames =
      kernel_system::SystemProcessFrameSequence(q, input_frames, sobel_coeffs);

  bool all_passed = true;

  for (size_t itr = 0; itr < output_frames.size(); itr++) {
    bmp_tools::BitmapRGB output_bmp = ConvertToBmpRgb(output_frames[itr]);

    // generate a name to output to
    std::string output_bmp_path = output_path + "output_" +
                                  std::to_string(itr) + "." +
                                  bmp_tools::kFileExtension;
    unsigned int write_error;
    bmp_tools::WriteBmp(output_bmp_path, output_bmp, write_error);

    if (bmp_tools::BmpError::OK == write_error) {
      std::cout << "Wrote convolved image " << output_bmp_path << std::endl;
    }

    std::string expected_bmp_path = expected_bmp_filenames[itr];

    std::cout << "Compare with " << expected_bmp_path << ". " << std::endl;
    unsigned int comparison_error = 0;
    bmp_tools::BitmapRGB output_frame_bmp = ConvertToBmpRgb(output_frames[itr]);
    bool passed = bmp_tools::CompareFrames(output_frame_bmp, expected_bmp_path,
                                           comparison_error);

    all_passed &= passed;

    std::cout << "frame " << itr << " " << ((passed) ? "passed" : "failed")
              << std::endl;
  }

  // Read the kernel version from its Control/Status Register
  int detected_version = VersionCSR::read(q);
  std::cout << "\nKernel version = " << detected_version << " (Expected "
            << kKernelVersion << ")" << std::endl;

  if (detected_version != kKernelVersion) {
    std::cerr << "ERROR: kernel version did not match!" << std::endl;
    all_passed = false;
  }

  std::cout << "\nFinished checking a sequence of good frames.\n\n"
            << std::endl;

  return all_passed;
}

/// @brief Test how the IP handles a defective frame by passing a defective
/// frame (which should emit a partial image) followed by a complete frame
/// which should match the 'known good' file.
/// @param[in] q SYCL queue
/// @param[in] input_bmp_filename Full path to a bitmap file containing the
/// image frame to process
/// @param[in] expected_bmp_filename Full path to 'known good' bitmap file to
/// compare output against
/// @param[in] output_path Full path to a directory where processed frame can be
/// output to
/// @param[in] print_debug_messages Pass to the `vvp_stream_adapters`
/// functions to print debug information.
/// @return `true` if the second frame emitted by the IP matches the `known
/// good` file, `false` otherwise.
bool TestDefectiveFrame(sycl::queue q, std::string input_bmp_filename,
                        std::string expected_bmp_filename,
                        std::string output_path,
                        bool print_debug_messages = false) {
  std::cout << "\n******************************************************\n"
            << "Check a defective frame followed by a good frame... "
            << "\n******************************************************\n"
            << std::endl;

  std::cout << "INFO: Load image " << input_bmp_filename << std::endl;
  unsigned int error;
  bmp_tools::BitmapRGB in_img = bmp_tools::ReadBmp(input_bmp_filename, error);
  if (bmp_tools::BmpError::OK != error) {
    std::cerr << "ERROR: Could not read image from " << input_bmp_filename
              << std::endl;
    return false;
  }
  vvp_rgb::ImageRGB vvp_rgb = ConvertToVvpRgb(in_img);

  bool passed = true;

  bool sidebands_ok = 0;
  int num_parsed_frames = 0;
  vvp_rgb::ImageRGB parsed_frame = kernel_system::SystemProcessFrameAndDefect(
      q, vvp_rgb, sobel_coeffs, sidebands_ok, num_parsed_frames);

  // expect the defective frame + the good frame
  if (2 != num_parsed_frames) {
    std::cerr << "ERROR: saw " << num_parsed_frames
              << " parsed frames (expected 2)." << std::endl;
    passed = false;
  }

  bmp_tools::BitmapRGB output_bmp = ConvertToBmpRgb(parsed_frame);

  std::string defect_output_bmp_path =
      output_path + "output_defect." + bmp_tools::kFileExtension;
  unsigned int write_error;
  bmp_tools::WriteBmp(defect_output_bmp_path, output_bmp, write_error);

  if (bmp_tools::BmpError::OK == write_error) {
    std::cout << "Wrote convolved image " << defect_output_bmp_path
              << std::endl;
  }
  // This should succeed since the defective pixels were overwritten by the
  // subsequent good frame.
  unsigned int comparison_error = 0;
  passed &= bmp_tools::CompareFrames(output_bmp, expected_bmp_filename,
                                     comparison_error);

  bool all_passed = passed & sidebands_ok;
  std::cout << "frame 'defect' "
            << ((passed && sidebands_ok) ? "passed" : "failed") << std::endl;

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
    std::vector<std::string> input_bmp_filenames = {
        "../test_bitmaps/test_0.bmp", "../test_bitmaps/test_1.bmp",
        "../test_bitmaps/test_2.bmp", "../test_bitmaps/test_3.bmp",
        "../test_bitmaps/test_4.bmp"};
    std::string output_path = "./";
    std::vector<std::string> expected_sobel_filenames = {
        "../test_bitmaps/expected_sobel_0.bmp",
        "../test_bitmaps/expected_sobel_1.bmp",
        "../test_bitmaps/expected_sobel_2.bmp",
        "../test_bitmaps/expected_sobel_3.bmp",
        "../test_bitmaps/expected_sobel_4.bmp"};

    bool all_passed = true;

#if TEST_CONV2D_ISOLATED
    all_passed &= TestTinyFrameOnStencil(q, false);
    all_passed &= TestBypass(q, false);
#else
    all_passed &= TestGoodFramesSequence(
        q, input_bmp_filenames, expected_sobel_filenames, output_path, false);
    all_passed &=
        TestDefectiveFrame(q, input_bmp_filenames[0],
                           expected_sobel_filenames[0], output_path, false);
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