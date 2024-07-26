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

  conv2d::PixelType grey_pixels_out[pixels_count];
  bool sidebands_ok;
  int parsed_frames;

  KernelProcessSingleFrame(q, grey_pixels_in, grey_pixels_out, rows_small,
                           cols_small, identity_coeffs, sidebands_ok,
                           parsed_frames);

  bool pixels_match = true;
  for (int i = 0; i < pixels_count; i++) {
    constexpr float kOutputOffset = ((1 << conv2d::kBitsPerChannel) / 2);
    constexpr float kNormalizationFactor = (1 << conv2d::kBitsPerChannel);
    conv2d::PixelType grey_pixel_expected =
        ((float)grey_pixels_in[i] / kNormalizationFactor) * kOutputOffset +
        kOutputOffset;
    pixels_match &= (grey_pixel_expected == grey_pixels_out[i]);
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

  conv2d::PixelType grey_pixels_in[] = {
      101, 201, 301, 401, 501, 601, 701, 801,  //
      102, 202, 302, 402, 502, 602, 702, 802,  //
      103, 203, 303, 403, 503, 603, 703, 803};

  // Enable 'bypass' mode by writing to CSR.
  BypassCSR::write(q, true);

  conv2d::PixelType grey_pixels_out[pixels_count];
  bool sidebands_ok;
  int parsed_frames;

  KernelProcessSingleFrame(q, grey_pixels_in, grey_pixels_out, rows_small,
                           cols_small, identity_coeffs, sidebands_ok,
                           parsed_frames);

  bool pixels_match = true;
  for (int i = 0; i < pixels_count; i++) {
    conv2d::PixelType grey_pixel_expected = grey_pixels_in[i];
    pixels_match &= (grey_pixel_expected == grey_pixels_out[i]);
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

  std::vector<bmp_tools::BitmapRGB> input_bitmaps;
  for (size_t itr = 0; itr < num_frames; itr++) {
    // load image
    std::string input_bmp_path =  //
        input_bmp_filename_base + "_" + std::to_string(itr) + DEFAULT_EXTENSION;

    std::cout << "INFO: Load image " << input_bmp_path << std::endl;
    unsigned int error;
    input_bitmaps.push_back(bmp_tools::ReadBmp(input_bmp_path, error));
    if (bmp_tools::BmpError::OK != error) {
      std::cerr << "ERROR: Could not read image from " << input_bmp_path
                << std::endl;
      return false;
    }
  }

  bool all_passed = false;
  std::vector<bmp_tools::BitmapRGB> output_bitmaps =
      SystemProcessFrameSequence(q, input_bitmaps, sobel_coeffs, all_passed);

  for (size_t itr = 0; itr < output_bitmaps.size(); itr++) {
    std::cout << "\n*********************\n"  //
              << "Reading out frame " << itr  //
              << std::endl;

    std::string absolute_output_bmp_path = output_bmp_filename_base + "_" +
                                           std::to_string(itr) +
                                           DEFAULT_EXTENSION;
    unsigned int write_error;
    bmp_tools::WriteBmp(absolute_output_bmp_path, output_bitmaps[itr],
                        write_error);

    if (bmp_tools::BmpError::OK == write_error) {
      std::cout << "Wrote convolved image " << absolute_output_bmp_path
                << std::endl;
    }

    std::string expected_bmp_path = expected_bmp_filename_base + "_" +
                                    std::to_string(itr) + DEFAULT_EXTENSION;

    std::cout << "Compare with " << expected_bmp_path << ". " << std::endl;
    unsigned int comparison_error = 0;
    bool passed = bmp_tools::CompareFrames(output_bitmaps[itr],
                                           expected_bmp_path, comparison_error);

    printf("frame %zu %s\n", itr, (passed && all_passed) ? "passed" : "failed");
  }

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
  std::string input_bmp_path =  //
      input_bmp_filename + DEFAULT_EXTENSION;
  std::string expected_bmp_path =  //
      expected_bmp_filename + DEFAULT_EXTENSION;

  std::cout << "INFO: Load image " << input_bmp_path << std::endl;
  unsigned int error;
  bmp_tools::BitmapRGB in_img = bmp_tools::ReadBmp(input_bmp_path, error);
  if (bmp_tools::BmpError::OK != error) {
    std::cerr << "ERROR: Could not read image from " << input_bmp_path
              << std::endl;
    return false;
  }

  bool passed = true;

  bool sidebands_ok = 0;
  int num_parsed_frames = 0;
  bmp_tools::BitmapRGB parsed_frame = SystemProcessFrameAndDefect(
      q, in_img, sobel_coeffs, num_parsed_frames, sidebands_ok);

  // expect the defective frame + the good frame
  if (2 != num_parsed_frames) {
    std::cerr << "ERROR: saw " << num_parsed_frames
              << " parsed frames (expected 2)." << std::endl;
    passed = false;
  }

  std::string defect_output_bmp_path =
      output_bmp_filename_base + "_defect" + DEFAULT_EXTENSION;
  unsigned int write_error;
  bmp_tools::WriteBmp(defect_output_bmp_path, parsed_frame, write_error);

  if (bmp_tools::BmpError::OK == write_error) {
    std::cout << "Wrote convolved image " << defect_output_bmp_path
              << std::endl;
  }
  // This should succeed since the defective pixels were overwritten by the
  // subsequent good frame.
  unsigned int comparison_error = 0;
  passed &= bmp_tools::CompareFrames(parsed_frame, expected_bmp_path,
                                     comparison_error);

  bool all_passed = passed & sidebands_ok;
  printf("frame 'defect' %s\n", (passed && sidebands_ok) ? "passed" : "failed");

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