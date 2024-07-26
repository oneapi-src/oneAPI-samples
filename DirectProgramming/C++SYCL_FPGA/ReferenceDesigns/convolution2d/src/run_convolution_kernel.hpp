#pragma once
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "convolution_kernel.hpp"
#include "exception_handler.hpp"
#include "grey2rgb_kernel.hpp"
#include "image_buffer_adapters.hpp"
#include "rgb2grey_kernel.hpp"
#include "vvp_stream_adapters.hpp"

/// @brief Pass a sequence of frames through the convolution system. This system
/// is made up of 3 kernels:
///
/// 1. `RGB2Grey`
///
/// 2. `Convolution`
///
/// 3. `Grey2RGB`
/// @param[in] q Queue to assign work to
/// @param[in] frame Frame to process
/// @param[in] coeffs Coefficients to put in FIR filter
/// @param[in] print_debug_messages Set this to `true` to print debug messages
/// @param[out] all_passed This is set to `false` if something went wrong
/// @return The same input frames after they have been processed by the
/// convolution system, or an empty vector if there was an error.
void KernelProcessSingleFrame(
    sycl::queue q, const conv2d::PixelType *const in_img_vvp,
    conv2d::PixelType *grey_pixels_out, size_t rows, size_t cols,
    const std::array<float, conv2d::kWindowSize * conv2d::kWindowSize> &coeffs,
    bool &sidebands_ok, int &parsed_frames, bool print_debug_messages = false) {
  // This will cause a hang if there is too much input data (e.g. frames are
  // too big, or there are too many frames)
  vvp_stream_adapters::WriteFrameToPipe<InputImageStreamGrey>(q, rows, cols,
                                                              in_img_vvp);

  // extra pixels to flush out the line buffer in the Convolution2D kernel
  int dummy_pixels = cols * (conv2d::kWindowSize - 1);
  constexpr auto kDummyVal = conv2d::PixelType{100};
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStreamGrey,
                                              conv2d::PixelType>(
      q, dummy_pixels, kDummyVal);

  std::cout << "Launch Convolution2d kernel" << std::endl;
  sycl::event e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, coeffs});

  vvp_stream_adapters::ReadFrameFromPipe<OutputImageStreamGrey>(
      q, rows, cols, grey_pixels_out, sidebands_ok, parsed_frames,
      print_debug_messages);

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  std::cout << "NOTE: KernelProcessSingleFrame(): Stop kernel...";
  StopCSR::write(q, true);
  e.wait();
  std::cout << "done. " << std::endl;

  return;
}