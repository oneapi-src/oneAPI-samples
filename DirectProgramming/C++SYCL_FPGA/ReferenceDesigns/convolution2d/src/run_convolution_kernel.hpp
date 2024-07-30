#pragma once
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "convolution_kernel.hpp"
#include "exception_handler.hpp"
#include "vvp_stream_adapters.hpp"

class ID_InStrGrey;
using InputImageStreamGrey =
    sycl::ext::intel::experimental::pipe<ID_InStrGrey, vvp_gray::GrayScaleBeat,
                                         0, InputImgStreamProperties>;

class ID_OutStr;
using OutputImageStreamGrey =
    sycl::ext::intel::experimental::pipe<ID_OutStr, vvp_gray::GrayScaleBeat, 0,
                                         OutputImgStreamProperties>;
namespace single_kernel {

/// @brief Pass a single frame through the Convolution2d kernel.
/// @param[in] q Queue to assign work to
/// @param[in] in_img_vvp Frame to process
/// @param[in] coeffs Coefficients to put in FIR filter
/// @param[out] sidebands_ok Set to `true` if sideband signals were correct
/// @param[out] parsed_frames Number of times that `start of packet` was seen
/// before a complete frame was seen. Ideally this value should be `1`.
/// @param[in] print_debug_messages Set this to `true` to print debug messages
/// @return A container of grayscale pixels
vvp_gray::ImageGrey KernelProcessSingleFrame(
    sycl::queue q, const vvp_gray::ImageGrey &in_img_vvp,
    const std::array<float, kWindowSize * kWindowSize> &coeffs,
    bool &sidebands_ok, int &parsed_frames, bool print_debug_messages = false) {
  // This will cause a hang if there is too much input data (e.g. frames are
  // too big, or there are too many frames)
  vvp_stream_adapters::WriteFrameToPipe<InputImageStreamGrey>(q, in_img_vvp);

  int cols = in_img_vvp.GetCols();
  int rows = in_img_vvp.GetRows();

  // extra pixels to flush out the line buffer in the Convolution2D kernel
  int dummy_pixels = cols * (kWindowSize - 1);
  constexpr auto kDummyVal = vvp_gray::PixelGray{100};
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStreamGrey,
                                              vvp_gray::PixelGray>(
      q, dummy_pixels, kDummyVal);

  std::cout << "Launch Convolution2d kernel" << std::endl;
  sycl::event e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, coeffs});

  vvp_gray::ImageGrey out_img =
      vvp_stream_adapters::ReadFrameFromPipe<OutputImageStreamGrey,
                                             vvp_gray::PixelGray>(
          q, rows, cols, sidebands_ok, parsed_frames, print_debug_messages);

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  std::cout << "INFO: KernelProcessSingleFrame(): Write 'true' to StopCSR...";
  StopCSR::write(q, true);
  e.wait();
  std::cout << "Kernel stopped. " << std::endl;

  return out_img;
}
}  // namespace single_kernel