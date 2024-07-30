#pragma once
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "colorspace_conversion_kernels.hpp"
#include "convolution_kernel.hpp"
#include "exception_handler.hpp"
#include "vvp_stream_adapters.hpp"

// top-level pipes (host pipes)

class ID_InStrRGB;
using InputImageStream =
    sycl::ext::intel::experimental::pipe<ID_InStrRGB, vvp_rgb::RGBBeat, 0,
                                         InputImgStreamProperties>;

class ID_OutStrRGB;
using OutputImageStream =
    sycl::ext::intel::experimental::pipe<ID_OutStrRGB, vvp_rgb::RGBBeat, 0,
                                         OutputImgStreamProperties>;

// inter-kernel pipes

class ID_InStrGrey;
using InputImageStreamGrey =
    sycl::ext::intel::experimental::pipe<ID_InStrGrey, vvp_gray::GrayScaleBeat,
                                         0>;

class ID_OutStrGray;
using OutputImageStreamGrey =
    sycl::ext::intel::experimental::pipe<ID_OutStrGray, vvp_gray::GrayScaleBeat,
                                         0>;
namespace kernel_system {

/// @brief Verify image dimensions from a just-read image and compare with
/// previous image dimensions if appropriate.
/// @param rows previous image rows (0 if no previous dimensions)
/// @param cols previous image cols (0 if no previous dimensions)
/// @param rows_new new image rows
/// @param cols_new new image columns
/// @throws std::invalid_argument if argument checks fail
void UpdateAndCheckImageDimensions(size_t &rows, size_t &cols, size_t rows_new,
                                   size_t cols_new) {
  // sanity check: all images should be the same size
  if (rows == 0)
    rows = rows_new;
  else if (rows != rows_new) {
    std::string exception_message =
        "Dimensions of subsequent images must match. Expected " +
        std::to_string(rows) + " but saw " + std::to_string(rows_new);
    throw std::invalid_argument(exception_message);
  }

  if (cols == 0)
    cols = cols_new;
  else if (cols != cols_new) {
    std::string exception_message =
        "Dimensions of subsequent images must match. Expected " +
        std::to_string(cols) + " but saw " + std::to_string(cols_new);
    throw std::invalid_argument(exception_message);
  }

  // Max allowable value for rows * cols must be less than the max value of a
  // signed 32-bit integer.
  constexpr int kRowsColsMax = 1 << 29;

  bool image_size_ok =
      (rows_new > 0) && (cols_new > 0) && (rows_new * cols_new < kRowsColsMax);

  if (!image_size_ok) {
    std::string exception_message =
        "Invalid image dimensions. rows_new=" + std::to_string(rows_new) +
        " cols_new=" + std::to_string(cols_new) +
        " rows_new * cols_new=" + std::to_string(rows_new * cols_new);
    throw std::invalid_argument(exception_message);
  }

  // sanity check; this design assumes that the number of columns in the input
  // image is a multiple of kParallelPixels.
  if (cols % vvp_rgb::kParallelPixels != 0) {
    std::string exception_message =
        "Image cols = " + std::to_string(cols) +
        " not compatible with kernel compiled for " +
        std::to_string(vvp_rgb::kParallelPixels) +
        " pixels in parallel. Please choose an image whose width is a multiple "
        "of " +
        std::to_string(vvp_rgb::kParallelPixels) +
        ", or recompile with a different value of the PARALLEL_PIXELS "
        "pre-processor macro.";
    throw std::invalid_argument(exception_message);
  }
}

/// @brief Invoke a system of kernels to perform 2D convolution. This system is
/// made up of 3 kernels:
///
/// 1. `RGB2Grey` (rgb2grey_kernel.hpp)
///
/// 2. `Convolution` (convolution_kernel.hpp)
///
/// 3. `Grey2RGB` (grey2rgb_kernel.hpp)
///
/// @attention This function reads input data written into the
/// `InputImageStream` pipe and writes output data to `OutputImageStream` pipe.
/// You can use the convenience functions in `vvp_stream_adapters.hpp` to fill
/// up `InputImageStream` and empty `OutputImageStream`.
/// @attention Since the
/// @param q SYCL queue to assign work to
/// @param rows Number of frame rows (height)
/// @param cols Number of frame columns (width)
/// @param coeffs Coefficients to put in filter
/// @return a `sycl::event` that you can use to monitor the status of the
/// Convolution kernel. See `convolution_kernel.hpp` for details of this kernel.
sycl::event LaunchKernelSystem(
    sycl::queue q, int rows, int cols,
    const std::array<float, kWindowSize * kWindowSize> &coeffs) {
  std::cout << "\n*********************" << std::endl;
  std::cout << "Launch RGB2Grey kernel" << std::endl;
  q.single_task<ID_RGB2Grey>(
      RGB2Grey<InputImageStream, InputImageStreamGrey>{});

  std::cout << "Launch Convolution2d kernel" << std::endl;
  sycl::event e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, coeffs});

  std::cout << "Launch Grey2RGB kernel" << std::endl;
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  return e;
}

/// @brief Pass a sequence of frames through the convolution system, as defined
/// in the function `LaunchKernelSystem()`.
/// @param[in] q SYCL queue to assign work to
/// @param[in] frames `std::vector` of frames to process
/// @param[in] coeffs Coefficients to put in FIR filter
/// @param[in] print_debug_messages Set this to `true` to print detailed debug
/// messages for each frame. Set to `false` (default) to only print detailed
/// debug messages for error frames.
/// @return The same input frames after they have been processed by the
/// convolution system, or an empty `std::vector` if there was an error.
std::vector<vvp_rgb::ImageRGB> SystemProcessFrameSequence(
    sycl::queue q, const std::vector<vvp_rgb::ImageRGB> &frames,
    const std::array<float, kWindowSize * kWindowSize> &coeffs,
    bool print_debug_messages = false) {
  size_t num_frames = frames.size();
  std::vector<vvp_rgb::ImageRGB> error_frames(num_frames,
                                              vvp_rgb::ImageRGB(0, 0));
  if (0 == num_frames) {
    return error_frames;
  }

  size_t rows = 0;
  size_t cols = 0;

  // fill pipe with pixels from input images
  for (size_t itr = 0; itr < num_frames; itr++) {
    UpdateAndCheckImageDimensions(rows, cols, frames[itr].GetRows(),
                                  frames[itr].GetCols());

    // This will cause a hang if there is too much input data (e.g. frames are
    // too big, or there are too many frames)
    vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(q, frames[itr]);
  }

  // extra pixels to flush out the line buffer in the Convolution2D kernel
  int dummy_pixels = cols * (kWindowSize - 1);
  constexpr auto kDummyVal = vvp_rgb::PixelRGB{100, 100, 100};
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStream,
                                              vvp_rgb::PixelRGB>(
      q, dummy_pixels, kDummyVal);

  sycl::event e = LaunchKernelSystem(q, rows, cols, coeffs);

  std::vector<vvp_rgb::ImageRGB> processed_frames;
  for (size_t itr = 0; itr < num_frames; itr++) {
    std::cout << "\n*********************\n"  //
              << "Reading out frame " << itr  //
              << std::endl;

    int parsed_frames = 0;
    bool sidebands_ok = false;
    vvp_rgb::ImageRGB out_img =
        vvp_stream_adapters::ReadFrameFromPipe<OutputImageStream,
                                               vvp_rgb::PixelRGB>(
            q, rows, cols, sidebands_ok, parsed_frames, print_debug_messages);

    if (1 != parsed_frames) {
      std::cerr << "WARNING: SystemProcessFrameSequence(): saw "
                << parsed_frames << " parsed frames (expected 1)." << std::endl;
    }

    processed_frames.push_back(out_img);
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  std::cout << "INFO: KernelProcessSingleFrame(): Write 'true' to StopCSR...";
  StopCSR::write(q, true);
  e.wait();
  std::cout << "Kernel stopped. " << std::endl;

  std::cout
      << "INFO: SystemProcessFrameSequence(): Finished checking a sequence of "
         "good frames.\n\n"
      << std::endl;

  return processed_frames;
}

/// @brief Pass a defective frame and a 'correct' frame through the convolution
/// system, as defined in the function `LaunchKernelSystem()`.
/// @param[in] q SYCL queue to assign work to
/// @param[in] frame Frame to process
/// @param[in] coeffs Coefficients to put in FIR filter
/// @param[out] sidebands_ok Set to `true` if sideband signals were correct
/// @param[out] num_parsed_frames Number of times that `start of packet` was seen
/// before a complete frame was seen. Ideally this value should be `1`.
/// @param[in] print_debug_messages Set this to `true` to print detailed debug
/// messages for each frame. Set to `false` (default) to only print detailed
/// debug messages for error frames.
/// @return A container containing RGB pixels as defined in `rgb_pixels.hpp`.
/// Should be the 'correct' frame.
vvp_rgb::ImageRGB SystemProcessFrameAndDefect(
    sycl::queue q, const vvp_rgb::ImageRGB &frame,
    const std::array<float, kWindowSize * kWindowSize> &coeffs,
    bool &sidebands_ok, int &num_parsed_frames,
    bool print_debug_messages = false) {
  size_t rows = frame.GetRows();
  size_t cols = frame.GetCols();

  size_t end_pixel = rows * cols / 2;

  vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(q, frame, end_pixel);

  // Now enqueue a good frame.
  vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(q, frame);

  // extra pixels to flush out the line buffer in the Convolution2D kernel
  int dummy_pixels = cols * (kWindowSize - 1);
  constexpr auto kDummyVal = vvp_rgb::PixelRGB{100, 100, 100};
  vvp_stream_adapters::WriteDummyPixelsToPipe<InputImageStream,
                                              vvp_rgb::PixelRGB>(
      q, dummy_pixels, kDummyVal);

  // Enqueue the kernels. Run it until we have read out the partial frame and
  // good frame, then stop.
  sycl::event e = LaunchKernelSystem(q, rows, cols, coeffs);

  std::cout << "\n****************************\n"  //
            << "Read out defective frame, and overwrite with good frame."
            << std::endl;

  vvp_rgb::ImageRGB frame_out =
      vvp_stream_adapters::ReadFrameFromPipe<OutputImageStream,
                                             vvp_rgb::PixelRGB>(
          q, rows, cols, sidebands_ok, num_parsed_frames, print_debug_messages);

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  std::cout << "INFO: KernelProcessSingleFrame(): Write 'true' to StopCSR...";
  StopCSR::write(q, true);
  e.wait();
  std::cout << "Kernel stopped. " << std::endl;
  return frame_out;
}
}  // namespace kernel_system