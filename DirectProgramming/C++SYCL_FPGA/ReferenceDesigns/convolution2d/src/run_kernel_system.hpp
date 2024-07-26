#pragma once
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "convolution_kernel.hpp"
#include "exception_handler.hpp"
#include "grey2rgb_kernel.hpp"
#include "image_buffer_adapters.hpp"
#include "rgb2grey_kernel.hpp"
#include "vvp_stream_adapters.hpp"

/// @brief Verify image dimensions from a just-read image and compare with
/// previous image dimensions if appropriate.
/// @param rows previous image rows (0 if no previous dimensions)
/// @param cols previous image cols (0 if no previous dimensions)
/// @param rows_new new image rows
/// @param cols_new new image columns
/// @throws std::invalid_arg
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
  if (cols % conv2d::kParallelPixels != 0) {
    std::string exception_message =
        "Image cols = " + std::to_string(cols) +
        " not compatible with kernel compiled for " +
        std::to_string(conv2d::kParallelPixels) +
        " pixels in parallel. Please choose an image whose width is a multiple "
        "of " +
        std::to_string(conv2d::kParallelPixels) +
        ", or recompile with a different value of the PARALLEL_PIXELS "
        "pre-processor macro.";
    throw std::invalid_argument(exception_message);
  }
}

/// @brief Pass a sequence of frames through the convolution system. This system
/// is made up of 3 kernels:
///
/// 1. `RGB2Grey`
///
/// 2. `Convolution`
///
/// 3. `Grey2RGB`
/// @param[in] q Queue to assign work to
/// @param[in] frames Array of frames to process
/// @param[in] coeffs Coefficients to put in FIR filter
/// @param[in] print_debug_messages Set this to `true` to print debug messages
/// @param[out] all_passed This is set to `false` if something went wrong
/// @return The same input frames after they have been processed by the
/// convolution system, or an empty vector if there was an error.
std::vector<bmp_tools::BitmapRGB> SystemProcessFrameSequence(
    sycl::queue q, const std::vector<bmp_tools::BitmapRGB> &frames,
    const std::array<float, conv2d::kWindowSize * conv2d::kWindowSize> &coeffs,
    bool &all_passed, bool print_debug_messages = false) {
  size_t num_frames = frames.size();
  std::vector<bmp_tools::BitmapRGB> error_frames(num_frames,
                                                 bmp_tools::BitmapRGB(0, 0));
  if (0 == num_frames) {
    return error_frames;
  }

  size_t rows = 0;
  size_t cols = 0;

  all_passed = true;
  // fill pipe with pixels from input images
  for (size_t itr = 0; itr < num_frames; itr++) {
    UpdateAndCheckImageDimensions(rows, cols, frames[itr].GetRows(),
                                  frames[itr].GetCols());

    conv2d::PixelRGB *in_img_vvp = new conv2d::PixelRGB[rows * cols];

    ConvertToVvpRgb(frames[itr], in_img_vvp, rows * cols);

    // This will cause a hang if there is too much input data (e.g. frames are
    // too big, or there are too many frames)
    vvp_stream_adapters::WriteFrameToPipe<InputImageStream>(q, rows, cols,
                                                            in_img_vvp);

    // don't need in_img_vvp anymore
    delete[] in_img_vvp;
  }

  // extra pixels to flush out the line buffer in the Convolution2D kernel
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
  sycl::event e = q.single_task<ID_Convolution2d>(
      Convolution2d<InputImageStreamGrey, OutputImageStreamGrey>{
          (int)rows, (int)cols, coeffs});

  std::cout << "Launch Grey2RGB kernel" << std::endl;
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  std::vector<bmp_tools::BitmapRGB> processed_frames(
      num_frames, bmp_tools::BitmapRGB(rows, cols));
  for (size_t itr = 0; itr < num_frames; itr++) {
    std::cout << "\n*********************\n"  //
              << "Reading out frame " << itr  //
              << std::endl;

    conv2d::PixelRGB *out_img_vvp = new conv2d::PixelRGB[rows * cols];
    InitializeBuffer(out_img_vvp, rows * cols);

    int parsed_frames = 0;
    bool sidebands_ok = false;
    if (out_img_vvp) {
      vvp_stream_adapters::ReadFrameFromPipe<OutputImageStream>(
          q, rows, cols, out_img_vvp, sidebands_ok, parsed_frames,
          print_debug_messages);
    } else {
      std::cerr
          << "ERROR: SystemProcessFrameSequence(): could not allocate memory "
             "for output image!"
          << std::endl;
      all_passed = false;
    }

    if (1 != parsed_frames) {
      std::cerr << "WARNING: SystemProcessFrameSequence(): saw "
                << parsed_frames << " parsed frames (expected 1)." << std::endl;
    }

    processed_frames[itr] = ConvertToBmpRgb(out_img_vvp, rows, cols);
    delete[] out_img_vvp;
  }

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  std::cout << "NOTE: SystemProcessFrameSequence(): Stop kernel...";
  StopCSR::write(q, true);
  e.wait();
  std::cout << "done. " << std::endl;

  std::cout
      << "NOTE: SystemProcessFrameSequence(): Finished checking a sequence of "
         "good frames.\n\n"
      << std::endl;

  return processed_frames;
}

/// @brief Pass a sequence of frames through the convolution system. This system
/// is made up of 3 kernels:
///
/// 1. `RGB2Grey`
///
/// 2. `Convolution`
///
/// 3. `Grey2RGB`
/// @param[in] q Queue to assign work to
/// @param[in] frames Array of frames to process
/// @param[in] coeffs Coefficients to put in FIR filter
/// @param[in] print_debug_messages Set this to `true` to print debug messages
/// @param[out] all_passed This is set to `false` if something went wrong
/// @return The same input frames after they have been processed by the
/// convolution system.
bmp_tools::BitmapRGB SystemProcessFrameAndDefect(
    sycl::queue q, const bmp_tools::BitmapRGB &frame,
    const std::array<float, conv2d::kWindowSize * conv2d::kWindowSize> &coeffs,
    int &num_parsed_frames, bool &sidebands_ok,
    bool print_debug_messages = false) {
  size_t rows = frame.GetRows();
  size_t cols = frame.GetCols();

  size_t end_pixel = rows * cols / 2;

  // Enqueue a defective frame that ends after `end_pixel` pixels.
  conv2d::PixelRGB *in_img_vvp = new conv2d::PixelRGB[rows * cols];
  conv2d::PixelRGB *out_img_vvp = new conv2d::PixelRGB[rows * cols];

  ConvertToVvpRgb(frame, in_img_vvp, rows * cols);

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
          (int)rows, (int)cols, coeffs});

  std::cout << "Launch Grey2RGB kernel" << std::endl;
  q.single_task<ID_Grey2RGB>(
      Grey2RGB<OutputImageStreamGrey, OutputImageStream>{});

  InitializeBuffer(out_img_vvp, rows * cols);

  std::cout << "\n****************************\n"  //
            << "Read out defective frame, and overwrite with good frame."
            << std::endl;

  if (out_img_vvp) {
    vvp_stream_adapters::ReadFrameFromPipe<OutputImageStream>(
        q, rows, cols, out_img_vvp, sidebands_ok, num_parsed_frames,
        print_debug_messages);
  }

  bmp_tools::BitmapRGB out_img = ConvertToBmpRgb(out_img_vvp, rows, cols);
  delete[] out_img_vvp;

  // Stop the kernel in case testbench wants to run again with different kernel
  // arguments.
  StopCSR::write(q, true);
  e.wait();
  return out_img;
}