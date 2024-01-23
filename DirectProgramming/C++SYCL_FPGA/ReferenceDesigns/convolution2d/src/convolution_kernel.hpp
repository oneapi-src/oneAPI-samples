#pragma once
#include <stdint.h>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "convolution_types.hpp"
#include "data_bundle.hpp"
#include "linebuffer2d.hpp"
#include "unrolled_loop.hpp"

/////////////////////////////////////////////
// Define input/output streaming interfaces
/////////////////////////////////////////////

class ID_InStr;
using InputImgStreamProperties =
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::bits_per_symbol<8>,
        sycl::ext::intel::experimental::uses_valid<true>,
        sycl::ext::intel::experimental::ready_latency<0>,
        sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>));
using InputImageStream =
    sycl::ext::intel::experimental::pipe<ID_InStr, conv2d::RGBBeat, 0,
                                         InputImgStreamProperties>;

class ID_InStrGrey;
using InputImageStreamGrey =
    sycl::ext::intel::experimental::pipe<ID_InStrGrey, conv2d::GreyScaleBeat, 0,
                                         InputImgStreamProperties>;

class ID_OutStr;
using OutputImgStreamProperties =
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::bits_per_symbol<8>,
        sycl::ext::intel::experimental::uses_valid<true>,
        sycl::ext::intel::experimental::ready_latency<0>,
        sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>));
using OutputImageStreamGrey =
    sycl::ext::intel::experimental::pipe<ID_OutStr, conv2d::GreyScaleBeat, 0,
                                         OutputImgStreamProperties>;

using OutputImageStream =
    sycl::ext::intel::experimental::pipe<ID_OutStr, conv2d::RGBBeat, 0,
                                         OutputImgStreamProperties>;

class ID_StopCSR;
using StopCSRProperties = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::protocol<
        sycl::ext::intel::experimental::protocol_name::avalon_mm>));
using StopCSR = sycl::ext::intel::experimental::pipe<ID_StopCSR, bool, 0,
                                                     StopCSRProperties>;

class ID_BypassCSR;
using BypassCSR = sycl::ext::intel::experimental::pipe<ID_BypassCSR, bool, 0,
                                                       StopCSRProperties>;

constexpr int NormalizeFactor = 1 << conv2d::kBitsPerChannel;

/// @brief Handle pixels at the edge of the input image by reflecting them.
/// @param[in] sRow current row in stencil
/// @param[in] sCol current column in stencil
/// @param[in] row row coordinate of pixel in the center of the stencil
/// @param[in] col column coordinate of pixel in the center of the stencil
/// @param[in] rows total rows in input image
/// @param[in] cols total columns in input image
/// @param[out] rWindow row of stencil to select
/// @param[out] cWindow column of stencil to select
void saturateWindowCoordinates(short sRow, short sCol, short row, short col,
                               short rows, short cols, short &rWindow,
                               short &cWindow) {
  // logic to deal with image borders: border pixel duplication
  rWindow = sRow;
  int rDiff = sRow - (conv2d::kWindowSize / 2) + row;
  if (rDiff < 0) {
    rWindow = (conv2d::kWindowSize / 2) - row;
  }
  if (rDiff >= rows) {
    rWindow = (conv2d::kWindowSize / 2) + ((rows - 1) - row);
  }

  cWindow = sCol;
  int cDiff = sCol - (conv2d::kWindowSize / 2) + col;
  if (cDiff < 0) {
    cWindow = (conv2d::kWindowSize / 2) - col;
  }
  if (cDiff >= cols) {
    cWindow = (conv2d::kWindowSize / 2) + ((cols - 1) - col);
  }
}

/// @brief Window function that performs a 2D Convolution in a stencil framework
/// @param row y-coordinate of pixel at the center of the stencil window
/// @param col x-coordinate of pixel at the center of the stencil window
/// @param rows total rows in input image
/// @param cols total columns in input image
/// @param buffer Window of pixels from input image
/// @param coefficients
/// @return pixel value to stream out
conv2d::PixelType convolutionFunction(
    short row, short col, short rows, short cols, conv2d::PixelType *buffer,
    const std::array<conv2d::WeightType,
                     conv2d::kWindowSize * conv2d::kWindowSize>
        coefficients) {
  float sum = 0.0f;
#pragma unroll
  for (int sRow = 0; sRow < conv2d::kWindowSize; sRow++) {
#pragma unroll
    for (int sCol = 0; sCol < conv2d::kWindowSize; sCol++) {
      short cWindow, rWindow;

      // handle the case where the center of the window is at the image edge.
      // In this design, simply 'reflect' pixels that are already in the
      // window.
      saturateWindowCoordinates(sRow, sCol,  //
                                row, col,    //
                                rows, cols,  //
                                rWindow, cWindow);
      conv2d::PixelType pixel = buffer[cWindow + rWindow * conv2d::kWindowSize];

      constexpr float normalization_factor = (1 << conv2d::kBitsPerChannel);

      // converting `pixel` to a floating-point value uses lots of FPGA
      // resources. If your expected coefficients have a narrow range, it will
      // be worthwhile to convert these operations to fixed-point.
      float normalizedPixel = (float)pixel / normalization_factor;

      float normalizedWeight = coefficients[sCol + sRow * conv2d::kWindowSize];

      sum += normalizedPixel * normalizedWeight;
    }
  }

  // map range [0, 1.0) to [0, 1<<kBitsPerChannel)
  // conv2d::PixelType retVal = sum * (float)(1 << conv2d::kBitsPerChannel);

  // map range (-1.0, 1.0) to [0, 1<<kBitsPerChannel)
  constexpr int kOutputOffset = ((1 << conv2d::kBitsPerChannel) / 2);
  conv2d::PixelType retVal =
      (int16_t)kOutputOffset + (int16_t)(sum * (float)(kOutputOffset));

  return retVal;
}

//////////////////////////////////////////////////////
// Convert RGB to Grayscale for the convolution
//////////////////////////////////////////////////////
class ID_RGB2Grey;

template <typename PipeIn, typename PipeOut>
struct RGB2Grey {
  // Kernel properties method to configure the kernel to be a kernel with
  // streaming invocation interface.
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::
            streaming_interface_remove_downstream_stall};
  }
  void operator()() const {
    // this loop is not necessary in hardware since I will pin the
    // `start` bit high, but it makes the testbench easier.
    while (1) {
      conv2d::RGBBeat rgb_beat = PipeIn::read();
      conv2d::GreyScaleBeat grey_beat;

      grey_beat.empty = rgb_beat.empty;
      grey_beat.sop = rgb_beat.sop;
      grey_beat.eop = rgb_beat.eop;

#pragma unroll
      for (int i = 0; i < conv2d::kParallelPixels; i++) {
        grey_beat.data[i] = (rgb_beat.data[i].r / 4) +  //
                            (rgb_beat.data[i].g / 2) +  //
                            (rgb_beat.data[i].b / 4);   //
      }
      PipeOut::write(grey_beat);
    }
  }
};

//////////////////////////////////////////////////////
// Perform a Convolution
//////////////////////////////////////////////////////

class ID_Convolution2d;

template <typename PipeIn, typename PipeOut>
struct Convolution2d {
  // these defaults are not propagated to the RTL
  int rows = 0;
  int cols = 0;

  // These coefficients used in convolution. kernel are often called a 'kernel'
  // in the image-processing field, but that term is avoided to reduce
  // confusion. Since kernel are a kernel argument, kernel can be specified at
  // runtime. The coefficients can only be updated by stopping the kernel and
  // re-starting it.
  std::array<conv2d::WeightType, conv2d::kWindowSize * conv2d::kWindowSize>
      coeffs;

  void operator()() const {
    // This instance of the line buffer will store previously read pixels so
    // that they can be operated on in a local filter. The filter is invoked
    // below in the loop.
    linebuffer2d::LineBuffer2d<conv2d::PixelType, conv2d::PixelType,
                               conv2d::kWindowSize, conv2d::kMaxCols,
                               conv2d::kParallelPixels>
        myLineBuffer(rows, cols);

    bool keepGoing = true;
    bool bypass = true;

    [[intel::initiation_interval(1)]]  //
    while (keepGoing) {
      // do non-blocking reads so that the kernel can be interrupted at any
      // time.
      bool didReadBeat = false;
      conv2d::GreyScaleBeat newBeat = PipeIn::read(didReadBeat);

      // the bypass signal lets the user disable the line buffer processing.
      bool didReadBypass = false;
      bool shouldBypass = BypassCSR::read(didReadBypass);

      // the stop signal lets the user instruct the kernel to halt so that new
      // coefficients can be read.
      bool didReadStop = false;
      bool shouldStop = StopCSR::read(didReadStop);

      if (didReadBypass) {
        bypass = shouldBypass;
      }

      if (didReadBeat) {
        conv2d::GreyScaleBeat outputBeat;
        if (bypass) {
          outputBeat = newBeat;
        } else {
          bool sop, eop;

          // Call `filter()` function on `LineBuffer2d` object. This inserts a
          // new pixel into the linebuffer, and runs the user-provided window
          // function (`convolutionFunction()`). The additional argument
          // `coeffs` is passed to `convolutionFunction()`. The return value of
          // `filter()` is the pixel data that we should propagate on to the
          // next link in the processing chain.
          conv2d::GreyPixelBundle outputBundle =
              myLineBuffer.filter<convolutionFunction>(
                  newBeat.data, newBeat.sop, newBeat.eop, sop, eop, coeffs);
          outputBeat = conv2d::GreyScaleBeat(outputBundle, sop, eop, 0);
        }
        PipeOut::write(outputBeat);
      }

      if (didReadStop) {
        keepGoing = !shouldStop;
      }
    }
  }
};

//////////////////////////////////////////////////////
// Convert Grayscale to RGB for the display
//////////////////////////////////////////////////////
class ID_Grey2RGB;

template <typename PipeIn, typename PipeOut>
struct Grey2RGB {
  // Kernel properties method to configure the kernel to be a kernel with
  // streaming invocation interface.
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::
            streaming_interface_remove_downstream_stall};
  }
  void operator()() const {
    // this loop is not necessary in hardware since the `start` bit will be
    // pulled high, but it makes the testbench easier.
    while (1) {
      conv2d::GreyScaleBeat grey_beat = PipeIn::read();
      conv2d::RGBBeat rgb_beat;
      rgb_beat.empty = grey_beat.empty;
      rgb_beat.sop = grey_beat.sop;
      rgb_beat.eop = grey_beat.eop;

#pragma unroll
      for (int i = 0; i < conv2d::kParallelPixels; i++) {
        rgb_beat.data[i].r = grey_beat.data[i];  //
        rgb_beat.data[i].g = grey_beat.data[i];  //
        rgb_beat.data[i].b = grey_beat.data[i];  //
      }
      PipeOut::write(rgb_beat);
    }
  }
};