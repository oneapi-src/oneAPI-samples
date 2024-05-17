//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// convolution_kernel.hpp

#pragma once
#include <stdint.h>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "convolution_types.hpp"
#include "linebuffer2d.hpp"
#include "unrolled_loop.hpp"

// The kernel version may be polled by an Avalon memory-mapped host that manages
// this IP. Update this when you make changes to kernel code.
constexpr int kKernelVersion = 1;

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

/////////////////////////////////////////////
// Define CSR locations that attach to pipes
/////////////////////////////////////////////
using CsrInProperties = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::protocol<
        // The `ready` signal is required for input CSR pipe. The host may check
        // the `ready` register to ensure the kernel is ready for a new value.
        sycl::ext::intel::experimental::protocol_name::avalon_mm_uses_ready>,
    // Enabling the `valid` signal ensures that the kernel only consumes new
    // data after the host changes the CSR. The host must write a `1` to the
    // associated `...CHANNEL_VALID_REG` register.
    sycl::ext::intel::experimental::uses_valid_on));

using CsrOutProperties = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::protocol<
        // Disable the `ready` signal so the device can update this value
        // without requiring the host to consent. Host doesn't care about
        // possibly missing an update.
        sycl::ext::intel::experimental::protocol_name::avalon_mm>,
    // `valid` signal is required for output CSR pipe. Host may check it to
    // ensure the data is 'real'.
    sycl::ext::intel::experimental::uses_valid_on));

class ID_StopCSR;
using StopCSR =
    sycl::ext::intel::experimental::pipe<ID_StopCSR, bool, 0, CsrInProperties>;

class ID_BypassCSR;
using BypassCSR = sycl::ext::intel::experimental::pipe<ID_BypassCSR, bool, 0,
                                                       CsrInProperties>;

class ID_VersionCSR;
using VersionCSR = sycl::ext::intel::experimental::pipe<ID_VersionCSR, int, 0,
                                                        CsrOutProperties>;

/// @brief Handle pixels at the edge of the input image by reflecting them.
/// @param[in] w_row current row in window
/// @param[in] w_col current column in window
/// @param[in] row row coordinate of pixel in the center of the window
/// @param[in] col column coordinate of pixel in the center of the window
/// @param[in] rows total rows in input image
/// @param[in] cols total columns in input image
/// @param[out] r_select row of window to select
/// @param[out] c_select column of window to select
void SaturateWindowCoordinates(short w_row, short w_col, short row, short col,
                               short rows, short cols, short &r_select,
                               short &c_select) {
  // saturate in case the input image is sized incorrectly
  if (row >= rows) {
    row = (rows - 1);
  }
  if (col >= cols) {
    col = (cols - 1);
  }

  // logic to deal with image borders: border pixel duplication
  r_select = w_row;
  int rDiff = w_row - (conv2d::kWindowSize / 2) + row;
  if (rDiff < 0) {
    r_select = (conv2d::kWindowSize / 2) - row;
  }
  if (rDiff >= rows) {
    r_select = (conv2d::kWindowSize / 2) + ((rows - 1) - row);
  }

  c_select = w_col;
  int cDiff = w_col - (conv2d::kWindowSize / 2) + col;
  if (cDiff < 0) {
    c_select = (conv2d::kWindowSize / 2) - col;
  }
  if (cDiff >= cols) {
    c_select = (conv2d::kWindowSize / 2) + ((cols - 1) - col);
  }
}

/// @brief Window function that performs a 2D Convolution in a line buffer
/// framework
/// @param row y-coordinate of pixel at the center of the window
/// @param col x-coordinate of pixel at the center of the window
/// @param rows total rows in input image
/// @param cols total columns in input image
/// @param buffer Window of pixels from input image
/// @param coefficients Array of coefficients to use for convolution
/// @return pixel value to stream out
conv2d::PixelType ConvolutionFunction(
    short row, short col, short rows, short cols, conv2d::PixelType *buffer,
    const std::array<float, conv2d::kWindowSize * conv2d::kWindowSize>
        coefficients) {
  float sum = 0.0f;
#pragma unroll
  for (int w_row = 0; w_row < conv2d::kWindowSize; w_row++) {
#pragma unroll
    for (int w_col = 0; w_col < conv2d::kWindowSize; w_col++) {
      short c_select, r_select;

      // handle the case where the center of the window is at the image edge.
      // In this design, simply 'reflect' pixels that are already in the
      // window.
      SaturateWindowCoordinates(w_row, w_col,  // NO-FORMAT: Alignment
                                row, col,      // NO-FORMAT: Alignment
                                rows, cols,    // NO-FORMAT: Alignment
                                r_select, c_select);
      conv2d::PixelType pixel =
          buffer[c_select + r_select * conv2d::kWindowSize];

      constexpr float kNormalizationFactor = (1 << conv2d::kBitsPerChannel);

      // converting `pixel` to a floating-point value uses lots of FPGA
      // resources. If your expected coefficients have a narrow range, it will
      // be worthwhile to convert these operations to fixed-point.
      float normalized_pixel = (float)pixel / kNormalizationFactor;

      float normalized_coeff =
          coefficients[w_col + w_row * conv2d::kWindowSize];

      sum += normalized_pixel * normalized_coeff;
    }
  }

  // map range [0, 1.0) to [0, 1<<kBitsPerChannel)
  // conv2d::PixelType return_val = sum * (float)(1 << conv2d::kBitsPerChannel);

  // map range (-1.0, 1.0) to [0, 1<<kBitsPerChannel)
  constexpr float kOutputOffset = ((1 << conv2d::kBitsPerChannel) / 2);
  conv2d::PixelType return_val =
      ((int16_t)kOutputOffset + (int16_t)(sum * (kOutputOffset)));

  return return_val;
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
    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    while (1) {
      conv2d::RGBBeat rgb_beat = PipeIn::read();
      conv2d::GreyScaleBeat grey_beat;

      grey_beat.empty = rgb_beat.empty;
      grey_beat.sop = rgb_beat.sop;
      grey_beat.eop = rgb_beat.eop;

#pragma unroll
      for (int i = 0; i < conv2d::kParallelPixels; i++) {
        grey_beat.data[i] = (rgb_beat.data[i].r / 4) +  // NO-FORMAT: Alignment
                            (rgb_beat.data[i].g / 2) +  // NO-FORMAT: Alignment
                            (rgb_beat.data[i].b / 4);   // NO-FORMAT: Alignment
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
  std::array<float, conv2d::kWindowSize * conv2d::kWindowSize> coeffs;

  void operator()() const {
    // Publish kernel version so that other IPs can poll it
    VersionCSR::write(kKernelVersion);

    // This instance of the line buffer will store previously read pixels so
    // that they can be operated on in a local filter. The filter is invoked
    // below in the loop.
    line_buffer_2d::LineBuffer2d<conv2d::PixelType, conv2d::PixelType,
                                 conv2d::kWindowSize, conv2d::kMaxCols,
                                 conv2d::kParallelPixels>
        myLineBuffer(rows, cols);

    bool keep_going = true;
    bool bypass = false;

    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    while (keep_going) {
      // do non-blocking reads so that the kernel can be interrupted at any
      // time.
      bool did_read_beat = false;
      conv2d::GreyScaleBeat new_beat = PipeIn::read(did_read_beat);

      // the bypass signal lets the user disable the line buffer processing.
      bool did_read_bypass = false;
      bool should_bypass = BypassCSR::read(did_read_bypass);

      // the stop signal lets the user instruct the kernel to halt so that new
      // coefficients can be read.
      bool did_read_stop = false;
      bool should_stop = StopCSR::read(did_read_stop);

      if (did_read_bypass) {
        bypass = should_bypass;
      }

      if (did_read_beat) {
        conv2d::GreyScaleBeat output_beat;
        if (bypass) {
          output_beat = new_beat;
        } else {
          bool sop, eop;

          // Call `Filter()` function on `LineBuffer2d` object. This inserts a
          // new pixel into the line buffer, and runs the user-provided window
          // function (`ConvolutionFunction()`). The additional argument
          // `coeffs` is passed to `ConvolutionFunction()`. The return value of
          // `Filter()` is the pixel data that we should propagate on to the
          // next link in the processing chain.
          conv2d::GreyPixelBundle output_bundle =
              myLineBuffer.Filter<ConvolutionFunction>(
                  new_beat.data, new_beat.sop, new_beat.eop, sop, eop, coeffs);
          output_beat = conv2d::GreyScaleBeat(output_bundle, sop, eop, 0);
        }
        PipeOut::write(output_beat);
      }

      if (did_read_stop) {
        keep_going = !should_stop;
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
    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
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