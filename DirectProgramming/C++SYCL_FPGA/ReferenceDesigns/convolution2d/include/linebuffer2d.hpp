//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// line_buffer_2d.hpp

#pragma once

#include <array>

#include "comparisons.hpp"
#include "shift_reg.hpp"

namespace line_buffer_2d {

#pragma pack(push, 1)
// struct type to wrap a pixel type with sop and eop signals. Could use the
// StreamingBeat type but this results in compilation failures due to
// https://hsdes.intel.com/appstore/article/#/18034529956.
template <typename PixelTypeIn>
struct PixelWithSignals_ {
  PixelTypeIn val;
  bool sop;
  bool eop;
  int empty;
};
#pragma pack(pop)

template <typename PixelTypeIn, typename PixelTypeOut, short kStencilSize,
          short kMaxImgCols, short kParallelPixels>
class LineBuffer2d {
 public:
  // types used by LineBuffer2d
  using LineBufferDataBundleIn = std::array<PixelTypeIn, kParallelPixels>;
  using LineBufferDataBundleOut = std::array<PixelTypeOut, kParallelPixels>;

  // public members
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  short rows;

  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  short cols;

 private:
  // types used internally
  using PixelWithSignals = PixelWithSignals_<PixelTypeIn>;
  using BundledPixels = std::array<PixelWithSignals, kParallelPixels>;
  constexpr static short kRowWriteInit = (short)(0 - kStencilSize);
  constexpr static short kColWriteInit = (short)(0 - kStencilSize);

  ///////////////////////////////
  // initialize state variables
  ///////////////////////////////
  // infer parameterization of FIFO and register windows
  constexpr static int kShifterCols = kStencilSize + kParallelPixels - 1;

  // line buffer
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  fpga_tools::ShiftReg2d<PixelWithSignals, kStencilSize, kShifterCols>
      my_shifter;

  constexpr static int kFifoCols =
      (kMaxImgCols / kParallelPixels) + kStencilSize;
  constexpr static int kFifoRows = kStencilSize - 1;

  // Line buffer for image Data
  [[intel::fpga_memory]]  // NO-FORMAT: Attribute
  BundledPixels line_buffer_fifo[kFifoCols][kFifoRows];

  short fifo_idx = 0;  // track top of FIFO
  short fifo_wrap = cols / kParallelPixels;

  // If we have multiple pixels in parallel, we need to insert some dummy
  // pixels before the 'real data' so the output has the same alignment as
  // the input.
  constexpr static short kBufferOffset = fpga_tools::Min(
      (short)((kParallelPixels - (short)(kStencilSize / 2))), kParallelPixels);

  constexpr static short kPreBufferSize = kParallelPixels + kBufferOffset;

  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  fpga_tools::ShiftReg<PixelWithSignals, kPreBufferSize>
      pre_buffer;

  // separate the loop bound calculation so loop iterations are easier to
  // compute.
  const short col_loop_bound = (cols / kParallelPixels);

  short row_write = kRowWriteInit;
  short col_loop = 0;

  // Shannonize col_write variable to get better fMAX
  short col_write = kColWriteInit;
  short col_write_next = kColWriteInit + kParallelPixels;

  bool eop_curr = false;
  bool sop_curr = false;
  int empty = 0, empty_curr = 0;

 public:
  LineBuffer2d(short m_rows, short m_cols) : rows(m_rows), cols(m_cols) {}

  /// @brief Callback function used by `Filter` function. This function must
  /// accept the following parameters:
  /// @param[in] rows The total number of rows in the image being processed
  /// @param[in] cols The total number of columns in the image being processed
  /// @param[in] row The row of the pixel at the centre of `pixels`.
  /// @param[in] col The column of the pixel at the centre of `pixels`.
  /// @param[in] pixels An array containing the pixels in the window. The centre
  /// of the window corresponds to location (`row`, `col`) in the image.
  template <typename... FunctionArgs_T>
  using WindowFunction = PixelTypeOut (*)(
      short rows, short cols, short row, short col,
      fpga_tools::ShiftReg2d<PixelTypeIn, kStencilSize, kStencilSize> pixels,
      FunctionArgs_T... window_fn_args);

  /// @brief This function structures a sliding-window stencil function in a way
  /// that is optimal for FPGAs. `Filter()` inserts a new pixel into the
  /// line buffer, and runs the user-provided window function. Any additional
  /// arguments you provide this function will be passed to your window function
  /// (in addition to the mandatory arguments it must accept). The return value
  /// of `Filter()` is the concatenation of the results of all the copies of
  /// your window function.
  ///
  /// @paragraph Schematic
  ///```                                           <br/>
  ///                   ┌─────────────┐            <br/>
  ///               ┌───┤ FIFO        ◄───┐        <br/>
  /// ┌───┬───┬───┐ │   └─────────────┘   │        <br/>
  /// │ r ◄ e ◄ g ◄─┘ ┌───────────────────┘        <br/>
  /// ├───┼───┼───┤   │ ┌─────────────┐            <br/>
  /// │ r ◄ e ◄ g ◄───┴─┤ FIFO        ◄───┐        <br/>
  /// ├───┼───┼───┤     └─────────────┘   │        <br/>
  /// │ r ◄ e ◄ g ◄───────────────────────┴─Input  <br/>
  /// └───┴───┴───┘                                <br/>
  ///```                                           <br/>
  /// @tparam window_function operation to perform on window. Must be a template
  /// parameter rather than a function pointer due to SYCL.
  /// @param[in] new_pixels input data
  /// @param[in] is_new_frame Set this to `true` if the pixel(s) you pass in
  /// `new_pixels` is/are at the start of a frame.
  /// @param[in] is_line_end Set this to `true` if the pixel(s) you pass in
  /// `new_pixels` is/are at the end of a line.
  /// @param[out] start_of_frame This is set to `true` if the returned pixel(s)
  /// is/are at the start of a new frame.
  /// @param[out] end_of_line This is set to `true` if the returned pixel(s)
  /// is/are at the end of a line.
  /// @return Filter result
  template <auto(&window_function), typename... FunctionArgs_T>
  LineBufferDataBundleOut Filter(LineBufferDataBundleIn new_pixels,
                                 bool is_new_frame, bool is_line_end,
                                 bool &start_of_frame, bool &end_of_line,
                                 FunctionArgs_T... window_fn_args) {
    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    BundledPixels new_pixels_structs;

#pragma unroll
    for (int i = 0; i < kParallelPixels; i++) {
      // wrap each pixel value in a struct
      PixelTypeIn new_pixel = new_pixels[i];

      [[intel::fpga_register]]  // NO-FORMAT: Attribute
      PixelWithSignals pixel_struct{new_pixel, is_new_frame, is_line_end,
                                    empty};
      new_pixels_structs[i] = pixel_struct;
    }

    pre_buffer.template ShiftMultiVals<kParallelPixels>(new_pixels_structs);

    // grab the first `kParallelPixels` samples to push into the stencil
    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    BundledPixels input_val;
#pragma unroll
    for (int i = 0; i < kParallelPixels; i++) {
      input_val[i] = pre_buffer[i];
    }

    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    BundledPixels pixel_column[kStencilSize];

    // load from FIFO to shift register
    //
    //                   ┌───────────
    // ┌───┬───┬───┐ ┌───┤ FIFO
    // │ r ◄─e ◄─g ◄─┘   └───────────
    // ├───┼───┼───┤     ┌───────────
    // │ r ◄─e ◄─g ◄─────┤ FIFO
    // ├───┼───┼───┤     └───────────
    // │ r ◄─e ◄─g ◄─────────────────Input
    // └───┴───┴───┘

    // using UnrolledLoop enables if constexpr
    fpga_tools::UnrolledLoop<kStencilSize>([&](auto stencil_row) {
      if constexpr (stencil_row == (kStencilSize - 1)) {
        pixel_column[stencil_row] = input_val;
      } else {
        pixel_column[stencil_row] = line_buffer_fifo[fifo_idx][stencil_row];
      }
    });

    // use this fix:
    // http://www.aerialmantis.co.uk/blog/2017/03/17/template-keywords/
    my_shifter.template ShiftCols<kParallelPixels>(pixel_column);

    // Continue processing through FIFOs
    //    ┌─────────────┐
    //    │ FIFO        ◄───┐
    //    └─────────────┘   │
    //  ┌───────────────────┘
    //  │ ┌─────────────┐
    //  └─┤ FIFO        ◄───┐
    //    └─────────────┘   │
    //                      └─Input

    // using UnrolledLoop enables if constexpr
    fpga_tools::UnrolledLoop<(kStencilSize - 1)>([&](auto fifo_row) {
      if constexpr (fifo_row != (kStencilSize - 2)) {
        line_buffer_fifo[fifo_idx][fifo_row] = pixel_column[fifo_row + 1];
      } else {
        line_buffer_fifo[fifo_idx][(kStencilSize - 2)] = input_val;
      }
    });

    // Get the sop and eop signals from the pixel currently being processed
    constexpr int kStencilCenter = (kStencilSize - 1) / 2;

    sop_curr = my_shifter[kStencilCenter][kStencilCenter].sop;
    eop_curr = my_shifter[kStencilCenter][kStencilCenter].eop;
    empty_curr = my_shifter[kStencilCenter][kStencilCenter].empty;

    // SOP=1 corresponds with the current pixel having Row=0 and col=0.
    // EOP=1 corresponds with the NEXT pixel having row=row+1 and col=0;
    if (sop_curr) {
      row_write = 0;
      col_write = 0;
      col_write_next = kParallelPixels;
    }

    LineBufferDataBundleOut window_results;

#pragma unroll
    for (int stencil_idx = 0; stencil_idx < kParallelPixels; stencil_idx++) {
      PixelTypeIn shifter_copy[kStencilSize * kStencilSize];

      short col_local = (col_write + stencil_idx);

      // Use the pixels in my_shifter to generate a window to pass to
      // window_function. This is where we could add some customization to
      // change the 'edge' behaviour (e.g. what do we do when we are at column
      // 0). That is probably up to the user to decide though, so we just copy
      // pixels from my_shifter and let window_function decide how to handle
      // edges.

#pragma unroll
      for (int stencil_row = 0; stencil_row < kStencilSize; stencil_row++) {
#pragma unroll
        for (int stencilCol = 0; stencilCol < kStencilSize; stencilCol++) {
          shifter_copy[stencilCol + stencil_row * kStencilSize] =
              my_shifter[stencil_row][stencilCol + stencil_idx].val;
        }
      }

      // in-line this function on a copy of the appropriate shifter data
      PixelTypeOut window_result = window_function(
          row_write, col_local, rows, cols, shifter_copy, window_fn_args...);
      window_results[stencil_idx] = window_result;
    }

    if ((row_write >= 0) && (col_write >= 0)) {
      start_of_frame = sop_curr;
      end_of_line = eop_curr;
    } else {
      start_of_frame = false;
      end_of_line = false;
    }

    fifo_idx++;
    if (fifo_idx == (fifo_wrap)) {
      // Possible optimization: make the reset of fifo_idx depend on previous EOP read from input to
      // remove need to compare with fifo_wrap.
      fifo_idx = (short)0;  // Reset Index
    }

    // update loop counter variables
    col_loop++;
    if (col_loop == col_loop_bound) {
      col_loop = 0;
    }

    // shannonize col_write variable to improve fMAX. This lets us break up the
    // the accumulate and the comparison operation to occur on separate loop
    // iterations.
    col_write = col_write_next;
    col_write_next += kParallelPixels;

    // reset col_write and row_write when SOP and EOP appear
    if (col_write >= cols) {
      col_write = 0;
      col_write_next = kParallelPixels;
      row_write++;
    }

    return window_results;
  }
};
}  // namespace line_buffer_2d