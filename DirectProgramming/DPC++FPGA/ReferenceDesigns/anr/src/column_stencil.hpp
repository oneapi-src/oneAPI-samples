#ifndef __COLUMN_STENCIL_HPP__
#define __COLUMN_STENCIL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "data_bundle.hpp"
#include "mp_math.hpp"
#include "shift_reg.hpp"

using namespace sycl;
using namespace hldutils;

//
// Generic 1D vertical window stencil
//
template <typename InType, typename OutType, typename IndexT, typename InPipe,
          typename OutPipe, unsigned filter_size, unsigned max_cols,
          unsigned parallel_cols, typename StencilFunction,
          typename... FunctionArgTypes>
void ColumnStencil(IndexT rows, IndexT cols, IndexT frames,
                   const InType zero_val, StencilFunction func,
                   FunctionArgTypes... stencil_args) {
  // types coming into and out of the kernel from pipes, respectively
  using InPipeT = DataBundle<InType, parallel_cols>;
  using OutPipeT = DataBundle<OutType, parallel_cols>;

  // static asserts
  constexpr int kPaddingPixels = filter_size / 2;
  constexpr int kShiftRegCols = 1 + parallel_cols - 1;
  constexpr int kShiftRegRows = filter_size;
  constexpr int kLineBufferFIFODepth =
      (max_cols / parallel_cols) + /*filter_size*/ 1;
  constexpr int kNumLineBuffers = filter_size - 1;
  constexpr IndexT kColThreshLow = kPaddingPixels;
  constexpr IndexT kRowThreshLow = kPaddingPixels;
  constexpr IndexT kRowOutputThreshLow = 2 * kPaddingPixels;

  // static asserts
  static_assert(filter_size > 1);
  static_assert(max_cols > parallel_cols);
  static_assert(parallel_cols > 0);
  static_assert(IsPow2(parallel_cols) > 0);
  static_assert(std::is_invocable_r_v<OutType, StencilFunction, int, int,
                                      ShiftReg<InType, filter_size>,
                                      FunctionArgTypes...>);

  // constants
  const IndexT row_thresh_high = kPaddingPixels + rows;
  const IndexT padded_rows = rows + 2 * kRowThreshLow;
  const IndexT fifo_wrap =
      (cols + /*filter_size*/ 1 - 1 + (parallel_cols - 1 /* round up*/)) /
      parallel_cols;
  const IndexT col_loop_bound = (cols / parallel_cols);

  [[intel::initiation_interval(1)]]
  for (IndexT frame = 0; frame < frames; frame++) {
    // the 2D shift register to store the 'kShiftRegCols' columns of size
    // 'kShiftRegRows'
    ShiftReg2d<InType, kShiftRegRows, kShiftRegCols> shifty_2d;

    // the line buffer fifo
    [[intel::fpga_memory]]
    InPipeT line_buffer_FIFO[kLineBufferFIFODepth][kNumLineBuffers];

    InPipeT last_new_pixels(zero_val);

    IndexT fifo_idx = 0;  // track top of FIFO

    // NOTE: speculated iterations here will cause a bubble, but
    // small number relative padded_rows * col_loop_bound and the
    // increase in Fmax justifies it.
    [[intel::loop_coalesce(2), intel::initiation_interval(1),
      intel::ivdep(line_buffer_FIFO)]]
    for (IndexT row = 0; row < padded_rows; row++) {
      [[intel::initiation_interval(1),
        intel::ivdep(line_buffer_FIFO)]]
      for (IndexT col_loop = 0; col_loop < col_loop_bound; col_loop++) {
        // the base column index for this iteration
        IndexT col = col_loop * parallel_cols;

        // read in values if it is time to start reading
        // (row >= kRowThreshLow) and if there are still more to read
        // (row < row_thresh_high)
        InPipeT new_pixels(zero_val);
        if ((row >= kRowThreshLow) && (row < row_thresh_high)) {
          new_pixels = InPipe::read();
        }

        InPipeT input_val(last_new_pixels);
        constexpr auto kInputShiftVals =
            Min(kColThreshLow, (IndexT)parallel_cols);
        input_val.template ShiftMultiVals<kInputShiftVals, parallel_cols>(
            new_pixels);

        [[intel::fpga_register]]
        InPipeT pixel_column[filter_size];

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

        UnrolledLoop<0, filter_size>([&](auto stencil_row) {
          if constexpr (stencil_row != (filter_size - 1)) {
            pixel_column[stencil_row] = line_buffer_FIFO[fifo_idx][stencil_row];
          } else {
            pixel_column[stencil_row] = input_val;
          }
        });
        shifty_2d.template ShiftCols<parallel_cols>(pixel_column);

        // Continue processing through FIFOs
        //      ┌─────────────┐
        //      │ FIFO        ◄───┐
        //      └─────────────┘   │
        //    ┌───────────────────┘
        //    │ ┌─────────────┐
        //    └─┤ FIFO        ◄───┐
        //      └─────────────┘   │
        //                        └─Input

        UnrolledLoop<0, (filter_size - 1)>([&](auto fifo_row) {
          if constexpr (fifo_row != (filter_size - 2)) {
            line_buffer_FIFO[fifo_idx][fifo_row] = pixel_column[fifo_row + 1];
          } else {
            line_buffer_FIFO[fifo_idx][(filter_size - 2)] = input_val;
          }
        });

        // Perform the convolution on the 1D window
        OutPipeT out_data((OutType)0);
        UnrolledLoop<0, parallel_cols>([&](auto stencil_idx) {
          ShiftReg<InType, kShiftRegRows> shifty_copy;

          int col_local = col + stencil_idx;

          UnrolledLoop<0, filter_size>([&](auto stencil_row) {
            shifty_copy[stencil_row] = shifty_2d[stencil_row][stencil_idx];
          });

          // pass a copy of the line buffer's register window.
          out_data[stencil_idx] = func((row - kRowOutputThreshLow), col_local,
                                       shifty_copy, stencil_args...);
        });

        // write the output data if it is in range (i.e., it is a real pixel
        // and not part of the padding)
        if (row >= kRowOutputThreshLow) {
          OutPipe::write(out_data);
        }

        // increment the fifo
        if (fifo_idx == (fifo_wrap - 1)) {
          fifo_idx = 0;
        } else {
          fifo_idx++;
        }
        last_new_pixels = new_pixels;
      }
    }
  }
}

#endif /* __COLUMN_STENCIL_HPP__ */