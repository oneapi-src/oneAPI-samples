#ifndef __LINE_KERNEL_HPP__
#define __LINE_KERNEL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "data_bundle.hpp"
#include "shift_reg.hpp"

using namespace sycl;
using namespace hldutils;

//
// Kernel to perform the horizontal filtering
// TODO: implement this
//
template<typename KernelId, typename LineType, typename LineOutputType,
         typename IndexT, typename InPipe, typename OutPipe,
         int line_size, int parallel_cols=1,
         typename LineFunction,
         typename... FunctionArgs_T>
event SubmitLineKernel(queue& q, IndexT rows, IndexT cols,
                       const LineType zero_val, LineFunction func,
                       FunctionArgs_T... stencil_args) {
  using InPipeT = DataBundle<LineType, parallel_cols>;
  using OutPipeT = DataBundle<LineOutputType, parallel_cols>;

  constexpr int kShiftRegSize           = line_size + parallel_cols - 1;
  constexpr int kPaddingPixels          = line_size / 2;
  constexpr IndexT kRowThreshLow        = kPaddingPixels;
  constexpr IndexT kColThreshLow        = kPaddingPixels;
  constexpr IndexT kRowOutputThreshLow  = 2 * kPaddingPixels;

  const IndexT row_thresh_high = kPaddingPixels + rows;
  const IndexT col_thresh_high = kPaddingPixels + cols;
  const IndexT padded_rows = rows + 2 * kRowThreshLow;
  //const IndexT padded_cols = cols + 2 * kColThreshLow;

  //const IndexT col_loop_bound = CeilDiv(padded_cols, (IndexT)parallel_cols);
  constexpr IndexT loop_bound_remainder =
    ((2 * kColThreshLow) / parallel_cols) + (((2 * kColThreshLow) % parallel_cols) != 0);
  const IndexT col_loop_bound = (cols / parallel_cols) + loop_bound_remainder;

  return q.submit([&](handler &rows) {
    rows.single_task<KernelId>([=] {
      // the shift register
      ShiftReg<LineType, kShiftRegSize> shifty_pixels;

      [[intel::loop_coalesce(2), intel::initiation_interval(1),
        intel::speculated_iterations(0)]]
      for (IndexT row = 0; row < padded_rows; row++) {
        [[intel::initiation_interval(1), intel::speculated_iterations(0)]]
        for (IndexT col_loop = 0; col_loop < col_loop_bound; col_loop++) {
          IndexT col = col_loop * parallel_cols;
          InPipeT new_pixels(zero_val);

          // read from the input pipe if in range
          if ((row >= kRowThreshLow) && (row < row_thresh_high) &&
            (col < cols)) {
            new_pixels = InPipe::read();
          }

          // shift in the input pixels
          shifty_pixels.shiftMultiVals(new_pixels);

          // Perform the convolution
          OutPipeT out_data(LineOutputType(0));
          
          UnrolledLoop<0, parallel_cols>([&](auto stencil_idx) {
            const int col_local = col + stencil_idx;
            ShiftReg<LineType, line_size> shifty_pixels_copy;

            // first, make an offsetted copy of the shift register
            UnrolledLoop<0, line_size>([&](auto x) {
              shifty_pixels_copy[x] = shifty_pixels[x + stencil_idx];
            });

            // call the user's callback function for the operator
            out_data[stencil_idx] = func((row - kRowOutputThreshLow),
                                         (col_local - kColThreshLow),
                                         shifty_pixels_copy,
                                         stencil_args...);
          });

          // write the output data if it is in range
          if ((row >= kRowOutputThreshLow) &&
              (col >= kColThreshLow) && (col < col_thresh_high)) {
            OutPipe::write(out_data);
          }
        }
      }
    });
  });
}

#endif /* __LINE_KERNEL_HPP__ */