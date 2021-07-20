#ifndef __HORIZONTAL_KERNEL_HPP__
#define __HORIZONTAL_KERNEL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "data_bundle.hpp"
#include "shift_reg.hpp"

using namespace sycl;
using namespace hldutils;

//
// Kernel to perform the horizontal filtering
//
template<typename KernelId, typename InType, typename OutType,
         typename IndexT, typename InPipe, typename OutPipe,
         int filter_size, int parallel_cols,
         typename LineFunction,
         typename... FunctionArgs_T>
event SubmitHorizontalKernel(queue& q, IndexT rows, IndexT cols, IndexT frames,
                             const InType zero_val, LineFunction func,
                             FunctionArgs_T... stencil_args) {
  // types coming into and out of the kernel from pipes, respectively
  using InPipeT = DataBundle<InType, parallel_cols>;
  using OutPipeT = DataBundle<OutType, parallel_cols>;

  constexpr int kShiftRegSize = filter_size + parallel_cols - 1;
  constexpr int kPaddingPixels = filter_size / 2;
  constexpr IndexT kColThreshLow = kPaddingPixels;

  // TODO: static asserts

  const IndexT col_thresh_high = cols + kPaddingPixels;
  const IndexT padded_cols = cols + 2 * kPaddingPixels;

  // validate number of columns
  if ((padded_cols % parallel_cols) != 0) {
    std::cerr << "ERROR: the number of parallel columns (" << parallel_cols
              << ") must be a multiple of the number of padded columns ("
              << padded_cols << ")\n";
    std::terminate();
  }

  const IndexT col_loop_bound = padded_cols / parallel_cols;

  return q.submit([&](handler &h) {
    h.single_task<KernelId>([=] {
      // the shift register
      ShiftReg<InType, kShiftRegSize> shifty_pixels;

      // initialize the contents of the shift register
      #pragma unroll
      for (int i = 0; i < kShiftRegSize; i++) {
        shifty_pixels[i] = zero_val;
      }

      [[intel::loop_coalesce(3), intel::initiation_interval(1)]]
      for (IndexT frame = 0; frame < frames; frame++) {
        for (IndexT row = 0; row < rows; row++) {
          for (IndexT col_loop = 0; col_loop < col_loop_bound; col_loop++) {
            IndexT col = col_loop * parallel_cols;
            InPipeT new_pixels(zero_val);

            // read from the input pipe if there are still pixels to read
            if (col < cols) {
              new_pixels = InPipe::read();
            }

            // shift in the input pixels
            shifty_pixels.shiftMultiVals(new_pixels);

            // Perform the convolution on the 1D window
            OutPipeT out_data(OutType(0));
            UnrolledLoop<0, parallel_cols>([&](auto stencil_idx) {
              const int col_local = col + stencil_idx;
              ShiftReg<InType, filter_size> shifty_pixels_copy;

              // first, make an offsetted copy of the shift register
              UnrolledLoop<0, filter_size>([&](auto x) {
                shifty_pixels_copy[x] = shifty_pixels[x + stencil_idx];
              });

              // call the user's callback function for the operator
              out_data[stencil_idx] = func(row,
                                           (col_local - kColThreshLow),
                                           shifty_pixels_copy,
                                           stencil_args...);
            });

            // write the output data if it is in range (i.e., it is a real pixel
            // and not part of the padding)
            if ((col >= kColThreshLow) && (col < col_thresh_high)) {
              OutPipe::write(out_data);
            }
          }
        }
      }
    });
  });
}

#endif /* __LINE_KERNEL_HPP__ */