#ifndef __ROW_STENCIL_HPP__
#define __ROW_STENCIL_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <limits>

#include "data_bundle.hpp"
#include "mp_math.hpp"
#include "shift_reg.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;
using namespace hldutils;

//
// helper function to pad the number of columns based on the filter size
//
template <typename IndexT, unsigned filter_size>
IndexT PadColumns(IndexT cols) {
  constexpr int kPaddingPixels = filter_size / 2;
  return cols + 2 * kPaddingPixels;
}

//
// Generic 1D row (i.e. horizontal) stencil.
//
// TEMPLATE PARAMETERS
// InType:            The input pixel type. This is read in by the row stencil
//                    through a SYCL pipe. The pipe should be hold
//                    'parallel_cols' elements of this type using the
//                    'DataBundle' type (DataBundle<InType, parallel_cols>).
// OutType:           The output pixel type. The same logic as the InType above.
//                    The data written to the output type is
//                    DataBundle<OutType, parallel_cols>
// IndexT:            The datatype used for indexing. This type should have
//                    enough bits to count up to the number or rows and columns.
// InPipe:            The input pipe to stream in 'parallel_cols' 'InT' values.
// OutPipe:           The output pipe to stream out 'parallel_cols' 'OutT'
//                    values.
// filter_size:       The filter size (i.e., the number of pixels to convolve).
// parallel_cols:     The number of columns to compute in parallel.
// StencilFunction:   The stencil callback functor, provided by the user, which
//                    is called for every pixel to perform the actual
//                    convolution. The function definition should be as follows:
//
//    OutT MyStencilFunction(int, int, ShiftReg<InT, filter_size>,
//                           FunctionArgTypes...)
//
//                    The user can provide extra arguments to the callback by
//                    using the FunctionArgTypes parameter pack.
// FunctionArgTypes:  The user-provided type parameter pack of the arguments to
//                    pass to the callback function.
//
//
// FUNCTION ARGUMENTS
// rows:            The number of rows in the image.
// cols:            The number of columns in the image.
//                  computed by the IP is rows*cols.
// zero_val:        The 'zero' value for the stencil. This is used to pad
//                  the columns of the image.
// func:            The user-defined functor. This is a callback that is called
//                  to perform the 1D convolution.
// stencil_args...: The parameter pack of arguments to be passed to the
//                  user-defined callback functor.
//
template <typename InType, typename OutType, typename IndexT, typename InPipe,
          typename OutPipe, unsigned filter_size, unsigned parallel_cols,
          typename StencilFunction, typename... FunctionArgTypes>
void RowStencil(IndexT rows, IndexT cols, const InType zero_val,
                StencilFunction func, FunctionArgTypes... stencil_args) {
  // types coming into and out of the kernel from pipes, respectively
  using InPipeT = DataBundle<InType, parallel_cols>;
  using OutPipeT = DataBundle<OutType, parallel_cols>;

  // number of pixels to pad to the columns with
  constexpr int kPaddingPixels = filter_size / 2;

  // the size of the shift register to hold the window
  constexpr int kShiftRegSize = filter_size + parallel_cols - 1;
  constexpr IndexT kColThreshLow = kPaddingPixels;

  // static asserts to validate template arguments
  static_assert(filter_size > 1);
  static_assert(parallel_cols > 0);
  static_assert(IsPow2(parallel_cols));
  static_assert(std::is_integral_v<IndexT>);
  static_assert(std::is_invocable_r_v<OutType, StencilFunction, int, int,
                                      ShiftReg<InType, filter_size>,
                                      FunctionArgTypes...>);

  // constants
  const IndexT col_thresh_high = cols + kPaddingPixels;
  const IndexT padded_cols = PadColumns<IndexT, filter_size>(cols);
  const IndexT col_loop_bound = padded_cols / parallel_cols;

  // the shift register
  [[intel::fpga_register]] ShiftReg<InType, kShiftRegSize> shifty_pixels;

  // initialize the contents of the shift register
  #pragma unroll
  for (int i = 0; i < kShiftRegSize; i++) {
    shifty_pixels[i] = zero_val;
  }

  // the main processing loop for the image
  [[intel::initiation_interval(1)]]
  for (IndexT row = 0; row < rows; row++) {
    [[intel::initiation_interval(1)]]
    for (IndexT col_loop = 0; col_loop < col_loop_bound; col_loop++) {
      IndexT col = col_loop * parallel_cols;

      // read from the input pipe if there are still pixels to read
      InPipeT new_pixels(zero_val);
      if (col < cols) {
        new_pixels = InPipe::read();
      }

      // shift in the input pixels
      shifty_pixels.ShiftMultiVals(new_pixels);

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
        out_data[stencil_idx] = func(row, (col_local - kColThreshLow),
                                      shifty_pixels_copy, stencil_args...);
      });

      // write the output data if it is in range (i.e., it is a real pixel
      // and not part of the padding)
      if ((col >= kColThreshLow) && (col < col_thresh_high)) {
        OutPipe::write(out_data);
      }
    }
  }
}

#endif /* __ROW_STENCIL_HPP__ */