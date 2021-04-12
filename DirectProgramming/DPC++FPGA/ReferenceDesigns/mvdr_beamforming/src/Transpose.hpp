#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Tuple.hpp"
#include "UnrolledLoop.hpp"

using namespace sycl;

// the generic transpose class. Defined below after SubmitTransposeKernel.
template <typename T, size_t k_num_cols_in, size_t k_pipe_width,
          typename MatrixInPipe, typename MatrixOutPipe>
struct Transposer;

// SubmitTransposeKernel
// Accept k_pipe_width elements of type T wrapped in NTuple
// on each read from DataInPipe.  Store elements locally until we can write
// to the output array of pipes with data in transposed order.
// For simplicity of terminology, incoming data is defined to enter in 'row'
// order and exit in 'column' order, although for the purposes of the transpose
// the terms 'row' and 'column' could be interchanged.
// Row order means all data from a given row is received before any data from
// the next row is received.
// This kernel also performs flow control and ensures that no partial matrices
// will be written into the output pipe.
template <typename TransposeKernelName,  // Name to use for the Kernel
          typename T,                    // type of element to transpose
          size_t k_num_cols_in,   // number of columns in the input matrix
          size_t k_pipe_width,    // number of elements read/written
                                  // (wrapped in NTuple) from/to pipes
          typename MatrixInPipe,  // Receive the input matrix in row order
                                  // Receive k_pipe_width elements of type
                                  // T wrapped in NTuple on each read
          typename MatrixOutPipe  // Send the output matrix in column order.
                                  // Send k_pipe_width elements of type T
                                  // wrapped in NTuple on each write.
          >
event SubmitTransposeKernel(queue& q) {
  // Template parameter checking
  static_assert(std::numeric_limits<short>::max() > k_num_cols_in,
                "k_num_cols_in must fit in a short");
  static_assert(k_num_cols_in % k_pipe_width == 0,
                "k_num_cols_in must be evenly divisible by k_pipe_width");

  return q.submit([&](handler& h) {
    h.single_task<TransposeKernelName>([=]() {
      // start the transposer
      Transposer<T, k_num_cols_in, k_pipe_width, MatrixInPipe, MatrixOutPipe>
          TheTransposer;
      TheTransposer();
    });
  });
}

// The generic transpose. We use classes here because we want to do partial
// template specialization on the 'k_pipe_width' to optimize the case where
// 'k_pipe_width' == 1 and this cannot be done with a function.
template <typename T, size_t k_num_cols_in, size_t k_pipe_width,
          typename MatrixInPipe, typename MatrixOutPipe>
struct Transposer {
  void operator()() const {
    using PipeType = NTuple<T, k_pipe_width>;

    while (1) {
      // This is a scratch pad memory that we will use to do the transpose.
      // We read the data in from a pipe (k_pipe_width elements at at time),
      // store it in this memory in row-major format and read it out in
      // column-major format (again, k_pipe_width elements at a time).
      T scratch[k_pipe_width][k_num_cols_in];

      // fill the matrix internally
      // NO-FORMAT comments are for clang-format
      [[intel::ii(1)]]                     // NO-FORMAT: Attribute
      [[intel::loop_coalesce(2)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (int y = 0; y < k_pipe_width; y++) {
        for (int x = 0; x < k_num_cols_in / k_pipe_width; x++) {
          PipeType in_data = MatrixInPipe::read();
          UnrolledLoop<k_pipe_width>([&](auto i) {
            scratch[y][x * k_pipe_width + i] = in_data.template get<i>();
          });
        }
      }

      // write output
      // NO-FORMAT comments are for clang-format
      [[intel::ii(1)]]                     // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (int x = 0; x < k_num_cols_in; x++) {
        PipeType out_data;
        UnrolledLoop<k_pipe_width>(
            [&](auto i) { out_data.template get<i>() = scratch[i][x]; });

        MatrixOutPipe::write(out_data);
      }
    }
  }
};

// Special case for a k_pipe_width=1
// In this case, the is just a pass through kernel since the matrix is
// 1xk_num_cols. Overriding this version allows us to save area.
template <typename T, size_t k_num_cols_in, typename MatrixInPipe,
          typename MatrixOutPipe>
struct Transposer<T, k_num_cols_in, 1, MatrixInPipe, MatrixOutPipe> {
  void operator()() const {
    while (1) {
      auto d = MatrixInPipe::read();
      MatrixOutPipe::write(d);
    }
  }
};

#endif  // ifndef __TRANSPOSE_HPP__
