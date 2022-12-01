#ifndef __STREAMING_QRD_WRAPPER_HPP__
#define __STREAMING_QRD_WRAPPER_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "mvdr_complex.hpp"

// utility classes found in DirectProgramming/C++SYCL_FPGA/include
#include "streaming_qrd.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;

// SubmitStreamingQRDKernel
// Accept an input matrix one column at a time from an array of pipes.  Perform
// Q R Decomposition on the array and send the result out through pipes.
template <typename StreamingQRDKernelName,  // Name to use for the Kernel

          size_t k_min_inner_loop_iterations,  // Minimum number of inner loop
                                               // iterations to achieve an outer
                                               // loop II of 1.  This value will
                                               // have to be tuned for optimal
                                               // performance.  Refer to the
                                               // Triangular Loop design pattern
                                               // tutorial.

          size_t k_a_num_rows,      // Number of rows in the incoming A matrix
          size_t k_a_num_cols,      // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          size_t k_pipe_width,      // number of elements read/written
                                    // (wrapped in NTuple) from/to pipes
          typename AMatrixInPipe,   // A matrix input, receive a full column
                                    // of complex numbers with each read,
                                    // wrapped in NTuple
          typename QMatrixOutPipe,  // Q output pipe, send a full column
                                    // of complex numbers with each write.
                                    // Column 0 is sent first, k_a_num_cols-1
                                    // is sent last
          typename RMatrixOutPipe   // R output pipe.  Send one complex number
                                    // per write.  Only upper-right elements
                                    // of R are sent.  Sent in row order,
                                    // starting with row 0.
          >
event SubmitStreamingQRDKernel(queue& q) {
  // Template parameter checking
  static_assert(std::numeric_limits<short>::max() > k_a_num_cols,
                "k_a_num_cols must fit in a short");
  static_assert(k_a_num_rows >= k_a_num_cols,
                "k_a_num_rows must be greater than or equal to k_a_num_cols");
  static_assert(std::numeric_limits<short>::max() > k_a_num_rows,
                "k_a_num_rows must fit in a short");
  static_assert(k_a_num_rows % k_pipe_width == 0,
                "k_a_num_rows must be evenly divisible by k_pipe_width");

  auto e = q.submit([&](sycl::handler& h) {
    h.single_task<StreamingQRDKernelName>(
        fpga_linalg::StreamingQRD<float, true, k_a_num_rows, k_a_num_cols,
                                  k_min_inner_loop_iterations, k_pipe_width,
                                  AMatrixInPipe, QMatrixOutPipe, RMatrixOutPipe,
                                  false>());
  });

  return e;
}

#endif  // ifndef __STREAMING_QRD_HPP_MVDR__
