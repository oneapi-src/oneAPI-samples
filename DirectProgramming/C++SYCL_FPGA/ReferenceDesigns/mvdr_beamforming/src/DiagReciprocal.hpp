#ifndef __DIAG_RECIPROCAL__
#define __DIAG_RECIPROCAL__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "mvdr_complex.hpp"

// SubmitDiagReciprocalKernel
// Accept an upper-triangular square R Matrix from QR Decomposition from
// StreamingQRD. Compute the reciprocals of the square diagonal and write them
// to an output pipe. All values on the diagonal must be real-valued (imaginary
// part equal to 0)
template <typename DiagReciprocalKernelName,  // name to use for kernel
          size_t k_r_num_rows,     // Number of rows (and cols) in R Matrix
          typename RMatrixInPipe,  // The R Matrix input
          typename RDiagRecipVectorOutPipe  // 1 / values on diagonal of R
                                            // Matrix input
          >
event SubmitDiagReciprocalKernel(queue& q) {
  auto e = q.submit([&](handler& h) {
    h.single_task<DiagReciprocalKernelName>([=] {
      // calculate total number of elements passed in for one processing
      // iteration
      constexpr int kElements = k_r_num_rows * (k_r_num_rows + 1) / 2;

      while (1) {
        int row = 1;
        int col = 1;
        for (int i = 0; i < kElements; i++) {
          ComplexType in = RMatrixInPipe::read();

          if (row == col) {
            // Reciprocal square root is cheaper to calculate than reciprocal so
            // we square the values first
            RDiagRecipVectorOutPipe::write(sycl::rsqrt(in.real() * in.real()));
          }

          // calculate next element's row and col
          if (col == k_r_num_rows) {
            col = row + 1;
            row++;
          } else {
            col++;
          }
        }
      }
    });  // end of h.single_task
  });    // end of q.submit

  return e;
}

#endif /* __DIAG_RECIPROCAL__ */