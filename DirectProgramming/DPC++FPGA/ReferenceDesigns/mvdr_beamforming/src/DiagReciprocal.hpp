#ifndef __DIAG_RECIPROCAL__
#define __DIAG_RECIPROCAL__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "mvdr_complex.hpp"

// SubmitDiagReciprocalKernel
// Accept an upper-triangular R Matrix from QR Decomposition from StreamingQRD.
// Compute the reciprocals of the diagonal and write them to an output pipe.

template <typename DiagReciprocalKernelName,  // name to use for kernel
          size_t k_r_num_rows,                // Number of rows in R Matrix
          typename RMatrixInPipe,             // The R Matrix input
          typename RDiagRecipVectorOutPipe    // 1 / values on diagonal of R
                                              // Matrix input
          >
event SubmitDiagReciprocalKernel(queue& q) {
  auto e = q.submit([&](handler& h) {
    h.single_task<DiagReciprocalKernelName>([=] {
      // calculate total number of elements passed in for one processing
      // iteration
      constexpr int elements = k_r_num_rows * (k_r_num_rows + 1) / 2;

      while (1) {
        int row = 1;
        int col = 1;
        for (int i = 0; i < elements; i++) {
          ComplexType in = RMatrixInPipe::read();

          if (row == col) {
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