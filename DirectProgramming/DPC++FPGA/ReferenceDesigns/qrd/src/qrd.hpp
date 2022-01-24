#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include <chrono>
#include <cstring>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_qrd.hpp"
#include "utils.hpp"

// Forward declare the kernel names
// (This prevents unwanted name mangling in the optimization report.)
class QRDDDRToLocalMem;
class QRD;
class QRDLocalMemToDDRQ;
class QRDLocalMemToDDRR;

/*
  Implementation of the QR decomposition using multiple streaming kernels
  Can be configured by datatype, matrix size (4x4 to
  512x512) and works with square or rectangular matrices, real and complex.
*/
template <unsigned columns,     // Number of columns in the input matrix
          unsigned rows,        // Number of rows in the input matrix
          unsigned raw_latency,  // RAW latency for triangular loop optimization
          bool is_complex,       // Selects between ac_complex<T> and T datatype
          typename T,           // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
                        // TT will be ac_complex<T> or T depending on is_complex
         >
void QRDecompositionImpl(
  std::vector<TT> &a_matrix, // Input matrix to decompose
  std::vector<TT> &q_matrix, // Output matrix Q
  std::vector<TT> &r_matrix, // Output matrix R
  sycl::queue &q,           // Device queue
  size_t matrices,          // Number of matrices to process
  size_t reps               // Number of repetitions, for performance evaluation
) {

  constexpr int kAMatrixSize = columns * rows;
  constexpr int kQMatrixSize = columns * rows;
  constexpr int kRMatrixSize = columns * (columns + 1) / 2;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  using pipe_type = PipeTable<kNumElementsPerDDRBurst, TT>;

  // Number of buffers to allocate to be able to read/compute/store without
  // overlap.
  constexpr short kNumBuffers = 3;

  // Pipes to communicate the A, Q and R matrices between kernels
  using a_matrix_pipe = sycl::ext::intel::pipe<class APipe, pipe_type, 3>;
  using q_matrix_pipe = sycl::ext::intel::pipe<class QPipe, pipe_type, 3>;
  using r_matrix_pipe = sycl::ext::intel::pipe<class RPipe, pipe_type, 3>;

  // We will process 'matrices_per_iter' number of matrices in each run of the
  // kernel
#if defined(FPGA_EMULATOR)
  constexpr int matrices_per_iter = 1;
  assert(matrices % matrices_per_iter == 0);
#else
  constexpr int matrices_per_iter = 2048;
  assert(matrices % matrices_per_iter == 0);
#endif

  // Create buffers and allocate space for them.
  TT *a_device[kNumBuffers];
  TT *q_device[kNumBuffers];
  TT *r_device[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    a_device[i] = sycl::malloc_device<TT>(kAMatrixSize * matrices_per_iter, q);
    q_device[i] = sycl::malloc_device<TT>(kAMatrixSize * matrices_per_iter, q);
    r_device[i] = sycl::malloc_device<TT>(kRMatrixSize * matrices_per_iter, q);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {
    // Go over all the matrices, rotating buffers every time
    for (size_t buffer_idx = 0, it = 0; it < matrices;
         it += matrices_per_iter, buffer_idx = (buffer_idx + 1) % kNumBuffers) {
      // Pointers to current input/output matrices in host memory
      const TT *kPtrA = a_matrix.data() + kAMatrixSize * it;
      TT *ptr_q = q_matrix.data() + kQMatrixSize * it;
      TT *ptr_r = r_matrix.data() + kRMatrixSize * it;

      auto copy_a_event = q.memcpy(a_device[buffer_idx], kPtrA,
                                 kAMatrixSize * matrices_per_iter * sizeof(TT));

      TT *current_a_buffer = a_device[buffer_idx];
      TT *current_q_buffer = q_device[buffer_idx];
      TT *current_r_buffer = r_device[buffer_idx];

      auto read_event = q.submit([&](sycl::handler &h) {
        h.depends_on(copy_a_event);
        h.single_task<QRDDDRToLocalMem>([=
        ]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                            matrices_per_iter, a_matrix_pipe>(current_a_buffer);
        });
      });

      // Read the A matrix from the a_matrix_pipe pipe and compute the QR
      // decomposition. Write the Q and R output matrices to the q_matrix_pipe
      // and r_matrix_pipe pipes.
      q.single_task<QRD>(
          StreamingQRD<T, is_complex, rows, columns, raw_latency, 
                       matrices_per_iter, kNumElementsPerDDRBurst, 
                       a_matrix_pipe, q_matrix_pipe, r_matrix_pipe>());

      auto q_event = q.single_task<QRDLocalMemToDDRQ>([=
                                          ]() [[intel::kernel_args_restrict]] {
          // Read the Q matrix from the q_matrix_pipe pipe and copy it to the
          // FPGA DDR
          MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
                            matrices_per_iter, q_matrix_pipe>(current_q_buffer);
      });

      auto r_event = q.single_task<QRDLocalMemToDDRR>([=
                                          ]() [[intel::kernel_args_restrict]] {
          // Read the R matrix from the r_matrix_pipe pipe and copy it to the
          // FPGA DDR
          VectorReadPipeToDDR<TT, kRMatrixSize, kNumElementsPerDDRBurst,
                            matrices_per_iter, r_matrix_pipe>(current_r_buffer);
      });

      r_event.wait();
      q_event.wait();

      // Copy the Q and R matrices result from the FPGA DDR to the host memory
      auto copy_q_event = q.memcpy(ptr_q, q_device[buffer_idx],
                                 kAMatrixSize * matrices_per_iter * sizeof(TT));
      auto copy_r_event = q.memcpy(ptr_r, r_device[buffer_idx],
                                 kRMatrixSize * matrices_per_iter * sizeof(TT));

      copy_r_event.wait();
      copy_q_event.wait();
      read_event.wait();
    }  // end of it
  }    // end of r

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    free(a_device[b], q);
    free(q_device[b], q);
    free(r_device[b], q);
  }
}