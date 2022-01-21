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
#include "streaming_qri.hpp"
#include "utils.hpp"

// Forward declare the kernel names
// (This prevents unwanted name mangling in the optimization report.)
class QRIDDRToLocalMem;
class QRD;
class QRI;
class QRILocalMemToDDRQ;

template <unsigned columns,         // Number of columns in the input matrix
          unsigned rows,            // Number of rows in the input matrix
          unsigned raw_latency_qrd, // RAW latency for triangular loop
                                    // optimization in the QRD kernel
          unsigned raw_latency_qri, // RAW latency for triangular loop
                                    // optimization in the QRI kernel
          bool is_complex,          // Selects between ac_complex<T> and T
                                    // datatype
          typename T,               // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
                                    // TT will be ac_complex<T> or T depending
                                    // on is_complex
          >
void QRIImpl(
    std::vector<TT> &a_matrix,       // Input matrix to inverse
    std::vector<TT> &inverse_matrix, // Output inverse matrix
    sycl::queue &q,                  // Device queue
    size_t matrices,                 // Number of matrices to process
    size_t reps                      // Number of repetitions (for performance
                                     // evaluation)
) {

  // Functional limitations
  static_assert(std::is_same_v<T, float>, "only float datatype is supported");
  static_assert(
      rows >= columns,
      "only rectangular matrices with rows>=columns are matrices supported");
  static_assert((columns <= 512) && (columns >= 4),
                "only matrices of size 4x4 to 512x512 are supported");

  constexpr int kAMatrixSize = rows * columns;
  constexpr int kInverseMatrixSize = rows * columns;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  // Number of buffers to allocate to be able to read/compute/store without
  // overlap.
  constexpr short kNumBuffers = 3;

  using PipeType = PipeTable<kNumElementsPerDDRBurst, TT>;

  using a_matrix_pipe = sycl::ext::intel::pipe<class APipe, PipeType, 3>;
  using q_matrix_pipe = sycl::ext::intel::pipe<class QPipe, PipeType, 3>;
  using r_matrix_pipe = sycl::ext::intel::pipe<class RPipe, TT, 3>;
  using inverse_matrix_pipe = sycl::ext::intel::pipe<class IPipe, PipeType, 3>;

  // We will process 'kMatricesPerIter' number of matrices in each run of the
  // kernel
#if defined(FPGA_EMULATOR)
  constexpr int kMatricesPerIter = 1;
#else
  constexpr int kMatricesPerIter = 2048;
#endif
  assert(matrices % kMatricesPerIter == 0);

  // Create buffers and allocate space for them.
  TT *a_device[kNumBuffers];
  TT *i_device[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    a_device[i] = sycl::malloc_device<TT>(kAMatrixSize * kMatricesPerIter, q);
    i_device[i] =
        sycl::malloc_device<TT>(kInverseMatrixSize * kMatricesPerIter, q);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {
    // Go over all the matrices, rotating buffers every time
    for (size_t buffer_idx = 0, it = 0; it < matrices;
         it += kMatricesPerIter, buffer_idx = (buffer_idx + 1) % kNumBuffers) {
      // Pointer to current input/output matrices in host memory
      const TT *ptr_a = a_matrix.data() + kAMatrixSize * it;
      TT *ptr_inverse = inverse_matrix.data() + kInverseMatrixSize * it;

      auto copy_a_event = q.memcpy(a_device[buffer_idx], ptr_a,
                                 kAMatrixSize * kMatricesPerIter * sizeof(TT));

      TT *current_a_buffer = a_device[buffer_idx];
      TT *current_i_buffer = i_device[buffer_idx];

      auto read_event = q.submit([&](sycl::handler &h) {
        h.depends_on(copy_a_event);
        h.single_task<QRIDDRToLocalMem>([=
        ]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                            kMatricesPerIter, a_matrix_pipe>(current_a_buffer);
        });
      });

      // Read the A matrix from the a_matrix_pipe pipe and compute the QR
      // decomposition. Write the Q and R output matrices to the q_matrix_pipe
      // and r_matrix_pipe pipes.
      q.single_task<QRD>(
          StreamingQRD<T, is_complex, rows, columns, raw_latency_qrd,
                       kMatricesPerIter, kNumElementsPerDDRBurst, 
                       a_matrix_pipe, q_matrix_pipe, r_matrix_pipe>());

      q.single_task<QRI>(
          // Read the Q and R matrices from pipes and compute the inverse of A.
          // Write the result to the inverse_matrix_pipe pipe.
          StreamingQRI<T, is_complex, rows, columns, raw_latency_qri,
                       kMatricesPerIter, kNumElementsPerDDRBurst, 
                       q_matrix_pipe, r_matrix_pipe, inverse_matrix_pipe>());


      auto i_event = q.single_task<QRILocalMemToDDRQ>([=
                                          ]() [[intel::kernel_args_restrict]] {
          // Read the Q matrix from the q_matrix_pipe pipe and copy it to the
          // FPGA DDR
          MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
                              kMatricesPerIter, inverse_matrix_pipe>(
              current_i_buffer);
      });

      i_event.wait();

      // Copy the Q and R matrices result from the FPGA DDR to the host memory
      auto copy_i_event =
          q.memcpy(ptr_inverse, i_device[buffer_idx],
                   kInverseMatrixSize * kMatricesPerIter * sizeof(TT));

      copy_i_event.wait();
      read_event.wait();
    }  // end of it
  }    // end of r

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    free(a_device[b], q);
    free(i_device[b], q);
  }
}