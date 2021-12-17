#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include <chrono>
#include <cstring>
#include <type_traits>
#include <vector>

#include "MemoryTransfers.hpp"
#include "StreamingQRD.hpp"
#include "StreamingQRI.hpp"
#include "Utils.hpp"

template <unsigned columns,        // Number of columns in the input matrix
          unsigned rows,           // Number of rows in the input matrix
          unsigned RAWLatencyQRD,  // RAW latency for triangular loop
                                   // optimization in the QRD kernel
          unsigned RAWLatencyQRI,  // RAW latency for triangular loop
                                   // optimization in the QRI kernel
          bool isComplex,          // Selects between ac_complex<T> and T
                                   // datatype
          typename T>              // The datatype for the computation
void QRI_impl(
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type>
        &AMatrix,  // Input matrix to inverse
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type>
        &inverseMatrix,  // Output inverse matrix
    sycl::queue &q,      // Device queue
    size_t matrices,     // Number of matrices to process
    size_t reps          // Number of repetitions (for performance
                         // evaluation)
) {
  // TT will be ac_complex<T> or T depending on isComplex
  typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

  // Functional limitations
  static_assert(std::is_same<T, float>::value,
                "only float datatype is supported");
  static_assert(
      rows >= columns,
      "only rectangular matrices with rows>=columns are matrices supported");
  static_assert((columns <= 512) && (columns >= 4),
                "only matrices of size 4x4 to 512x512 are supported");

  constexpr int kAMatrixSize = rows * columns;
  constexpr int kInverseMatrixSize = rows * columns;
  constexpr int kNumElementsPerDDRBurst = isComplex ? 4 : 8;

  // Number of buffers to allocate to be able to read/compute/store without
  // overlap.
  constexpr short kNumBuffers = 3;

  using PipeType = pipeTable<kNumElementsPerDDRBurst, TT>;

  using AMatrixPipe = sycl::ext::intel::pipe<class APipe, PipeType, 3>;
  using QMatrixPipe = sycl::ext::intel::pipe<class QPipe, PipeType, 3>;
  using RMatrixPipe = sycl::ext::intel::pipe<class RPipe, PipeType, 3>;
  using InverseMatrixPipe = sycl::ext::intel::pipe<class IPipe, PipeType, 3>;

  // We will process 'matricesPerIter' number of matrices in each run of the
  // kernel
#if defined(FPGA_EMULATOR)
  constexpr int matricesPerIter = 1;
  assert(matrices % matricesPerIter == 0);
#else
  constexpr int matricesPerIter = 2048;
  assert(matrices % matricesPerIter == 0);
#endif

  // Create buffers and allocate space for them.
  TT *ADevice[kNumBuffers];
  TT *IDevice[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    ADevice[i] = sycl::malloc_device<TT>(kAMatrixSize * matricesPerIter, q);
    IDevice[i] =
        sycl::malloc_device<TT>(kInverseMatrixSize * matricesPerIter, q);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {
    // Go over all the matrices, rotating buffers every time
    for (size_t bufferIdx = 0, it = 0; it < matrices;
         it += matricesPerIter, bufferIdx = (bufferIdx + 1) % kNumBuffers) {
      // Pointer to current input/output matrices in host memory
      const TT *kPtrA = AMatrix.data() + kAMatrixSize * it;
      TT *kPtrInverse = inverseMatrix.data() + kInverseMatrixSize * it;

      auto copyAEvent = q.memcpy(ADevice[bufferIdx], kPtrA,
                                 kAMatrixSize * matricesPerIter * sizeof(TT));

      TT *currentABuffer = ADevice[bufferIdx];
      TT *currentIBuffer = IDevice[bufferIdx];

      auto readEvent = q.submit([&](sycl::handler &h) {
        h.depends_on(copyAEvent);
        h.single_task<class QRI_DDR_to_local_mem>([=
        ]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                                  matricesPerIter, AMatrixPipe>(currentABuffer);
        });
      });

      // Read the A matrix from the AMatrixPipe pipe and compute the QR
      // decomposition. Write the Q and R output matrices to the QMatrixPipe
      // and RMatrixPipe pipes.
      q.single_task<class QRD>(
          StreamingQRD<T, isComplex, rows, columns, RAWLatencyQRD,
                       matricesPerIter, kNumElementsPerDDRBurst, AMatrixPipe,
                       QMatrixPipe, RMatrixPipe>());

      q.single_task<class QRI>(
          // Read the Q and R matrices from pipes and compute the inverse of A.
          // Write the result to the InverseMatrixPipe pipe.
          StreamingQRI<T, isComplex, rows, columns, RAWLatencyQRI,
                       matricesPerIter, kNumElementsPerDDRBurst, QMatrixPipe,
                       RMatrixPipe, InverseMatrixPipe>());

      auto IEvent = q.submit([&](sycl::handler &h) {
        h.single_task<class QRI_local_mem_to_DDR_Q>([=
        ]() [[intel::kernel_args_restrict]] {
          // Read the Q matrix from the QMatrixPipe pipe and copy it to the
          // FPGA DDR
          MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
                              matricesPerIter, InverseMatrixPipe>(
              currentIBuffer);
        });
      });

      IEvent.wait();

      // Copy the Q and R matrices result from the FPGA DDR to the host memory
      auto copyIEvent =
          q.memcpy(kPtrInverse, IDevice[bufferIdx],
                   kInverseMatrixSize * matricesPerIter * sizeof(TT));

      copyIEvent.wait();
      readEvent.wait();
    }  // end of it
  }    // end of r

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    free(ADevice[b], q);
    free(IDevice[b], q);
  }
}