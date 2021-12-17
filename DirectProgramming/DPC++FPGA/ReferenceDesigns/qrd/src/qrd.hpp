#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <chrono>
#include <cstring>
#include <vector>
#include <type_traits>

#include "UnrolledLoop.hpp"
#include "Utils.hpp"
#include "MemoryTransfers.hpp"
#include "StreamingQRD.hpp"

/*
  Implementation of the QR decomposition using multiple streaming kernels
  Can be configured by datatype, matrix size (4x4 to
  512x512) and works with square or rectangular matrices, real and complex.
*/
template< unsigned columns,    // Number of columns in the input matrix
          unsigned rows,       // Number of rows in the input matrix
          unsigned rawLatency, // RAW latency for triangular loop optimization
          bool isComplex,      // Selects between ac_complex<T> and T datatype
          typename T>          // The datatype for the computation
void QRDecomposition_impl(
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type>
                          &AMatrix,         // Input matrix to decompose
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type>
                          &QMatrix,         // Output matrix Q
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type>
                          &RMatrix,         // Output matrix R
                          sycl::queue &q,   // Device queue
                          size_t matrices,  // Number of matrices to process
                          size_t reps       // Number of repetitions (for
                                            // performance evaluation)
                          ) {

  // TT will be ac_complex<T> or T depending on isComplex
  typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

  constexpr int kAMatrixSize = columns * rows;
  constexpr int kQMatrixSize = columns * rows;
  constexpr int kRMatrixSize = columns * (columns + 1) / 2;
  constexpr int kNumElementsPerDDRBurst = isComplex ? 4 : 8;

  using PipeType = pipeTable<kNumElementsPerDDRBurst, TT>;

  // Number of buffers to allocate to be able to read/compute/store without
  // overlap.
  constexpr short kNumBuffers = 3;

  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<class APipe, PipeType, 3>;
  using QMatrixPipe = sycl::ext::intel::pipe<class QPipe, PipeType, 3>;
  using RMatrixPipe = sycl::ext::intel::pipe<class RPipe, PipeType, 3>;

  // We will process 'matricesPerIter' number of matrices in each run of the
  // kernel
#if defined(FPGA_EMULATOR)
  constexpr int matricesPerIter = 1;
  assert(matrices%matricesPerIter == 0);
#else
  constexpr int matricesPerIter = 2048;
  assert(matrices%matricesPerIter == 0);
#endif

  // Create buffers and allocate space for them.
  TT * ADevice[kNumBuffers];
  TT * QDevice[kNumBuffers];
  TT * RDevice[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    ADevice[i] = sycl::malloc_device<TT>(kAMatrixSize * matricesPerIter, q);
    QDevice[i] = sycl::malloc_device<TT>(kAMatrixSize * matricesPerIter, q);
    RDevice[i] = sycl::malloc_device<TT>(kRMatrixSize * matricesPerIter, q);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {
    // Go over all the matrices, rotating buffers every time
    for (size_t bufferIdx = 0, it = 0; it < matrices;
            it += matricesPerIter, bufferIdx = (bufferIdx + 1) % kNumBuffers) {
      // Pointers to current input/output matrices in host memory
      const TT *kPtrA = AMatrix.data() + kAMatrixSize * it;
      TT *ptrQ = QMatrix.data() + kQMatrixSize * it;
      TT *ptrR = RMatrix.data() + kRMatrixSize * it;

      auto copyAEvent = q.memcpy(ADevice[bufferIdx], kPtrA,
                                    kAMatrixSize * matricesPerIter*sizeof(TT));

      TT * currentABuffer = ADevice[bufferIdx];
      TT * currentQBuffer = QDevice[bufferIdx];
      TT * currentRBuffer = RDevice[bufferIdx];

      auto readEvent = q.submit([&](sycl::handler &h) {
        h.depends_on(copyAEvent);
        h.single_task<class QRD_DDR_to_local_mem>
                                        ([=]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRToPipe<TT,
                                  rows,
                                  columns,
                                  kNumElementsPerDDRBurst,
                                  matricesPerIter,
                                  AMatrixPipe>
                                  (currentABuffer);
        });
      });

      // Read the A matrix from the AMatrixPipe pipe and compute the QR
      // decomposition. Write the Q and R output matrices to the QMatrixPipe
      // and RMatrixPipe pipes.
      q.single_task<class QRD>(
          StreamingQRD< T,
                        isComplex,
                        rows,
                        columns,
                        rawLatency,
                        matricesPerIter,
                        kNumElementsPerDDRBurst,
                        AMatrixPipe,
                        QMatrixPipe,
                        RMatrixPipe>()
      );

      auto QEvent = q.submit([&](sycl::handler &h) {
        h.single_task<class QRD_local_mem_to_DDR_Q>
                                        ([=]() [[intel::kernel_args_restrict]] {
          // Read the Q matrix from the QMatrixPipe pipe and copy it to the
          // FPGA DDR
          MatrixReadPipeToDDR<TT,
                              rows,
                              columns,
                              kNumElementsPerDDRBurst,
                              matricesPerIter,
                              QMatrixPipe>
                              (currentQBuffer);
        });
      });

      auto REvent = q.submit([&](sycl::handler &h) {
        h.single_task<class QRD_local_mem_to_DDR_R>
                                        ([=]() [[intel::kernel_args_restrict]] {
          // Read the R matrix from the RMatrixPipe pipe and copy it to the
          // FPGA DDR
          VectorReadPipeToDDR<TT,
                              kRMatrixSize,
                              kNumElementsPerDDRBurst,
                              matricesPerIter,
                              RMatrixPipe>
                              (currentRBuffer);
        });
      });

      REvent.wait();
      QEvent.wait();

      // Copy the Q and R matrices result from the FPGA DDR to the host memory
      auto copyQEvent = q.memcpy(ptrQ, QDevice[bufferIdx],
                                    kAMatrixSize * matricesPerIter*sizeof(TT));
      auto copyREvent = q.memcpy(ptrR, RDevice[bufferIdx],
                                    kRMatrixSize * matricesPerIter*sizeof(TT));

      copyREvent.wait();
      copyQEvent.wait();
      readEvent.wait();
    } // end of it
  } // end of r


  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    free(ADevice[b], q);
    free(QDevice[b], q);
    free(RDevice[b], q);
  }
}