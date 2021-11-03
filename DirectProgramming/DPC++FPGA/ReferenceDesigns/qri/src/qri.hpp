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
#include "StreamingQRD.hpp"
#include "StreamingQRI.hpp"
#include "MemoryTransfers.hpp"

template< unsigned columns,       // Number of columns in the input matrix
          unsigned rows,          // Number of rows in the input matrix
          unsigned RAWLatencyQRD, // RAW latency for triangular loop 
                                  // optimization in the QRD kernel
          unsigned RAWLatencyQRI, // RAW latency for triangular loop 
                                  // optimization in the QRI kernel
          bool isComplex,         // Selects between ac_complex<T> and T 
                                  // datatype
          typename T>             // The datatype for the computation
void QRI_impl(  
      std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
              &AMatrix,         // Input matrix to inverse
      std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
              &inverseMatrix,   // Output inverse matrix 
              sycl::queue &q,   // Device queue
              size_t matrices,  // Number of matrices to process
              size_t reps       // Number of repetitions (for performance 
                                //evaluation)
              ) {

  // TT will be ac_complex<T> or T depending on isComplex
  typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

  // Functional limitations
  static_assert(std::is_same<T, float>::value, 
                                          "only float datatype is supported");
  static_assert(rows>=columns, 
       "only rectangular matrices with rows>=columns are matrices supported");
  static_assert((columns <= 512) && (columns >= 4), 
                        "only matrices of size 4x4 to 512x512 are supported");
  
  constexpr int kAMatrixSize = rows * columns;
  constexpr int kInverseMatrixSize = rows * columns;
  constexpr int kNumElementsPerDDRBurst = isComplex ? 4 : 8;

  // Number of buffers to allocate to be able to read/compute/store without 
  // overlap.
  constexpr short kNumBuffers = 3;
  
  using PipeType = pipeTable<rows, TT>;

  using AMatrixPipe = sycl::ext::intel::pipe<class APipe, PipeType, 4>;
  using QMatrixPipe = sycl::ext::intel::pipe<class QPipe, PipeType, 4>;
  using RMatrixPipe = sycl::ext::intel::pipe<class RPipe, TT, 
                                                    kNumElementsPerDDRBurst*3>;
  using InverseMatrixPipe = sycl::ext::intel::pipe<class IPipe, PipeType, 4>;

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
  sycl::buffer<TT, 1> *ABuffer[kNumBuffers];
  sycl::buffer<TT, 1> *inverseBuffer[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    ABuffer[i] = new sycl::buffer<TT, 1>(kAMatrixSize * matricesPerIter);
    inverseBuffer[i] = new sycl::buffer<TT, 1>(kInverseMatrixSize * 
                                                              matricesPerIter);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {

    // Go over all the matrices, rotating buffers every time
    for (size_t bufferIdx = 0, it = 0; it < matrices; 
            it += matricesPerIter, bufferIdx = (bufferIdx + 1) % kNumBuffers) {

      // Pointer to current input/output matrices in host memory 
      const TT *kPtrA = AMatrix.data() + kAMatrixSize * it;
      TT *kPtrInverse = inverseMatrix.data() + kInverseMatrixSize * it;

      // Copy a new set of input matrices from the host memory into the FPGA DDR
      q.submit([&](sycl::handler &h) {
        auto AMatrixDevice = ABuffer[bufferIdx]->
                      template get_access<sycl::access::mode::discard_write>(h);
        h.copy(kPtrA, AMatrixDevice);
      });

      // Read an input matrix A from the host memory to the FPGA DDR
      // Stream the A matrix to the AMatrixPipe pipe
      MatrixReadFromDDRToPipeByColumns< class QRI_DDR_to_local_mem, 
                                        TT,
                                        rows,
                                        columns,
                                        kNumElementsPerDDRBurst,
                                        matricesPerIter,
                                        AMatrixPipe>
                                        (q, ABuffer[bufferIdx]);

      // Read the A matrix from the AMatrixPipe pipe and compute the QR 
      // decomposition. Write the Q and R output matrices to the QMatrixPipe
      // and RMatrixPipe pipes.
      StreamingQRDKernel< class QRD_compute, 
                          T, 
                          isComplex,
                          rows, 
                          columns,
                          RAWLatencyQRD, 
                          matricesPerIter,
                          AMatrixPipe,
                          QMatrixPipe,
                          RMatrixPipe>(q);

      // Read the Q and R matrices from pipes and compute the inverse of A. 
      // Write the result to the InverseMatrixPipe pipe.
      StreamingQRIKernel< class QRI_compute, 
                          T, 
                          isComplex, 
                          rows, 
                          columns,
                          RAWLatencyQRI, 
                          matricesPerIter,
                          QMatrixPipe,
                          RMatrixPipe,
                          InverseMatrixPipe>(q);

      // Read the inverse matrix from the InverseMatrixPipe pipe and copy it to
      // the FPGA DDR
      MatrixReadPipeByColumnsToDDR< class QRI_local_mem_to_DDR, 
                                    TT,
                                    rows,
                                    columns,
                                    kNumElementsPerDDRBurst,
                                    matricesPerIter,
                                    InverseMatrixPipe>
                                    (q, inverseBuffer[bufferIdx]);

      // Copy the output result from the FPGA DDR to the host memory
      q.submit([&](sycl::handler &h) {
        sycl::accessor inverseMatrixAcc(*inverseBuffer[bufferIdx], h, 
                                                              sycl::read_only);
        h.copy(inverseMatrixAcc, kPtrInverse);
      });
    } // end of it 
  } // end of r 

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    delete ABuffer[b];
    delete inverseBuffer[b];
  }
}