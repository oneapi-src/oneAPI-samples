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
#include "MemoryTransfers.hpp"

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

  using PipeType = column<rows, TT>;

  // Number of buffers to allocate to be able to read/compute/store without 
  // overlap.
  constexpr short kNumBuffers = 3;
  
  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<class APipe, PipeType, 1*2>;
  using QMatrixPipe = sycl::ext::intel::pipe<class QPipe, PipeType, 1*2>;
  using RMatrixPipe = sycl::ext::intel::pipe<class RPipe, TT, 
                                                    kNumElementsPerDDRBurst*3>;

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
  sycl::buffer<TT, 1> *QBuffer[kNumBuffers];
  sycl::buffer<TT, 1> *RBuffer[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    ABuffer[i] = new sycl::buffer<TT, 1>(kAMatrixSize * matricesPerIter);
    QBuffer[i] = new sycl::buffer<TT, 1>(kQMatrixSize * matricesPerIter);
    RBuffer[i] = new sycl::buffer<TT, 1>(kRMatrixSize * matricesPerIter);
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

      // Copy a new set of input matrices from the host memory to the FPGA DDR 
      q.submit([&](sycl::handler &h) {
        auto AMatrixDevice = ABuffer[bufferIdx]->
                      template get_access<sycl::access::mode::discard_write>(h);
        h.copy(kPtrA, AMatrixDevice);
      });

      // Read an input matrix A from the host memory to the FPGA DDR
      // Stream the A matrix to the AMatrixPipe pipe
      MatrixReadFromDDRToPipeByColumns< class QRD_DDR_to_local_mem, 
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
                          rawLatency, 
                          matricesPerIter,
                          AMatrixPipe,
                          QMatrixPipe,
                          RMatrixPipe>(q);

      // Read the Q matrix from the QMatrixPipe pipe and copy it to the
      // FPGA DDR
      MatrixReadPipeByColumnsToDDR< class QRD_local_mem_to_DDR_Q, 
                                    TT,
                                    rows,
                                    columns,
                                    kNumElementsPerDDRBurst,
                                    matricesPerIter,
                                    QMatrixPipe>
                                    (q, QBuffer[bufferIdx]);

      // Read the R matrix from the RMatrixPipe pipe and copy it to the
      // FPGA DDR
      VectorReadPipeByElementsToDDR<  class QRD_local_mem_to_DDR_R, 
                                      TT,
                                      kRMatrixSize,
                                      kNumElementsPerDDRBurst,
                                      matricesPerIter,
                                      RMatrixPipe>
                                      (q, RBuffer[bufferIdx]);

      // Copy the Q matrix result from the FPGA DDR to the host memory
      q.submit([&](sycl::handler &h) {
        sycl::accessor QMatrix(*QBuffer[bufferIdx], h, sycl::read_only);
        h.copy(QMatrix, ptrQ);
      });

      // Copy the R matrix result from the FPGA DDR to the host memory
      q.submit([&](sycl::handler &h) {
        sycl::accessor RMatrix(*RBuffer[bufferIdx], h, sycl::read_only);
        h.copy(RMatrix, ptrR);
      });

    } // end of it 
  } // end of r 

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    delete ABuffer[b];
    delete QBuffer[b];
    delete RBuffer[b];
  }
}