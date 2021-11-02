// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

// The values for FIXED_ITERATIONS, ROWS_COMPONENT and COLS_COMPONENT will be
// supplied by the build system 


// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#pragma once

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <chrono>
#include <cstring>
#include <vector>
#include <type_traits>

#include "Tuple.hpp"
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
                                                                    &AMatrix, 
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                    &QMatrix,
    std::vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                    &RMatrix,
                                sycl::queue &q, size_t matrices, size_t reps) {

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

      // Copy a new input matrix from the host memory to the FPGA DDR 
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

    } // end for it=0:matrices-1 
  } // end for r=0:reps-1 

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    delete ABuffer[b];
    delete QBuffer[b];
    delete RBuffer[b];
  }
}


/*
  Computes Q and R matrices such that A=QR where:
  - A is the input matrix
  - Q is a unitary/orthogonal matrix
  - R is an upper triangular matrix

  This function implements a OneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan 
  Pasca.

  Each matrix (input and output) are represented using vectors in a column
  fashion (transposed).

  Function arguments:
  - AMatrix:    The input matrix. Interpreted as a transposed matrix.
  - QMatrix:    The Q matrix. The function will overwrite this matrix.
  - RMatrix     The R matrix. The function will overwrite this matrix.
                The vector will only contain the upper triangular elements
                of the matrix, in a row by row fashion.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read sequentially from the AMatrix 
                vector.
  - reps:       The number of repetitions of the computation to execute.
                (for performance evaluation)

  This function requires the following template parameters:
  - columns:    The number of columns in the matrix
  - rows:       The number of rows in the matrix     
  - rawLatency: The latency between the RAW dependency in the triangular
                loop that prevents the compiler to achieve an II of 1.
                This helps create a loop structure that can reach an II
                of 1 following the triangular loop optimization tutorial
                method.
*/

// Complex single precision floating-point QR Decomposition
template<unsigned columns, unsigned rows, unsigned rawLatency, typename T>
void QRDecomposition( std::vector<ac_complex<T>> &AMatrix, 
                      std::vector<ac_complex<T>> &QMatrix,
                      std::vector<ac_complex<T>> &RMatrix,
                      sycl::queue &q, 
                      size_t matrices, 
                      size_t reps) {

  constexpr bool isComplex = true;
  QRDecomposition_impl<columns, rows, rawLatency, isComplex, T>
                                (AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}

// Real single precision floating-point QR Decomposition
template<unsigned columns, unsigned rows, unsigned rawLatency, typename T>
void QRDecomposition( std::vector<T> &AMatrix, 
                      std::vector<T> &QMatrix,
                      std::vector<T> &RMatrix,
                      sycl::queue &q, 
                      size_t matrices, 
                      size_t reps) {

  constexpr bool isComplex = false;
  QRDecomposition_impl<columns, rows, rawLatency, isComplex, T>
                                (AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}
