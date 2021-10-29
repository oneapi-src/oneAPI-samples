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

using std::vector;
using namespace sycl;

namespace QRDInternal{

  // Forward declare the kernel name
  // (This prevents unwanted name mangling in the optimization report.)
  class QRD;
       
  template< unsigned columns,    // Number of columns in the input matrix
            unsigned rows,       // Number of rows in the input matrix
            unsigned rawLatency, // RAW latency for triangular loop optimization
            bool isComplex,      // Selects between ac_complex<T> and T datatype
            typename T>          // The datatype for the computation
  void QRDecomposition_impl(  
            vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                      &AMatrix, 
            vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                      &QMatrix,
            vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                      &RMatrix,
                                       queue &q, size_t matrices, size_t reps) {

    // TT will be ac_complex<T> or T depending on isComplex
    typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

    // Functional limitations
    static_assert(std::is_same<T, float>::value, 
                                            "only float datatype is supported");
    static_assert(rows>=columns, 
         "only rectangular matrices with rows>=columns are matrices supported");
    static_assert((columns <= 512) && (columns >= 4), 
                          "only matrices of size 4x4 to 512x512 are supported");
    
    constexpr int kAMatrixSize = columns * rows;
    constexpr int kQMatrixSize = columns * rows;
    constexpr int kRMatrixSize = columns * (columns + 1) / 2;
    constexpr int kNumElementsPerBank = isComplex ? 4 : 8;

    using PipeCol = column<rows, TT>;

    // Number of buffers to allocate to be able to read/compute/store 
    // without overlap.
    constexpr short kNumBuffers = 3;
    
    using AMatrixPipe = sycl::ext::intel::pipe<class APipe, PipeCol, 1>;
    using QMatrixPipe = sycl::ext::intel::pipe<class QPipe, PipeCol, 1>;
    using RMatrixPipe = sycl::ext::intel::pipe<class RPipe, TT, 1>;

    // Create buffers and allocate space for them.
    buffer<TT, 1> *ABuffer[kNumBuffers];
    buffer<TT, 1> *QBuffer[kNumBuffers];
    buffer<TT, 1> *RBuffer[kNumBuffers];
    for (short i = 0; i < kNumBuffers; i++) {
      ABuffer[i] = new buffer<TT, 1>(kAMatrixSize);
      QBuffer[i] = new buffer<TT, 1>(kQMatrixSize);
      RBuffer[i] = new buffer<TT, 1>(kRMatrixSize);
    }

    // Repeat the computation multiple times (for performance analysis)
    for (size_t r = 0; r < reps; r++) {

      // Go over all the matrices, rotating buffers every time
      for (size_t bufferIdx = 0, it = 0; it < matrices; 
                      it += 1, bufferIdx = (bufferIdx + 1) % kNumBuffers) {

        // Pointer to current input/output matrices in host memory 
        const TT *kPtrA = AMatrix.data() + kAMatrixSize * it;
        TT *kPtrQ = QMatrix.data() + kQMatrixSize * it;
        TT *kPtrR = RMatrix.data() + kRMatrixSize * it;

        // Copy a new input matrix from the host memory into the FPGA DDR 
        q.submit([&](handler &h) {
          auto AMatrixDevice =
       ABuffer[bufferIdx]->template get_access<access::mode::discard_write>(h);
          h.copy(kPtrA, AMatrixDevice);
        });

        MatrixReadFromDDRToPipeByColumns< class QRD_DDR_to_local_mem, 
                                        TT,
                                        rows,
                                        columns,
                                        kNumElementsPerBank,
                                        AMatrixPipe>
                                        (q, ABuffer[bufferIdx]);

        StreamingQRDKernel< class QRD_compute, 
                            T, 
                            isComplex,
                            rows, 
                            columns,
                            rawLatency, 
                            AMatrixPipe,
                            QMatrixPipe,
                            RMatrixPipe>(q);

        MatrixReadPipeByColumnsToDDR< class QRD_local_mem_to_DDR_Q, 
                                      TT,
                                      rows,
                                      columns,
                                      kNumElementsPerBank,
                                      QMatrixPipe>
                                      (q, QBuffer[bufferIdx]);

        VectorReadPipeByElementsToDDR<  class QRD_local_mem_to_DDR_R, 
                                        TT,
                                        kRMatrixSize,
                                        kNumElementsPerBank,
                                        RMatrixPipe>
                                        (q, RBuffer[bufferIdx]);

        // Copy the output result from the FPGA DDR to the host memory
        q.submit([&](handler &h) {
          accessor QMatrix(*QBuffer[bufferIdx], h, read_only);
          h.copy(QMatrix, kPtrQ);
        });

        // Copy the output result from the FPGA DDR to the host memory
        q.submit([&](handler &h) {
          accessor RMatrix(*RBuffer[bufferIdx], h, read_only);
          h.copy(RMatrix, kPtrR);
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

} // namespace QRDInternal



////////////////////////////////////////////////////////////////////////////////
/////////////////////////// User facing QRD functions //////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*
  Computes Q and R matrices such that A=QR where:
  - A is the input matrix
  - Q is a unitary/orthogonal matrix
  - R is an upper triangular matrix

  This function implements a OneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan 
  Pasca.

  Each matrix (input and output) are represented using vectors in a column
  fashion.

  Function arguments:
  - AMatrix:   The input matrix. Interpreted as a transposed matrix.
  - QRMatrix:  The output matrix. The function will overwrite this matrix.
                The first values of this output vector will contain the upper
                triangular values of the R matrix, row by row.
                e.g. for a 4x4 QRD, QRMatrix[5] will contain R[1][1].
                There are exactly N*(N+1)/2 elements of R.
                So rest of the values hold the transposed matrix Q (N*N).
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
void QRDecomposition( vector<ac_complex<T>> &AMatrix, 
                      vector<ac_complex<T>> &QMatrix,
                      vector<ac_complex<T>> &RMatrix,
                      queue &q, 
                      size_t matrices, 
                      size_t reps) {

  constexpr bool isComplex = true;
  QRDInternal::QRDecomposition_impl<columns, rows, rawLatency, isComplex, T>
                                (AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}

// Real single precision floating-point QR Decomposition
template<unsigned columns, unsigned rows, unsigned rawLatency, typename T>
void QRDecomposition( vector<T> &AMatrix, 
                      vector<T> &QMatrix,
                      vector<T> &RMatrix,
                      queue &q, 
                      size_t matrices, 
                      size_t reps) {

  constexpr bool isComplex = false;
  QRDInternal::QRDecomposition_impl<columns, rows, rawLatency, isComplex, T>
                                (AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}
