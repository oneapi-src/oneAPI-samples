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
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#include <CL/sycl/INTEL/ac_types/ac_complex.hpp>
#include <chrono>
#include <cstring>
#include <vector>
#include <type_traits>

#include "Tuple.hpp"
#include "UnrolledLoop.hpp"
#include "Utils.hpp"
#include "StreamingQRD.hpp"
#include "StreamingQRI.hpp"
#include "MemoryTransfers.hpp"

using std::vector;
using namespace sycl;

// namespace QRIInternal{

  // Forward declare the kernel name
  // (This prevents unwanted name mangling in the optimization report.)
  class QRI;
       
  template< unsigned columns,    // Number of columns in the input matrix
            unsigned rows,       // Number of rows in the input matrix
            unsigned rawLatency, // RAW latency for triangular loop optimization
            bool isComplex,      // Selects between ac_complex<T> and T datatype
            typename T>          // The datatype for the computation
  void QRI_impl(  
            vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                      &A_matrix, 
            vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                     &inverse_matrix,
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
    
    using dim = QRInversionDim<isComplex, rows, columns, rawLatency>;

    constexpr int kAMatrixSize = dim::AMatrixSize;
    constexpr int kInverseMatrixSize = dim::InverseMatrixSize;

    // Number of buffers to allocate to be able to read/compute/store 
    // without overlap.
    constexpr short kNumBuffers = 3;
    
    // using AP = sycl::ext::intel::pipe<class APipe, PipeType, 4>;
    using AP = sycl::ext::intel::pipe<class APipe, column<rows, TT>, 4>;
    // using QP = sycl::ext::intel::pipe<class QPipe, TT, 4>;
    using QP = sycl::ext::intel::pipe<class QPipe, column<rows, TT>, 4>;
    using RP = sycl::ext::intel::pipe<class RPipe, TT, 4>;
    using IP = sycl::ext::intel::pipe<class IPipe, column<rows, TT>, 4>;

    // We will process 'chunk' number of matrices in each run of the kernel
    short chunk = 2048;
    if (matrices % chunk) {
      chunk = 1;
    }

    // Create buffers and allocate space for them.
    buffer<TT, 1> *A_buffer[kNumBuffers];
    buffer<TT, 1> *inverse_buffer[kNumBuffers];
    for (short i = 0; i < kNumBuffers; i++) {
      A_buffer[i] = new buffer<TT, 1>(kAMatrixSize * chunk);
      inverse_buffer[i] = new buffer<TT, 1>(kInverseMatrixSize * chunk);
    }

    // Repeat the computation multiple times (for performance analysis)
    for (size_t r = 0; r < reps; r++) {

      // Go over all the matrices, rotating buffers every time
      for (size_t bufferIdx = 0, it = 0; it < matrices; 
                      it += chunk, bufferIdx = (bufferIdx + 1) % kNumBuffers) {

        // Pointer to current input/output matrices in host memory 
        const TT *kPtrA = A_matrix.data() + kAMatrixSize * it;
        TT *kPtrInverse = inverse_matrix.data() + kInverseMatrixSize * it;

        // int matrices = chunk;

        // Copy a new input matrix from the host memory into the FPGA DDR 
        q.submit([&](handler &h) {
          auto A_matrix2 =
       A_buffer[bufferIdx]->template get_access<access::mode::discard_write>(h);
          h.copy(kPtrA, A_matrix2);
        });

        DDRToLocalMemoryCopy< class QRI_DDR_to_local_mem, 
                              isComplex,
                              TT,
                              rows,
                              columns,
                              AP,
                              kNumBuffers>
                              (q, A_buffer, bufferIdx);

        StreamingQRDKernel<   class QRD_compute, 
                              isComplex, 
                              T, 
                              rawLatency, 
                              rows, 
                              columns,
                              AP,
                              QP,
                              RP>(q);

        StreamingQRIKernel<   class QRI_compute, 
                              isComplex, 
                              T, 
                              rawLatency, 
                              rows, 
                              columns,
                              QP,
                              RP,
                              IP>(q);

        LocalMemoryToDDRCopy< class QRI_local_mem_to_DDR, 
                              isComplex,
                              TT,
                              rows,
                              columns,
                              IP,
                              kNumBuffers>
                              (q, inverse_buffer, bufferIdx);

        // Copy the output result from the FPGA DDR to the host memory
        q.submit([&](handler &h) {
          accessor final_inverse_matrix(*inverse_buffer[bufferIdx], h, read_only);
          h.copy(final_inverse_matrix, kPtrInverse);
        });


      } // end for it=0:matrices-1 
    } // end for r=0:reps-1 

    // Clean allocated buffers
    for (short b = 0; b < kNumBuffers; b++) {
      delete A_buffer[b];
      delete inverse_buffer[b];
    }
  }

// } // namespace QRIInternal



////////////////////////////////////////////////////////////////////////////////
/////////////////////////// User facing QRI functions //////////////////////////
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
  - A_matrix:   The input matrix. Interpreted as a transposed matrix.
  - inverse_matrix:  The output matrix. The function will overwrite this matrix.
                The first values of this output vector will contain the upper
                triangular values of the R matrix, row by row.
                e.g. for a 4x4 QRI, inverse_matrix[5] will contain R[1][1].
                There are exactly N*(N+1)/2 elements of R.
                So rest of the values hold the transposed matrix Q (N*N).
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read sequentially from the A_matrix 
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
void QRI( vector<ac_complex<T>> &A_matrix, 
                  vector<ac_complex<T>> &inverse_matrix,
                  queue &q, 
                  size_t matrices, 
                  size_t reps) {

  constexpr bool isComplex = true;
  QRI_impl<columns, rows, rawLatency, isComplex, T>
                                      (A_matrix, inverse_matrix, q, matrices, reps); 
}

// Real single precision floating-point QR Decomposition
template<unsigned columns, unsigned rows, unsigned rawLatency, typename T>
void QRI( vector<T> &A_matrix, 
                  vector<T> &inverse_matrix,
                  queue &q, 
                  size_t matrices, 
                  size_t reps) {

  constexpr bool isComplex = false;
  QRI_impl<columns, rows, rawLatency, isComplex, T>
                                      (A_matrix, inverse_matrix, q, matrices, reps); 
}
