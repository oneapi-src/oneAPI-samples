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
#include "StreamingQRI.hpp"
#include "MemoryTransfers.hpp"
#include "QRInversionDim.hpp"

using std::vector;
using namespace sycl;

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
  constexpr int kNumElementsPerDDRBurst = isComplex ? 4 : 8;

  // Number of buffers to allocate to be able to read/compute/store 
  // without overlap.
  constexpr short kNumBuffers = 3;
  
  using AMatrixPipe = sycl::ext::intel::pipe<class APipe, column<rows, TT>, 4>;
  using QMatrixPipe = sycl::ext::intel::pipe<class QPipe, column<rows, TT>, 4>;
  using RMatrixPipe = sycl::ext::intel::pipe<class RPipe, TT, 4>;
  using InverseMatrixPipe = sycl::ext::intel::pipe<class IPipe, column<rows, TT>, 4>;

  // We will process 'chunk' number of matrices in each run of the kernel
  short chunk = 2048;
  if (matrices % chunk) {
    chunk = 1;
  }

  // Create buffers and allocate space for them.
  buffer<TT, 1> *ABuffer[kNumBuffers];
  buffer<TT, 1> *inverseBuffer[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    ABuffer[i] = new buffer<TT, 1>(kAMatrixSize * chunk);
    inverseBuffer[i] = new buffer<TT, 1>(kInverseMatrixSize * chunk);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {

    // Go over all the matrices, rotating buffers every time
    for (size_t bufferIdx = 0, it = 0; it < matrices; 
                    it += chunk, bufferIdx = (bufferIdx + 1) % kNumBuffers) {

      constexpr int matricesPerIter = 1;

      // Pointer to current input/output matrices in host memory 
      const TT *kPtrA = A_matrix.data() + kAMatrixSize * it;
      TT *kPtrInverse = inverse_matrix.data() + kInverseMatrixSize * it;

      // int matrices = chunk;

      // Copy a new input matrix from the host memory into the FPGA DDR 
      q.submit([&](handler &h) {
        auto A_matrix2 =
     ABuffer[bufferIdx]->template get_access<access::mode::discard_write>(h);
        h.copy(kPtrA, A_matrix2);
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
                          rawLatency, 
                          matricesPerIter,
                          AMatrixPipe,
                          QMatrixPipe,
                          RMatrixPipe>(q);

      StreamingQRIKernel< class QRI_compute, 
                          isComplex, 
                          T, 
                          rawLatency, 
                          rows, 
                          columns,
                          QMatrixPipe,
                          RMatrixPipe,
                          InverseMatrixPipe>(q);

      // Read the Q matrix from the QMatrixPipe pipe and copy it to the
      // FPGA DDR
      MatrixReadPipeByColumnsToDDR< class QRI_local_mem_to_DDR, 
                                    TT,
                                    rows,
                                    columns,
                                    kNumElementsPerDDRBurst,
                                    matricesPerIter,
                                    InverseMatrixPipe>
                                    (q, inverseBuffer[bufferIdx]);

      // Copy the output result from the FPGA DDR to the host memory
      q.submit([&](handler &h) {
        accessor final_inverse_matrix(*inverseBuffer[bufferIdx], h, read_only);
        h.copy(final_inverse_matrix, kPtrInverse);
      });


    } // end for it=0:matrices-1 
  } // end for r=0:reps-1 

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    delete ABuffer[b];
    delete inverseBuffer[b];
  }
}