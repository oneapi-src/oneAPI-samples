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

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <chrono>
#include <cstring>
#include <vector>
#include <type_traits>

#include "UnrolledLoop.hpp"


using std::vector;
using namespace sycl;

namespace QRDInternal{
  /*
    Static implementation of the base 2 logarithm function
  */
  template <typename T>
  static constexpr T Log2(T n) {
    T ret = T(0);
    T val = n;
    while (val > T(1)) {
      val >>= 1;
      ret++;
    }
    return ret;
  }

  /*
    Static implementation of the CEIL base 2 logarithm function
  */
  template<unsigned int N, uint8_t remains=0>
  static constexpr inline unsigned int CeilLog2()
  {
    return (N <= 1) ? remains : 1 + CeilLog2<(N>>1), remains | (N%2)>();
  }

  /*
    Static implementation of the base 2 power function
  */
  template <typename T>
  static constexpr T Pow2(T n) {
    return T(1) << n;
  }

  /*
    Return the number of bits required to encode all the values between 0 and N
  */
  template<unsigned int N>
  static constexpr inline unsigned int BitsForMaxValue()
  {
    return CeilLog2<N+1>();
  }

  /*
    A structure that represents a complex number.
    - xx represents its real value
    - yy represents its imaginary value
    Two operators are overloaded (+, *) to help with manipulating this structure.
  */
  template <typename T>
  struct Complex {
    T xx;
    T yy;

    Complex(){}

    Complex(T x, T y) {
      xx = x;
      yy = y;
    }

    Complex(T v) {
      xx = v;
      yy = v;
    }

    // Complex addition
    const Complex operator+(const Complex rhs) const {
      return Complex(xx + rhs.xx, yy + rhs.yy);
    }

    // Complex multiplication
    const Complex operator*(const Complex rhs) const {
      Complex c;
      c.xx = xx * rhs.xx + yy * rhs.yy;
      c.yy = yy * rhs.xx - xx * rhs.yy;
      return c;
    }
  };

  /*
    A structure that hold a column a of matrix of type T.
  */
  template<unsigned rows, typename T>
  struct column{
    T d[rows];
  };

  /*
    A structure that hold a row a of matrix of type T.
  */
  template<unsigned columns, typename T>
  struct row{
    [[intel::fpga_memory("BLOCK_RAM")]] // NO-FORMAT: Attribute
    T d[columns];
  };

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
                                                                      &A_matrix, 
            vector<typename std::conditional<isComplex, ac_complex<T>, T>::type> 
                                                                     &QR_matrix,
                                       queue &q, size_t matrices, size_t reps) {

    // TT will be ac_complex<T> or T depending on isComplex
    typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;
    // CTT will be Complex<T> or T depending on isComplex
    typedef typename std::conditional<isComplex, Complex<T>, T>::type CTT;

    // Functional limitations
    static_assert(std::is_same<T, float>::value, 
                                            "only float datatype is supported");
    static_assert(rows>=columns, 
         "only rectangular matrices with rows>=columns are matrices supported");
    static_assert((columns <= 512) && (columns >= 4), 
                          "only matrices of size 4x4 to 512x512 are supported");
    // static_assert(columns%4 == 0, 
    //             "only matrices of size that are a multiple of 4 are supported");

    // Number of complex elements in the matrix
    constexpr int kAMatrixSize = columns * rows;

    // Sizes of allocated memories for input and output matrix
    // Both the input matrix and Q are full matrices of complex elements
    // R only contains columns + 
    //                 (columns - 1) + 
    //                 (columns - 2) +
    //                 (columns - 3) + 
    //                 etc.
    // So R contains columns * (columns + 1) / 2 complex elements.
    constexpr int kRMatrixSize = columns * (columns + 1) / 2;
    constexpr int kQMatrixSize = kAMatrixSize;
    constexpr int kQRMatrixSize = kQMatrixSize + kRMatrixSize;

    // Constants related to the memory configuration of the kernel's local
    // memories
    // We want 8 floating-point values in each memory bank
    constexpr int kNumElementsPerBank = isComplex || rows < 8 ? 4 : 8;
    // Set the bankwidth in bytes
    constexpr int kBankwidth = kNumElementsPerBank * 8;
    constexpr bool kNonCompleteBank = rows%kNumElementsPerBank != 0;
    constexpr int kExtraBank = kNonCompleteBank ? 1 : 0;
    constexpr int kNumBanks = rows / kNumElementsPerBank + kExtraBank;
    constexpr int kNumBanksNextPow2 = Pow2(CeilLog2<kNumBanks>());

    // Number of load and store iterations for a single matrix given the size
    // of the input matrices and the number of elements per bank
    constexpr bool kNonCompleteIter = rows%kNumElementsPerBank != 0;
    constexpr int kExtraIter = kNonCompleteIter ? 1 : 0;
    constexpr int kLoadIter = ((rows / kNumElementsPerBank) + kExtraIter) 
                                                                      * columns;
    constexpr int kStoreIter = kLoadIter;
    // Number of bits required by the loop counters for the load/store iterators
    constexpr int kLoadIterBitSize = BitsForMaxValue<kLoadIter + 1>();
    constexpr int kStoreIterBitSize = BitsForMaxValue<kStoreIter + 1>();
    // The indexes kLoadIter and kStoreIter iterators are being divided 
    // by kNumBanks. So we precompute the size of the output.
    constexpr int kLiNumBankBitSize = kLoadIterBitSize - Log2(kNumBanks);
    constexpr int kSiNumBankBitSize = kStoreIterBitSize - Log2(kNumBanks);

    // Number of buffers to allocate to be able to read/compute/store 
    // without overlap.
    // TODO: We technically need 3, but having 4 maybe improves FMax?
    //       Having 3 seems to improve latency without compromising FMax.
    constexpr short kNumBuffers = 3;
      
    constexpr int kNValue = columns;
    // Number of iterations performed without any dummy work added for the 
    // triangular loop optimization
    constexpr int kVariableIterations = kNValue - rawLatency;
    // Total number of dummy iterations
    constexpr int kDummyIterations = rawLatency > columns ?
                (columns - 1) * columns / 2 + (rawLatency - columns) * columns :
                rawLatency * (rawLatency - 1) / 2;

    // Total number of iterations (including dummy iterations)
    constexpr int kIterations = columns +
                                columns * (columns+1) / 2 +  
                                kDummyIterations;

    // Sizes in bits for the triangular loop indexes
    // i starts from -1 and goes up to rows
    // So we need:
    // -> enough bits to encode rows+1 for the positive iterations and 
    //    the exit condition
    // -> one extra bit for the -1
    constexpr int kIBitSize = BitsForMaxValue<rows + 1>() + 1;
    // j starts from i, so from -1 and goes up to columns
    // So we need:
    // -> enough bits to encode columns+1 for the positive iterations and 
    //    the exit condition
    // -> one extra bit for the -1
    // But j may start below -1 if we perform more dummy iterations than the 
    // number of columns in the matrix.
    // In that case, we need:
    // -> enough bits to encode columns+1 for the positive iterations and 
    //    the exit condition
    // -> enough bits to encode the maximum number of negative iterations
    constexpr int kJNegativeIterations = 
                            kVariableIterations < 0 ? -kVariableIterations : 1;
    constexpr int kJBitSize = BitsForMaxValue<columns + 1>() 
                                      + BitsForMaxValue<kJNegativeIterations>();

    // We will process 'chunk' number of matrices in each run of the kernel
    short chunk = 2048;
    if (matrices % chunk) {
      chunk = 1;
    }

    // Create buffers and allocate space for them.
    buffer<TT, 1> *A_buffer[kNumBuffers];
    buffer<TT, 1> *QR_buffer[kNumBuffers];
    for (short i = 0; i < kNumBuffers; i++) {
      A_buffer[i] = new buffer<TT, 1>(kAMatrixSize * chunk);
      QR_buffer[i] = new buffer<TT, 1>(kQRMatrixSize * chunk);
    }

    // Repeat the computation multiple times (for performance analysis)
    for (size_t r = 0; r < reps; r++) {

      // Go over all the matrices, rotating buffers every time
      for (size_t bufferIdx = 0, it = 0; it < matrices; 
                      it += chunk, bufferIdx = (bufferIdx + 1) % kNumBuffers) {

        // Pointer to current input/output matrices in host memory 
        const TT *kPtrA = A_matrix.data() + kAMatrixSize * it;
        TT *kPtrQR = QR_matrix.data() + kQRMatrixSize * it;

        int matrices = chunk;

        // Copy a new input matrix from the host memory into the FPGA DDR 
        q.submit([&](handler &h) {
          auto A_matrix2 =
       A_buffer[bufferIdx]->template get_access<access::mode::discard_write>(h);
          h.copy(kPtrA, A_matrix2);
        });

        // Compute job
        q.submit([&](handler &h) {

          // Create accessors to the FPGA DDR buffers
          accessor A_matrix_accessor(*A_buffer[bufferIdx], h, read_only);
          accessor QR_matrix_accessor(*QR_buffer[bufferIdx], h, write_only, 
                                                                      no_init);

          // Create alias to the output matrix accessor
          auto QR_matrix_accessor_2 = QR_matrix_accessor;

          sycl::stream out(64000, 64000, h);

          h.single_task<class QRD>([=]() [[intel::kernel_args_restrict]] {

            // Go over the matrices
            for (int matrixIdx = 0; matrixIdx < matrices; matrixIdx++) {

              // out << "matrixIdx " << matrixIdx << sycl::endl;

              // Instantiate 3 versions of the input matrix
              // There are three loops that:
              // - load the input matrix into A_load
              // - read A_load and do the computation on A_compute, then 
              //   writes the results in A_store
              // - writes A_store into the output matrix
              [[intel::bankwidth(kBankwidth)]] // NO-FORMAT: Attribute
              [[intel::numbanks(kNumBanksNextPow2)]]   // NO-FORMAT: Attribute
              column<rows, CTT>  A_load[columns], 
                              // A_compute[columns], 
                              A_store[columns];

              row<columns, CTT> A_compute[rows];
              // CTT A_compute[rows*columns]; 

              /*
                ================================================================
                Loop 1: Copy data from DDR memory to on-chip memory.
                ================================================================
              */
              // Get the index of the first bank of the current matrix l
              int loadBankIndex = matrixIdx * kAMatrixSize;

              [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
              for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; 
                                                                        li++) {
                // Load a single bank of the input matrix 
                CTT bank[kNumElementsPerBank];

                bool lastRow = false;

                if constexpr(kNonCompleteIter){
                  constexpr int kLoadItersPerColumn = rows/kNumElementsPerBank 
                                                                  + kExtraIter; 
                  lastRow = (li%kLoadItersPerColumn) == kLoadItersPerColumn - 1;
                }

                UnrolledLoop<kNumElementsPerBank>([&](auto k) {

                  bool outOfBounds = false;
                  if constexpr(kNonCompleteIter){
                   outOfBounds = lastRow && 
                 ((k % kNumElementsPerBank) > ((rows-1) % kNumElementsPerBank));
                  }

                  if(!outOfBounds){
                    if constexpr(isComplex){
                      bank[k].xx = A_matrix_accessor[
                                    loadBankIndex + k].r();
                      bank[k].yy = A_matrix_accessor[
                                    loadBankIndex + k].i();
                    }
                    else{
                      bank[k] = A_matrix_accessor[
                                         loadBankIndex + k];
                    }
                  }
                });

                if constexpr(kNonCompleteIter){
                  int readElements = (rows % kNumElementsPerBank != 0) 
                                                && lastRow ?
                                                rows % kNumElementsPerBank :  
                                                kNumElementsPerBank;

                  // Increase the bank index
                  loadBankIndex += readElements;
                }
                else{
                  loadBankIndex += kNumElementsPerBank;
                }

                // Write the current bank to the A_load matrix.
                ac_int<BitsForMaxValue<kNumBanks>(), false> jtmp = 
                                                              li % (kNumBanks);
                ac_int<kLiNumBankBitSize, false> liNumBank = li / kNumBanks;

                UnrolledLoop<kNumBanks>([&](auto k) {
                  UnrolledLoop<kNumElementsPerBank>([&](auto t) {
                    constexpr auto rowIdx = k * kNumElementsPerBank + t;
                    if constexpr (rowIdx < rows){
                      if ((jtmp == k)) {
                        if constexpr(isComplex){
                          A_load[liNumBank].d[rowIdx].xx = bank[t].xx;
                          A_load[liNumBank].d[rowIdx].yy = bank[t].yy;
                        }
                        else{
                          A_load[liNumBank].d[rowIdx] = bank[t];
                        }
                      }
                    }

                    // Delay data signals to create a vine-based data 
                    // distribution to lower signal fanout.
                    if constexpr(isComplex){
                      bank[t].xx = sycl::ext::intel::fpga_reg(bank[t].xx);
                      bank[t].yy = sycl::ext::intel::fpga_reg(bank[t].yy);
                    }
                    else{
                      bank[t] = sycl::ext::intel::fpga_reg(bank[t]);
                    }
                  });

                  jtmp = sycl::ext::intel::fpga_reg(jtmp);
                });
              }

      // out << "A_load MATRIX" << sycl::endl;
      // for (size_t i = 0; i < ROWS_COMPONENT; i++) {
      //   for (size_t j = 0; j < COLS_COMPONENT; j++) {
      //     out << A_load[j].d[i] << " ";
      //   }
      //   out << sycl::endl;
      // }


              /*
                ================================================================
                Loop 2: Main computation the QR Decomposition.
                ================================================================
              
                Main computation of the QR Decomposition.

                This code implements a OneAPI optimized variation of the 
                following algorithm:

                for i=0:n
                  for j=max(i,1):n

                    if(j==i)
                      Q_i = a_i*ir
                    else
                      if(i>=0)
                        a_j = a_j - s[j]*a_i

                      if j=i+1
                        pip1         = <a_{i+1},a_{i+1}>
                        ir           = 1/sqrt(pip1)
                        R_{i+1,i+1}  = sqrt(pip1)
                      else
                        p            = <a_{i+1}, a_j>
                        s[j]         = p/pip1
                        R_{i+1,j}    = p*ir


                Where:
                -> X_i represents the column i of the matrix X
                -> <x,y> represents the dot product of the vectors x and y
              */

              // Get index in the output at which to start writing the outputs 
              // for input matrix l.
              int qr_idx = matrixIdx * kQRMatrixSize;

              // a local copy of a_{i+1} that is used across multiple j 
              // iterations for the computation of pip1 and p
              CTT a_ip1[rows];
              // a local copy of a_ip1 that is used across multiple j iterations 
              // for the computation of a_j
              CTT a_i[rows];
              // Depending on the context, will contain:
              // -> -s[j]: for all the iterations to compute a_j
              // -> ir: for one iteration per j iterations to compute Q_i
              constexpr int super_dummy_iterations = rawLatency - columns;
              constexpr int increasedBufferSize = super_dummy_iterations < 0 ? 
                                                    0 : super_dummy_iterations; 
              CTT s_or_i[columns + increasedBufferSize];

              // Adding increasedBufferSize is a waste of resource because we 
              // are going to read and write only to "columns" different places
              // If we don't add it, the compiler does not achieve II 1 because 
              // it is not able to determine that we are not going to read at 
              // the same location the last iteration just wrote.
              // This is probably due to the fact that when the access index is 
              // negative, we are not actually reading/writing to it (gated by 
              // an if statement) but it seems to make the compiler confused.
             
              T pip1, ir;

              // The triangular loop over i and j has been optimized using the 
              // method described in the triangular loop optimization tutorial.
              // Therefore, the code iterates over the total number of 
              // iterations, including "dummy" ones (that ensures II=1).
              ac_int<kIBitSize, true> i = -1;
              ac_int<kJBitSize, true> j = 0;
              
              [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
              [[intel::ivdep(rawLatency)]]  // NO-FORMAT: Attribute
              // [[intel::speculated_iterations(0)]]
              for (int s = 0; s < kIterations; s++) {

                ac_int<kIBitSize, true> next_i;
                ac_int<kJBitSize, true> next_j;
                // Update the loop indexes.
                if (j == kNValue - 1) {
                  // If i reached an index at which the j inner loop don't have
                  // enough time to write its result for the next i iteration,
                  // some "dummy" iterations are introduced 
                  next_j = (kVariableIterations > i) ? 
                                  ac_int<kJBitSize, true>{i + 1} : 
                                  ac_int<kJBitSize, true>{kVariableIterations};
                  next_i = i + 1;
                } else {
                  next_j = j + 1;
                  next_i = i;
                }


                // Temporary storage for a column of the input matrix and for
                // partial results.
                CTT col[rows];

                // Current value of s_or_i depending on the value of j
                // It is replicated kNumBanks times to reduce fanout
                CTT sori[kNumBanks];

                // All the control signals are precomputed and replicated
                // kNumBanks times to reduce fanout
                bool  j_eq_i[kNumBanks], 
                      i_gt_0[kNumBanks],
                      i_ge_0_j_ge_i[kNumBanks], 
                      j_eq_i_plus_1[kNumBanks],
                      i_lt_0[kNumBanks];

                UnrolledLoop<kNumBanks>([&](auto k) {
                  i_gt_0[k] = INTEL::fpga_reg(i > 0);
                  i_lt_0[k] = INTEL::fpga_reg(i < 0);
                  j_eq_i[k] = INTEL::fpga_reg(j == i);
                  i_ge_0_j_ge_i[k] = INTEL::fpga_reg(i >= 0 && j >= i);
                  j_eq_i_plus_1[k] = INTEL::fpga_reg(j == i + 1);
                  int idx = j + increasedBufferSize;
                  if constexpr(isComplex){
                    sori[k].xx = INTEL::fpga_reg(s_or_i[idx].xx);
                    sori[k].yy = INTEL::fpga_reg(s_or_i[idx].yy);
                  }
                  else{
                    sori[k] = INTEL::fpga_reg(s_or_i[idx]);
                  }
                });

                // Preload col and a_i with the correct data for the current 
                // iteration.
                // These are going to be use to compute the dot product of 
                // two different column of the input matrix.
                UnrolledLoop<rows>([&](auto k) {
                  // find which bank this unrolled iteration is going to use
                  constexpr auto bank = k / kNumElementsPerBank;

                  // Load col with the current column of matrix a.
                  // At least one iteration of the outer loop i is required
                  // for the "working copy" A_compute to contain data.
                  // If no i iteration elapsed, we must read the column of 
                  // matrix a directly from the A_load col then contains a_j
                  if constexpr(isComplex){
                    if(i_gt_0[bank]){
                      // col[k].xx = A_compute[j].d[k].xx;
                      // col[k].yy = A_compute[j].d[k].yy;

                      // col[k].xx = A_compute[int(j) + k*columns].xx;
                      // col[k].yy = A_compute[int(j) + k*columns].yy;

                      col[k].xx = A_compute[k].d[j].xx;
                      col[k].yy = A_compute[k].d[j].yy;
                    }
                    else{
                      col[k].xx = A_load[j].d[k].xx;
                      col[k].yy = A_load[j].d[k].yy;
                    }
                    
                    // Load a_i for reuse across j iterations
                    if (j_eq_i[bank]) {
                      a_i[k].xx = col[k].xx;
                      a_i[k].yy = col[k].yy;
                    }
                  }
                  else{
                    if(i_gt_0[bank]){
                      // col[k] = A_compute[int(j) + k*columns];
                      // col[k] = A_compute[j].d[k];
                      col[k] = A_compute[k].d[j];
                    }
                    // Using an else statement makes the compiler throw an
                    // inexplicable warning:
                    // "Compiler Warning: Memory instruction with unresolved 
                    // pointer may lead to bad QoR."
                    if(!i_gt_0[bank]){
                      col[k] = A_load[j].d[k];
                    }

                    // Load a_i for reuse across j iterations
                    if (j_eq_i[bank]) {
                      a_i[k] = col[k];
                    }
                  }

                });

                UnrolledLoop<rows>([&](auto k) {
                  // find which bank this unrolled iteration is going to use
                  constexpr auto bankIdx = k / kNumElementsPerBank;

                  // Depending on the iteration this code will compute either:
                  // -> If i=j, a column of Q: Q_i = a_i*ir
                  //    In that case, no term is added to the mult_add construct
                  // -> If i!=j, an updated column of a: a_j - s[j]*a_i
                  //    There is a special case if i<0 where a_j is unmodified 
                  //    but the i iteration is still required to fill ir and s 
                  //    for subsequent iterations
                  auto prod_lhs = a_i[k];
                  auto prod_rhs = i_lt_0[bankIdx] ? CTT(0.0) : sori[bankIdx];
                  auto add = j_eq_i[bankIdx] ? CTT(0.0) : col[k];
                  col[k] = prod_lhs * prod_rhs + add;

                  // Store Q_i in A_store and the modified a_j in A_compute
                  // To reduce the amount of control, A_store and A_compute
                  // are both written to for each iteration of i>=0 && j>=i
                  // In fact:
                  // -> A_store could only be written to at iterations i==j
                  // -> A_compute could only be written to at iterations 
                  //    j!=i && i>=0  
                  // The extra writes are harmless as the locations written to 
                  // are either going to be:
                  // -> overwritten for the matrix Q (A_store)
                  // -> unused for the A_compute
                  if (i_ge_0_j_ge_i[bankIdx]) {
                    if constexpr(isComplex){
                      // A_store[j].d[k].xx = A_compute[j].d[k].xx = col[k].xx;
                      // A_store[j].d[k].yy = A_compute[j].d[k].yy = col[k].yy;
                      // TODO: Remove A_store?
                      // A_store[j].d[k].xx = A_compute[int(j) + k*columns].xx = col[k].xx;
                      // A_store[j].d[k].yy = A_compute[int(j) + k*columns].yy = col[k].yy;  
                      A_store[j].d[k].xx = A_compute[k].d[j].xx = col[k].xx;
                      A_store[j].d[k].yy = A_compute[k].d[j].yy = col[k].yy;                      
                    }
                    else{
                      // A_store[j].d[k] = A_compute[j].d[k] = col[k];
                      // A_store[j].d[k] = A_compute[int(j) + k*columns] = col[k];
                      A_store[j].d[k] = A_compute[k].d[j] = col[k];
                    }
                  }

                  // Store a_{i+1} for subsequent iterations of j
                  if (j_eq_i_plus_1[bankIdx]) {
                    a_ip1[k] = col[k];
                  }
                });

                // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
                CTT p_ij = CTT(0.0);

                UnrolledLoop<rows>([&](auto k) {
                  p_ij = p_ij + col[k] * a_ip1[k];
                });

                if (j == i + 1) {
                  if constexpr(isComplex){
                    pip1 = p_ij.xx;
                    ir = rsqrt(p_ij.xx);
                  }
                  else{
                    pip1 = p_ij;
                    ir = rsqrt(p_ij); 
                  }
                }

                CTT s_j;
                if constexpr(isComplex){
                  s_j = CTT(0.0f - (p_ij.xx) / pip1, p_ij.yy / pip1);
                }
                else{
                  s_j = - p_ij / pip1;
                }

                // j may be negative if the number of "dummy" iterations is 
                // larger than the matrix size
                if (j >= 0) {
                  int idx = j + increasedBufferSize;
                  if constexpr(isComplex){
                    s_or_i[idx] = CTT(j == i + 1 ? ir : s_j.xx,
                                        j == i + 1 ? 0.0f : s_j.yy);
                  }
                  else{
                    s_or_i[idx] = j == i + 1 ? ir : s_j; 
                  }
                }

                // Compute the R_{i+1,i+1} or R_{i+1,j} 
                CTT r_ip1j;
                if constexpr(isComplex){
                  r_ip1j = j == i + 1 ? CTT(sycl::sqrt(pip1), 0.0) : 
                                                CTT(ir * p_ij.xx, ir * p_ij.yy);
                }
                else{
                  r_ip1j = j == i + 1 ? sycl::sqrt(pip1) : ir * p_ij;
                }

                // Write the computed R value when j is not a "dummy" iteration
                // introduced to optimized the triangular loop
                if (j >= i + 1 && i + 1 < kNValue) {
                  if constexpr(isComplex){
                    QR_matrix_accessor_2[qr_idx] = {r_ip1j.xx, r_ip1j.yy};
                  }
                  else{
                    QR_matrix_accessor_2[qr_idx] = r_ip1j;
                  }
                  qr_idx++;
                }



                // // Update the loop indexes.
                // if (j == kNValue - 1) {
                //   // If i reached an index at which the j inner loop don't have
                //   // enough time to write its result for the next i iteration,
                //   // some "dummy" iterations are introduced 
                //   j = (kVariableIterations > i) ? 
                //                   ac_int<kJBitSize, true>{i + 1} : 
                //                   ac_int<kJBitSize, true>{kVariableIterations};
                //   i++;
                // } else {
                //   j++;
                // }




                j = next_j;
                i = next_i;

              } // end for s=0:kIterations-1

              /*
                ================================================================
                Loop 3: Copy the result from on-chip memory to DDR memory.
                ================================================================
              */

              [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
              for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                        si++) {
                // Only one bank is going to be stored per si iteration
                // To reduce fanout on si, a "get" table will contain for each 
                // bank a boolean to check if it should store this si iteration
                ac_int<BitsForMaxValue<kNumBanks>(), false> desired = 
                                                              si % (kNumBanks);
                bool get[kNumBanks];
                UnrolledLoop<kNumBanks>([&](auto k) {
                  get[k] = desired == k;
                  desired = INTEL::fpga_reg(desired);
                });

                // Each bank will then check the get table to potentially 
                // read kNumElementsPerBank from A_store and store the elements
                // in bank
                CTT bank[kNumElementsPerBank] = {0};

                ac_int<kSiNumBankBitSize, false> siNumBank = si / kNumBanks;
                UnrolledLoop<kNumBanks>([&](auto t) {
                  UnrolledLoop<kNumElementsPerBank>([&](auto k) {
                    constexpr auto rowIdx = t * kNumElementsPerBank + k;
                    if constexpr(rowIdx < rows){
                      if constexpr(isComplex){
                        bank[k].xx = get[t] ? A_store[siNumBank].d[rowIdx].xx : 
                                                    INTEL::fpga_reg(bank[k].xx);
                        bank[k].yy = get[t] ? A_store[siNumBank].d[rowIdx].yy : 
                                                    INTEL::fpga_reg(bank[k].yy);
                      }
                      else{
                        bank[k] = get[t] ? A_store[siNumBank].d[rowIdx] : 
                                                      INTEL::fpga_reg(bank[k]);
                      }
                    }
                  });
                });

                bool lastRow = false;
                if constexpr(kNonCompleteIter){
                  lastRow = si % kNumBanks == kNumBanks-1; 
                } 

                // Finally, the kNumElementsPerBank elements from bank are 
                // written to the QR_matrix_accessor
                UnrolledLoop<kNumElementsPerBank>([&](auto k) {
                  bool outOfBounds = false;
                  if constexpr(kNonCompleteIter){
                    outOfBounds = lastRow && 
                          (k > ((rows-1) % kNumElementsPerBank));
                  }

                  if(!outOfBounds){
                    if constexpr(isComplex){
                      QR_matrix_accessor[qr_idx + k] = {bank[k].xx, bank[k].yy};
                    }
                    else{
                      QR_matrix_accessor[qr_idx + k] = bank[k];
                    }
                  }
                });

                if constexpr(kNonCompleteIter){
                  int wroteElements = lastRow ? rows % kNumElementsPerBank :  
                                                            kNumElementsPerBank;
                  qr_idx += wroteElements;
                }
                else{
                  qr_idx += kNumElementsPerBank;
                }                
              } // end for si=0:kStoreIter-1
            } // end for matrixIdx=0:matrices-1
          });
        });

        // Copy the output result from the FPGA DDR to the host memory
        q.submit([&](handler &h) {
          accessor final_QR_matrix(*QR_buffer[bufferIdx], h, read_only);
          h.copy(final_QR_matrix, kPtrQR);
        });


      } // end for it=0:matrices-1 
    } // end for r=0:reps-1 

    // Clean allocated buffers
    for (short b = 0; b < kNumBuffers; b++) {
      delete A_buffer[b];
      delete QR_buffer[b];
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
  - A_matrix:   The input matrix. Interpreted as a transposed matrix.
  - QR_matrix:  The output matrix. The function will overwrite this matrix.
                The first values of this output vector will contain the upper
                triangular values of the R matrix, row by row.
                e.g. for a 4x4 QRD, QR_matrix[5] will contain R[1][1].
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
void QRDecomposition( vector<ac_complex<T>> &A_matrix, 
                      vector<ac_complex<T>> &QR_matrix,
                      queue &q, 
                      size_t matrices, 
                      size_t reps) {

  constexpr bool isComplex = true;
  QRDInternal::QRDecomposition_impl<columns, rows, rawLatency, isComplex, T>
                                      (A_matrix, QR_matrix, q, matrices, reps); 
}

// Real single precision floating-point QR Decomposition
template<unsigned columns, unsigned rows, unsigned rawLatency, typename T>
void QRDecomposition( vector<T> &A_matrix, 
                      vector<T> &QR_matrix,
                      queue &q, 
                      size_t matrices, 
                      size_t reps) {

  constexpr bool isComplex = false;
  QRDInternal::QRDecomposition_impl<columns, rows, rawLatency, isComplex, T>
                                      (A_matrix, QR_matrix, q, matrices, reps); 
}