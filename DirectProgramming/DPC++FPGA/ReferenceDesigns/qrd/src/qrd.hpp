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

// Forward declare the kernel name
// (This prevents unwanted name mangling in the optimization report.)
class QRD;

/*
  Complex single precision floating-point QR Decomposition
  Computes Q and R matrices such that A=QR where:
  - A is the input matrix
  - Q is a unitary matrix
  - R is an upper triangular matrix

  This function implements a OneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan 
  Pasca.

  Each matrix (input and output) are represented using floating-point vectors.
  Each complex element is interpreted as two consecutive values.
  e.g. A_matrix[2*i + 0] and A_matrix[2*i + 1] represent the real and
  imaginary values of the i-th complex element of the matrix.

  Function arguments:
  - A_matrix:  The input matrix. Interpreted as a transposed matrix.
  - QR_matrix: The output matrix. The function will overwrite this matrix.
                The first values of this output vector will contain the upper
                triangular values of the R matrix, row by row.
                e.g. for a 4x4 QRD, out_matrix[5*2+1] will contain the imaginary
                part of R[1][1].
                So there are exactly N*(N+1)/2 elements of R.
                So rest of the values hold the transposed matrix Q.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read from the A_matrix vector.
  - reps:       The number of repetitions of the computation to execute.
                (for performance evaluation)

  This function requires the definition of the following global defines:
  - columns:   The number of columns in the matrix
  - rows:   The number of rows in the matrix     
  - raw_latency: The latency between the RAW dependency in the triangular
                      loop that prevents the compiler to achieve an II of 1.
                      This helps create a loop structure that can reach an II
                      of 1 following the triangular loop optimization tutorial
                      method.
  
*/
template<unsigned columns, unsigned rows, unsigned raw_latency, typename T>
void QRDecomposition_impl(  vector<ac_complex<T>> &A_matrix, 
                            vector<ac_complex<T>> &QR_matrix,
                            queue &q, 
                            size_t matrices, 
                            size_t reps,
                            typename std::enable_if<std::is_same<T, float>::value>::type* = 0) {

  // Number of complex elements in the matrix
  constexpr int kNumComplexElements = columns * rows;

  // Sizes of allocated memories for input and output matrix
  // Both the input matrix and Q are full matrices of complex elements
  // R only contains columns + 
  //                 (columns - 1) + 
  //                 (columns - 2) +
  //                 (columns - 3) + 
  //                 etc.
  // So R contains columns * (columns + 1) / 2 complex elements.
  // Each complex element takes two indexes.
  constexpr int kAMatrixSize = kNumComplexElements;
  constexpr int kQMatrixSize = kNumComplexElements;
  constexpr int kRMatrixSize = columns * (columns + 1) / 2;
  constexpr int kQRMatrixSize = kQMatrixSize + kRMatrixSize;

  // Constants related to the memory configuration of the kernel's local
  // memories
  // We want 4 complex elements (8 floating-point values) in each memory bank
  constexpr int kNumElementsPerBank = 4;
  // Set the bankwidth in bytes
  constexpr int kBankwidth = kNumElementsPerBank * 8;
  constexpr int kNumBanks = rows / kNumElementsPerBank;
  constexpr int kNumBanksNextPow2 = Pow2(CeilLog2<kNumBanks>());

  // Number of load and store iterations for a single matrix given the size
  // of the input matrices and the number of complex elements per banks
  constexpr int kLoadIter = kNumComplexElements / kNumElementsPerBank;
  constexpr int kStoreIter = kNumComplexElements / kNumElementsPerBank;
  // Number of bits required by the loop counters for the loads/stores iterators
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
    
  // Number of iterations performed without any dummy work added for the 
  // triangular loop optimization
  constexpr int kNValue = columns;
  constexpr int kVariableIterations = kNValue - raw_latency;
  // Total number of dummy iterations
  constexpr int kDummyIterations = raw_latency > columns ?
          (columns - 1) * columns / 2 + (raw_latency - columns) * columns :
          raw_latency * (raw_latency - 1) / 2;

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
  buffer<ac_complex<T>, 1> *A_buffer[kNumBuffers];
  buffer<ac_complex<T>, 1> *QR_buffer[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    A_buffer[i] = new buffer<ac_complex<T>, 1>(kAMatrixSize * chunk);
    QR_buffer[i] = new buffer<ac_complex<T>, 1>(kQRMatrixSize * chunk);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {

    // Go over all the matrices, rotating buffers every time
    for (size_t b = 0, it = 0; it < matrices; 
                                      it += chunk, b = (b + 1) % kNumBuffers) {

      // Pointer to current input/output matrices in host memory 
      const ac_complex<T> *kPtrA = A_matrix.data() + kAMatrixSize * it;
      ac_complex<T> *kPtrQR = QR_matrix.data() + kQRMatrixSize * it;

      int matrices = chunk;

      // Copy a new input matrix from the host memory into the FPGA DDR 
      q.submit([&](handler &h) {
        auto A_matrix2 =
                  A_buffer[b]->template get_access<access::mode::discard_write>(h);
        h.copy(kPtrA, A_matrix2);
      });

      // Compute job
      q.submit([&](handler &h) {

    //// DEBUG
    sycl::stream out(32000, 1024, h);
    //// END DEBUG

        // Create accessors to the FPGA DDR buffers
        accessor A_matrix_accessor(*A_buffer[b], h, read_only);
        accessor QR_matrix_accessor(*QR_buffer[b], h, write_only, noinit);

        // Create alias to the output matrix accessor
        auto QR_matrix_accessor_2 = QR_matrix_accessor;

        h.single_task<class QRD>([=]() [[intel::kernel_args_restrict]] {
          // Go over the matrices
          for (int l = 0; l < matrices; l++) {

            // Instantiate 3 versions of the input matrix
            // There are three loops that:
            // - load the input matrix into A_load
            // - read A_load and do the computation on A_compute, then 
            //   writes the results in A_store
            // - writes A_store into the output matrix
            [[intel::bankwidth(kBankwidth)]] // NO-FORMAT: Attribute
            [[intel::numbanks(kNumBanksNextPow2)]]   // NO-FORMAT: Attribute
            struct {
              Complex<T> d[rows];
            } A_load[columns],
              A_compute[columns], 
              A_store[columns];

            /*
              ==================================================================
              Loop 1: Copy data from DDR memory to on-chip memory.
              ==================================================================
            */
            // Get the index of the first bank of the current matrix l
            int loadBankIndex = l * kNumComplexElements / kNumElementsPerBank;

            [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
            for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; li++) {

              // Load a single bank of the input matrix 
              Complex<T> bank[kNumElementsPerBank];
              UnrolledLoop<kNumElementsPerBank>([&](auto k) {
                bank[k].xx = A_matrix_accessor[loadBankIndex * kNumElementsPerBank + k].r();
                bank[k].yy = A_matrix_accessor[loadBankIndex * kNumElementsPerBank + k].i();
              });

              // Increase the bank index
              loadBankIndex++;

              // Write the current bank to the A_load matrix.
              ac_int<BitsForMaxValue<kNumBanks>(), false> jtmp = li % (kNumBanks);
              ac_int<kLiNumBankBitSize, false> liNumBank = li / kNumBanks;
              UnrolledLoop<kNumBanks>([&](auto k) {
                UnrolledLoop<kNumElementsPerBank>([&](auto t) {
                  constexpr auto rowIdx = k * kNumElementsPerBank + t;
                  if (jtmp == k) {
                    A_load[liNumBank].d[rowIdx].xx = bank[t].xx;
                    A_load[liNumBank].d[rowIdx].yy = bank[t].yy;
                  }

                  // Delay data signals to create a vine-based data distribution
                  // to lower signal fanout.
                  bank[t].xx = sycl::ext::intel::fpga_reg(bank[t].xx);
                  bank[t].yy = sycl::ext::intel::fpga_reg(bank[t].yy);
                });

                jtmp = sycl::ext::intel::fpga_reg(jtmp);
              });
            }

            /*
              ==================================================================
              Loop 2: Compute the QR Decomposition.
              ==================================================================
              This code implements a OneAPI optimized variation of the following
              algorithm:

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

            // a local copy of a_{i+1} that is used across multiple j iterations
            // for the computation of pip1 and p
            Complex<T> a_ip1[rows];
            // a local copy of a_ip1 that is used across multiple j iterations 
            // for the computation of a_j
            Complex<T> a_i[rows];
            // Depending on the context, will contain:
            // -> -s[j]: for all the iterations to compute a_j
            // -> ir: for one iteration per j iterations to compute Q_i
            Complex<T> s_or_i[columns];


            // Get index in the output at which to start writing the outputs for
            // input matrix l.
            int qr_idx = l * kQRMatrixSize;

            // Only the real part of the complex pip1 and ir are needed for the 
            // computation
            T pip1, ir;

            // The triangular loop over i and j has been optimized using the 
            // method described in the triangular loop optimization tutorial.
            // Therefore, the code iterates over the total number of iterations,
            // including "dummy" ones (that ensures II=1).
            ac_int<kIBitSize, true> i = -1;
            ac_int<kJBitSize, true> j = 0; 
            // ac_int<kJBitSize, true> j = kVariableIterations < 0 ? 
            //                             kVariableIterations : 0;
            
            [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
            [[intel::ivdep(raw_latency)]]  // NO-FORMAT: Attribute
            for (int s = 0; s < kIterations; s++) {

              // Temporary storage for a column of the input matrix and for
              // partial results.
              Complex<T> col[rows];

              // Current value of s_or_i depending on the value of j
              // It is replicated kNumBanks times to reduce fanout
              Complex<T> sori[kNumBanks];

              // All the control signals are precomputed and replicated
              // kNumBanks times to reduce fanout
              bool  j_eq_i[kNumBanks], 
                    i_gt_0[kNumBanks],
                    i_ge_0_j_ge_i[kNumBanks], 
                    j_eq_i_plus_1[kNumBanks],
                    i_lt_0[kNumBanks];

              UnrolledLoop<kNumBanks>([&](auto k) {
                i_gt_0[k] = sycl::ext::intel::fpga_reg(i > 0);
                i_lt_0[k] = sycl::ext::intel::fpga_reg(i < 0);
                j_eq_i[k] = sycl::ext::intel::fpga_reg(j == i);
                i_ge_0_j_ge_i[k] = sycl::ext::intel::fpga_reg(i >= 0 && j >= i);
                j_eq_i_plus_1[k] = sycl::ext::intel::fpga_reg(j == i + 1);
                sori[k].xx = sycl::ext::intel::fpga_reg(s_or_i[j].xx);
                sori[k].yy = sycl::ext::intel::fpga_reg(s_or_i[j].yy);
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
                // If no i iteration elapsed, we must read the column of matrix
                // a directly from the A_load 
                // col then contains a_j
                if(i_gt_0[bank]){
                  col[k].xx = A_compute[j].d[k].xx;
                  col[k].yy = A_compute[j].d[k].yy;
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
              });

              UnrolledLoop<rows>([&](auto k) {
                // find which bank this unrolled iteration is going to use
                constexpr auto bankIdx = k / kNumElementsPerBank;

                // Depending on the iteration this code will compute either:
                // -> If i=j, a column of Q: Q_i = a_i*ir
                //    In that case, no term is added to the mult_add construct
                // -> If i!=j, an updated column of a: a_j - s[j]*a_i
                //    There is a special case if i<0 where a_j is unmodifed but
                //    the i iteration is still required to fill ir and s for
                //    subsequent iterations
                auto prod_lhs = a_i[k];
                auto prod_rhs = i_lt_0[bankIdx] ? Complex<T>(0.0, 0.0) : 
                                                                  sori[bankIdx];
                auto add = j_eq_i[bankIdx] ? Complex<T>(0.0, 0.0) : col[k];
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
                  A_store[j].d[k].xx = A_compute[j].d[k].xx = col[k].xx;
                  A_store[j].d[k].yy = A_compute[j].d[k].yy = col[k].yy;
                }

                // Store a_{i+1} for subsequent iterations of j
                if (j_eq_i_plus_1[bankIdx]) {
                  a_ip1[k] = col[k];
                }
              });

              // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
              Complex<T> p_ij = Complex<T>(0.0, 0.0);
              UnrolledLoop<rows>([&](auto k) {
                p_ij = p_ij + col[k] * a_ip1[k];
              });

              if (j == i + 1) {
                pip1 = p_ij.xx;
                ir = rsqrt(p_ij.xx);
              }

              Complex<T> s_j = Complex<T>(0.0f - (p_ij.xx) / pip1, p_ij.yy / pip1);

              // j may be negative if the number of "dummy" iterations is larger
              // than the matrix size
              if (j >= 0) {
                s_or_i[j] = Complex<T>(j == i + 1 ? ir : s_j.xx,
                                    j == i + 1 ? 0.0f : s_j.yy);
              }

              // Compute the R_{i+1,i+1} or R_{i+1,j} 
              Complex<T> r_ip1j = j == i + 1 ? Complex<T>(sycl::sqrt(pip1), 0.0) : 
                                            Complex<T>(ir * p_ij.xx, ir * p_ij.yy);

              // Write the computed R value when j is not a "dummy" iteration
              // introduced to optimized the triangular loop
              if (j >= i + 1 && i + 1 < kNValue) {
                QR_matrix_accessor[qr_idx] = {r_ip1j.xx, r_ip1j.yy};
                qr_idx++;
              }

              // Update the loop indexes.
              if (j == kNValue - 1) {
                // If i reached an index at which the j inner loop don't have
                // enough time to write its result for the next i iteration,
                // some "dummy" iterations are introduced 
                j = (kVariableIterations > i) ? 
                                  ac_int<kJBitSize, true>{i + 1} : 
                                  ac_int<kJBitSize, true>{kVariableIterations};
                i++;
              } else {
                j++;
              }
            } // end for s=0:kIterations-1

            /*
              ==================================================================
              Loop 3: Copy the result from on-chip memory to DDR memory.
              ==================================================================
            */

            [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
            for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                        si++) {
              // Only one bank is going to be stored per si iteration
              // To reduce fanout on si, a "get" table will contain for each 
              // bank a boolean to check if it should store this si iteration
              ac_int<BitsForMaxValue<kNumBanks>(), false> desired = si % (kNumBanks);
              bool get[kNumBanks];
              UnrolledLoop<kNumBanks>([&](auto k) {
                get[k] = desired == k;
                desired = sycl::ext::intel::fpga_reg(desired);
              });

              // Each bank will then check the get table to potentially 
              // read kNumElementsPerBank from A_store and store the elements
              // in bank
              Complex<T> bank[kNumElementsPerBank];
              ac_int<kSiNumBankBitSize, false> siNumBank = si / kNumBanks;
              UnrolledLoop<kNumBanks>([&](auto t) {
                UnrolledLoop<kNumElementsPerBank>([&](auto k) {
                  constexpr auto rowIdx = t * kNumElementsPerBank + k;
                  bank[k].xx = get[t] ? A_store[siNumBank].d[rowIdx].xx : 
                                                    sycl::ext::intel::fpga_reg(bank[k].xx);
                  bank[k].yy = get[t] ? A_store[siNumBank].d[rowIdx].yy : 
                                                    sycl::ext::intel::fpga_reg(bank[k].yy);
                });
              });

              // Finally, the kNumElementsPerBank elements from bank are 
              // written to the out_matrix2 (alias to out_matrix)
              UnrolledLoop<kNumElementsPerBank>([&](auto k) {
                QR_matrix_accessor_2[qr_idx + k] = {bank[k].xx, bank[k].yy};
              });

              qr_idx += kNumElementsPerBank;
            } // end for si=0:kStoreIter-1
          } // end for l=0:matrices-1
        });
      });

      // Copy the output result from the FPGA DDR to the host memory
      q.submit([&](handler &h) {
        accessor final_QR_matrix(*QR_buffer[b], h, read_only);
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


template<unsigned columns, unsigned rows, unsigned raw_latency, typename T>
void QRDecomposition( vector<ac_complex<T>> &A_matrix, 
                      vector<ac_complex<T>> &QR_matrix,
                      queue &q, 
                      size_t matrices, 
                      size_t reps) {
  QRDecomposition_impl<columns, rows, raw_latency, T>(A_matrix, QR_matrix, 
                                                            q, matrices, reps);
}