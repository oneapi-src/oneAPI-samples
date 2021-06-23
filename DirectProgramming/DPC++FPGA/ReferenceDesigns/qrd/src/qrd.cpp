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
#include <chrono>
#include <cstring>
#include <vector>

#include "qrd.hpp"

using std::vector;
using namespace sycl;

/*
  The Unroller functions are utility functions used to do static loop unrolling
*/
template <int begin, int end> struct Unroller {
  template <typename Action> static void Step(const Action &action) {
    action(begin);
    Unroller<begin + 1, end>::Step(action);
  }
};

template <int end> struct Unroller<end, end> {
  template <typename Action> static void Step(const Action &action) {}
};

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
template <typename T>
static constexpr T CeilLog2(T n) {
  return ((n == 1) ? T(0) : Log2(n - 1) + T(1));
}

/*
  A structure that represents a complex number.
  - xx represents its real value
  - yy represents its imaginary value
  Two operators are overloaded (+, *) to help with manipulating this structure.
*/
struct Complex {
  float xx;
  float yy;

  Complex(float x, float y) {
    xx = x;
    yy = y;
  }

  Complex() {}

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
  e.g. in_matrix[2*i + 0] and in_matrix[2*i + 1] represent the real and
  imaginary values of the i-th complex element of the matrix.

  Function arguments:
  - in_matrix:  The input matrix. Interpreted as a transposed matrix.
  - out_matrix: The output matrix. The function will overwrite this matrix.
                The first values of this output vector will contain the upper
                triangular values of the R matrix, row by row.
                e.g. for a 4x4 QRD, out_matrix[5*2+1] will contain the imaginary
                part of R[1][1].
                So there are exactly N*(N+1)/2 elements of R.
                So rest of the values hold the transposed matrix Q.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read from the in_matrix vector.
  - reps:       The number of repetitions of the computation to execute.
                (for performance evaluation)

  This function requires the definition of the following global defines:
  - COLS_COMPONENT:   The number of columns in the matrix
  - ROWS_COMPONENT:   The number of rows in the matrix     
  - FIXED_ITERATIONS: The latency between the RAW dependency in the triangular
                      loop that prevents the compiler to achieve an II of 1.
                      This helps create a loop structure that can reach an II
                      of 1 following the triangular loop optimization tutorial
                      method.
  
*/
void QRDecomposition(vector<float> &in_matrix, vector<float> &out_matrix,
                     queue &q, size_t matrices, size_t reps) {

  // Number of complex elements in the matrix
  constexpr int kNumComplexElements = COLS_COMPONENT * ROWS_COMPONENT;

  // Sizes of allocated memories for input and output matrix
  // Both the input matrix and Q are full matrices of complex elements
  // R only contains COLS_COMPONENT + 
  //                 (COLS_COMPONENT - 1) + 
  //                 (COLS_COMPONENT - 2) +
  //                 (COLS_COMPONENT - 3) + 
  //                 etc.
  // So R contains COLS_COMPONENT * (COLS_COMPONENT + 1) / 2 complex elements.
  // Each complex element takes two indexes.
  constexpr int kInputMatrixSize = kNumComplexElements * 2;
  constexpr int kQMatrixSize = kNumComplexElements * 2;
  constexpr int kRMatrixSize = COLS_COMPONENT * (COLS_COMPONENT + 1);
  constexpr int kOutputMatrixSize = kQMatrixSize + kRMatrixSize;

  // Constants related to the memory configuration of the kernel's local
  // memories
  // We want 4 complex elements (2 floating point values) in each memory bank
  constexpr int kNumElementsPerBank = 4;
  // Set the bankwidth in bytes
  constexpr int kBankwidth = kNumElementsPerBank * 8;
  constexpr int kNumBanks = ROWS_COMPONENT / kNumElementsPerBank;

  // Number of load and store iterations for a single matrix given the size
  // of the input matrices and the number of complex elements per banks
  constexpr int kLoadIter = kNumComplexElements / kNumElementsPerBank;
  constexpr int kStoreIter = kNumComplexElements / kNumElementsPerBank;
  // Number of bits required by the loop counters for the loads/stores iterators
  constexpr int kLoadIterBitSize = CeilLog2(kLoadIter + 1);
  constexpr int kStoreIterBitSize = CeilLog2(kStoreIter + 1);

  // Number of buffers to allocate to be able to read/compute/store 
  // without overlap. 
  // TODO: We technically need 3, but having 4 maybe improves FMax?
  //       Having 3 seems to improve latency without compromising FMax.
  constexpr short kNumBuffers = 3;
    
  // Number of iterations performed without any dummy work added for the 
  // triangular loop optimization
  constexpr int kVariableIterations = N_VALUE - FIXED_ITERATIONS;

  // Sizes in bits for the triangular loop indexes
  // i starts from -1 and goes up to ROWS_COMPONENT
  // So we need:
  // -> enough bits to encode ROWS_COMPONENT+1 for the positive iterations and 
  //    the exit condition
  // -> one extra bit for the -1
  constexpr int kIBitSize = CeilLog2(ROWS_COMPONENT + 1) + 1;
  // j starts from i, so from -1 and goes up to COLS_COMPONENT
  // So we need:
  // -> enough bits to encode COLS_COMPONENT+1 for the positive iterations and 
  //    the exit condition
  // -> one extra bit for the -1
  // But j may start below -1 if we perform more dummy iterations than the 
  // number of columns in the matrix.
  // In that case, we need:
  // -> enough bits to encode COLS_COMPONENT+1 for the positive iterations and 
  //    the exit condition
  // -> enough bits to encode the maximum number of negative iterations
  constexpr int kJNegativeIterations = 
                            kVariableIterations < 0 ? -kVariableIterations : 1;
  constexpr int kJBitSize = CeilLog2(COLS_COMPONENT + 1) 
                            + CeilLog2(kJNegativeIterations);

  // We will process 'chunk' number of matrices in each run of the kernel
  short chunk = 2048;
  if (matrices % chunk) {
    chunk = 1;
  }

  // Create buffers and allocate space for them.
  buffer<float, 1> *input_matrix[kNumBuffers], *output_matrix[kNumBuffers];
  for (short i = 0; i < kNumBuffers; i++) {
    input_matrix[i] = new buffer<float, 1>(kInputMatrixSize * chunk);
    output_matrix[i] = new buffer<float, 1>(kOutputMatrixSize * chunk);
  }

  // Repeat the computation multiple times (for performance analysis)
  for (size_t r = 0; r < reps; r++) {

    // Go over all the matrices, rotating buffers every time
    for (size_t b = 0, it = 0; it < matrices; 
                                      it += chunk, b = (b + 1) % kNumBuffers) {

      // Pointer to current input/output matrices in host memory 
      const float *kPtr = in_matrix.data() + kInputMatrixSize * it;
      float *kPtr2 = out_matrix.data() + kOutputMatrixSize * it;

      int matrices = chunk;

      // Copy a new input matrix from the host memory into the FPGA DDR 
      q.submit([&](handler &h) {
        auto in_matrix2 =
            input_matrix[b]->get_access<access::mode::discard_write>(h);
        h.copy(kPtr, in_matrix2);
      });

      // Compute job
      q.submit([&](handler &h) {

        // Create accessors to the FPGA DDR buffers
        accessor in_matrix(*input_matrix[b], h, read_only);

        accessor out_matrix(*output_matrix[b], h, write_only, no_init);

        auto out_matrix2 = out_matrix;

    //// DEBUG
    sycl::stream out(16384, 1024, h);
    //// END DEBUG

        h.single_task<class QRD>([=]() [[intel::kernel_args_restrict]] {
          // Go over the matrices
          for (int l = 0; l < matrices; l++) {

            // Instantiate 3 versions of the input matrix
            // There are three loops that:
            // - load the input matrix into aload_matrix
            // - read aload_matrix and do the computation on a_matrix, then 
            //   writes the results in astore_matrix
            // - writes astore_matrix into the output matrix
            [[intel::bankwidth(kBankwidth)]] // NO-FORMAT: Attribute
            [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
            struct {
              Complex d[ROWS_COMPONENT];
            } aload_matrix[COLS_COMPONENT],
              a_matrix[COLS_COMPONENT], 
              astore_matrix[COLS_COMPONENT];

            /*
              ==================================================================
              Loop 1: Copy data from DDR memory to on-chip memory.
              ==================================================================
            */
            // Get the index of the first bank of the current matrix l
            int loadBankIndex = l * kNumComplexElements / kNumElementsPerBank;

            for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; li++) {

              // Load a single bank of the input matrix 
              Complex tmp[kNumElementsPerBank];
              Unroller<0, kNumElementsPerBank>::Step([&](int k) {
                tmp[k].xx = in_matrix[loadBankIndex * 2 * kNumElementsPerBank + 
                                      k * 2];
                tmp[k].yy = in_matrix[loadBankIndex * 2 * kNumElementsPerBank + 
                                      k * 2 + 1];
              });

              // Increase the bank index
              loadBankIndex++;

              // Write the current bank to the aload_matrix matrix.
              int jtmp = li % (kNumBanks);
              int liNumBank = li / kNumBanks;
              Unroller<0, kNumBanks>::Step([&](int k) {
                Unroller<0, kNumElementsPerBank>::Step([&](int t) {
                  if (jtmp == k) {
                    aload_matrix[liNumBank].d[k * kNumElementsPerBank + t].xx 
                                                                    = tmp[t].xx;
                    aload_matrix[liNumBank].d[k * kNumElementsPerBank + t].yy 
                                                                    = tmp[t].yy;
                  }

                  // Delay data signals to create a vine-based data distribution
                  // to lower signal fanout.
                  tmp[t].xx = ext::intel::fpga_reg(tmp[t].xx);
                  tmp[t].yy = ext::intel::fpga_reg(tmp[t].yy);
                });

                jtmp = ext::intel::fpga_reg(jtmp);
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
            Complex a_ip1[ROWS_COMPONENT];
            // a local copy of a_ip1 that is used across multiple j iterations 
            // for the computation of a_j
            Complex a_i[ROWS_COMPONENT];
            // Depending on the context, will contain:
            // -> -s[j]: for all the iterations to compute a_j
            // -> ir: for one iteration per j iterations to compute Q_i
            Complex s_or_i[COLS_COMPONENT];


            // Get index in the output at which to start writing the outputs for
            // input matrix l.
            int qr_idx = l * kOutputMatrixSize;

            // Only the real part of the complex pip1 and ir are needed for the 
            // computation
            float pip1, ir;

            // The triangular loop over i and j has been optimized using the 
            // method described in the triangular loop optimization tutorial.
            // Therefore, the code iterates over the total number of iterations,
            // including "dummy" ones (that ensures II=1).
            ac_int<kIBitSize, true> i = -1;
            ac_int<kJBitSize, true> j = kVariableIterations < 0 ? 
                                        kVariableIterations : 0;

            [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
            [[intel::ivdep(FIXED_ITERATIONS)]]  // NO-FORMAT: Attribute
            for (int s = 0; s < ITERATIONS; s++) {

              // Temporary storage for a column of the input matrix and for
              // partial results.
              Complex col[ROWS_COMPONENT];

              // Current value of s_or_i depending on the value of j
              // It is replicated kNumBanks times to reduce fanout
              Complex sori[kNumBanks];

              // All the control signals are precomputed and replicated
              // kNumBanks times to reduce fanout
              bool  j_eq_i[kNumBanks], 
                    i_gt_0[kNumBanks],
                    i_ge_0_j_ge_i[kNumBanks], 
                    j_eq_i_plus_1[kNumBanks],
                    i_lt_0[kNumBanks];

              Unroller<0, kNumBanks>::Step([&](int k) {
                i_gt_0[k] = ext::intel::fpga_reg(i > 0);
                i_lt_0[k] = ext::intel::fpga_reg(i < 0);
                j_eq_i[k] = ext::intel::fpga_reg(j == i);
                i_ge_0_j_ge_i[k] = ext::intel::fpga_reg(i >= 0 && j >= i);
                j_eq_i_plus_1[k] = ext::intel::fpga_reg(j == i + 1);
                sori[k].xx = ext::intel::fpga_reg(s_or_i[j].xx);
                sori[k].yy = ext::intel::fpga_reg(s_or_i[j].yy);
              });

              // Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
              //   col[k].xx = aload_matrix[j].d[k].xx;
              //   col[k].yy = aload_matrix[j].d[k].yy;
              //   if (i_gt_0[k / kNumElementsPerBank]) {
              //     col[k].xx = a_matrix[j].d[k].xx;
              //     col[k].yy = a_matrix[j].d[k].yy;
              //   }
              //   if (j_eq_i[k / kNumElementsPerBank]) {
              //     a_i[k].xx = col[k].xx;
              //     a_i[k].yy = col[k].yy;
              //   }
              // });
              // Preload col and a_i with the correct data for the current 
              // iteration.
              // These are going to be use to compute the dot product of 
              // two different column of the input matrix.
              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                // find which bank this unrolled iteration is going to use
                auto bank = k / kNumElementsPerBank;

                // Load col with the current column of matrix a.
                // At least one iteration of the outer loop i is required
                // for the "working copy" a_matrix to contain data.
                // If no i iteration elapsed, we must read the column of matrix
                // a directly from the aload_matrix 
                // col then contains a_j
                col[k].xx = i_gt_0[bank] ? a_matrix[j].d[k].xx : 
                                                        aload_matrix[j].d[k].xx;
                col[k].yy = i_gt_0[bank] ? a_matrix[j].d[k].yy : 
                                                        aload_matrix[j].d[k].yy;

                // Load a_i for reuse across j iterations
                if (j_eq_i[bank]) {
                  a_i[k].xx = col[k].xx;
                  a_i[k].yy = col[k].yy;
                }
              });


              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                // find which bank this unrolled iteration is going to use
                auto bank = k / kNumElementsPerBank;

                // Depending on the iteration this code will compute either:
                // -> If i=j, a column of Q: Q_i = a_i*ir
                //    In that case, no term is added to the mult_add construct
                // -> If i!=j, an updated column of a: a_j - s[j]*a_i
                //    There is a special case if i<0 where a_j is unmodifed but
                //    the i iteration is still required to fill ir and s for
                //    subsequent iterations
                auto prod_lhs = a_i[k];
                auto prod_rhs = i_lt_0[bank] ? Complex(0.0, 0.0) : sori[bank];
                auto add = j_eq_i[bank] ? Complex(0.0, 0.0) : col[k];
                col[k] = prod_lhs * prod_rhs + add;

                // Store Q_i in astore_matrix and the modified a_j in a_matrix
                // To reduce the amount of control, astore_matrix and a_matrix
                // are both written to for each iteration of i>=0 && j>=i
                // In fact:
                // -> astore_matrix could only be written to at iterations i==j
                // -> a_matrix could only be written to at iterations 
                //    j!=i && i>=0  
                // The extra writes are harmless as the locations written to 
                // are either going to be:
                // -> overwritten for the matrix Q (astore_matrix)
                // -> unused for the a_matrix
                if (i_ge_0_j_ge_i[bank]) {
                  astore_matrix[j].d[k].xx = a_matrix[j].d[k].xx = col[k].xx;
                  astore_matrix[j].d[k].yy = a_matrix[j].d[k].yy = col[k].yy;
                }

                // Store a_{i+1} for subsequent iterations of j
                if (j_eq_i_plus_1[bank]) {
                  a_ip1[k] = col[k];
                }
              });

              // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
              Complex p_ij = Complex(0, 0);
              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                p_ij = p_ij + col[k] * a_ip1[k];
              });

              if (j == i + 1) {
                pip1 = p_ij.xx;
                ir = rsqrt(p_ij.xx);
              }

              Complex s_j = Complex(0.0f - (p_ij.xx) / pip1, p_ij.yy / pip1);

              // j may be negative if the number of "dummy" iterations is larger
              // than the matrix size
              if (j >= 0) {
                s_or_i[j] = Complex(j == i + 1 ? ir : s_j.xx,
                                    j == i + 1 ? 0.0f : s_j.yy);
              }

              // Compute the R_{i+1,i+1} or R_{i+1,j} 
              Complex r_ip1j = j == i + 1 ? Complex(sycl::sqrt(pip1), 0.0) : 
                                            Complex(ir * p_ij.xx, ir * p_ij.yy);

              // Write the computed R value when j is not a "dummy" iteration
              // introduced to optimized the triangular loop
              if (j >= i + 1 && i + 1 < N_VALUE) {
                out_matrix[qr_idx] = r_ip1j.xx;
                out_matrix[qr_idx + 1] = r_ip1j.yy;
                qr_idx+=2;
              }

              // Update the loop indexes.
              if (j == N_VALUE - 1) {
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
            } // end for s=0:ITERATIONS-1

            /*
              ==================================================================
              Loop 3: Copy the result from on-chip memory to DDR memory.
              ==================================================================
            */
            for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; si++) {
              int desired = si % (kNumBanks);
              bool get[kNumBanks];
              Unroller<0, kNumBanks>::Step([&](int k) {
                get[k] = desired == k;
                desired = ext::intel::fpga_reg(desired);
              });

              Complex tmp[kNumElementsPerBank];
              int siNumBank = si / kNumBanks;
              Unroller<0, kNumBanks>::Step([&](int t) {
                Unroller<0, kNumElementsPerBank>::Step([&](int k) {
                  tmp[k].xx = get[t] ? 
                        astore_matrix[siNumBank].d[t * kNumElementsPerBank + k].xx : 
                        INTEL::fpga_reg(tmp[k].xx);
                  tmp[k].yy = get[t] ? 
                        astore_matrix[siNumBank].d[t * kNumElementsPerBank + k].yy : 
                        INTEL::fpga_reg(tmp[k].yy);
                });
              });

              Unroller<0, kNumElementsPerBank>::Step([&](int k) {
                out_matrix2[qr_idx + k * 2]     = tmp[k].xx;
                out_matrix2[qr_idx + k * 2 + 1] = tmp[k].yy;
              });

              qr_idx += kNumElementsPerBank*2;
            } // end for si=0:kStoreIter-1
          } // end for l=0:matrices-1
        });
      });

      // Copy the output result from the FPGA DDR to the host memory
      q.submit([&](handler &h) {
        accessor final_matrix(*output_matrix[b], h, read_only);
        h.copy(final_matrix, kPtr2);
      });

    } // end for it=0:matrices-1 
  } // end for r=0:reps-1 

  // Clean allocated buffers
  for (short b = 0; b < kNumBuffers; b++) {
    delete input_matrix[b];
    delete output_matrix[b];
  }
}
