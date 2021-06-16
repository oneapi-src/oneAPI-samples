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
struct MyComplex {
  float xx;
  float yy;

  MyComplex(float x, float y) {
    xx = x;
    yy = y;
  }

  MyComplex() {}

  // Complex addition
  const MyComplex operator+(const MyComplex rhs) const {
    return MyComplex(xx + rhs.xx, yy + rhs.yy);
  }

  // Complex multiplication
  const MyComplex operator*(const MyComplex rhs) const {
    MyComplex c;
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

  This function implements a OneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan
  Pasca.

*/
void QRDecomposition(vector<float> &in_matrix, vector<float> &out_matrix,
                     queue &q, size_t matrices, size_t reps) {

  // Number of complex elements in the matrix
  constexpr int kNumComplexElements = COLS_COMPONENT * ROWS_COMPONENT;

  // Bits to encode the COLS_COMPONENT and ROWS_COMPONENT
  constexpr int kColsComponentBitSize = CeilLog2(COLS_COMPONENT);
  constexpr int kRowsComponentBitSize = CeilLog2(ROWS_COMPONENT);

  // Sizes of allocated memories for input and output matrix
  constexpr int kInputMatrixSize = kNumComplexElements * 2;
  constexpr int kOutputMatrixSize = (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3;

  // Constants related to the memory configuration of the kernel's local
  // memories
  // We want 4 complex elements (2 floating point values) in each memory bank
  constexpr int kNumElementsPerBank = 4;
  // Set the bankwidth in bytes
  constexpr int kBankwidth = kNumElementsPerBank * 8;
  constexpr int kNumBanks = ROWS_COMPONENT / kNumElementsPerBank;

  constexpr int kLoadIter = kNumComplexElements / kNumElementsPerBank;
  constexpr int kLoadIterBitSize = CeilLog2(kLoadIter);
  constexpr int kStoreIter = kNumComplexElements / kNumElementsPerBank;
  constexpr int kStoreIterBitSize = CeilLog2(kStoreIter);
  constexpr short kNumBuffers = 4;

  // Number of iterations performed without any dummy work added
  // for the triangular loop optimization
  constexpr int kVariableIterations = N_VALUE - FIXED_ITERATIONS;

  constexpr int kjBitSize = (kVariableIterations < 0 ? 
                            CeilLog2(COLS_COMPONENT + 1) + 
                            CeilLog2(-kVariableIterations) : 
                            CeilLog2(COLS_COMPONENT + 1) + 1) 
                            + 1;

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

  for (size_t r = 0; r < reps; r++) {
    for (size_t b = 0, it = 0; it < matrices;
         it += chunk, b = (b + 1) % kNumBuffers) {
      const float *kPtr = in_matrix.data() + kInputMatrixSize * it;
      float *kPtr2 = out_matrix.data() + kOutputMatrixSize * it;
      int matrices = chunk;

      q.submit([&](handler &h) {
        auto in_matrix2 =
            input_matrix[b]->get_access<access::mode::discard_write>(h);
        h.copy(kPtr, in_matrix2);
      });

      q.submit([&](handler &h) {
    //// DEBUG
    sycl::stream out(16384, 1024, h);
    //// END DEBUG


        accessor in_matrix(*input_matrix[b], h, read_only);
        accessor out_matrix(*output_matrix[b], h, write_only, no_init);
        auto out_matrix2 = out_matrix;
        h.single_task<class QRD>([=]() [[intel::kernel_args_restrict]] {
          for (int l = 0; l < matrices; l++) {
            [[intel::bankwidth(kBankwidth),
              intel::numbanks(kNumBanks)]] struct {
              MyComplex d[ROWS_COMPONENT];
            } a_matrix[COLS_COMPONENT], ap_matrix[COLS_COMPONENT],
                aload_matrix[COLS_COMPONENT];

            MyComplex vector_ai[ROWS_COMPONENT], vector_ti[ROWS_COMPONENT];
            MyComplex s_or_i[COLS_COMPONENT];

            // Copy data from DDR memory to on-chip memory.
            int idx = l * kNumComplexElements / kNumElementsPerBank;
            for (ac_int<kLoadIterBitSize+1, false> li = 0; li < kLoadIter; li++) {
              MyComplex tmp[kNumElementsPerBank];
              Unroller<0, kNumElementsPerBank>::Step([&](int k) {
                tmp[k].xx = in_matrix[idx * 2 * kNumElementsPerBank + k * 2];
                tmp[k].yy =
                    in_matrix[idx * 2 * kNumElementsPerBank + k * 2 + 1];
              });

              idx++;
              int jtmp = li % (kNumBanks);

              Unroller<0, kNumBanks>::Step([&](int k) {
                Unroller<0, kNumElementsPerBank>::Step([&](int t) {
                  if (jtmp == k) {
                    aload_matrix[li / (kNumBanks)]
                        .d[k * kNumElementsPerBank + t]
                        .xx = tmp[t].xx;
                    aload_matrix[li / (kNumBanks)]
                        .d[k * kNumElementsPerBank + t]
                        .yy = tmp[t].yy;
                  }

                  // Delay data signals to create a vine-based data distribution
                  // to lower signal fanout.
                  tmp[t].xx = ext::intel::fpga_reg(tmp[t].xx);
                  tmp[t].yy = ext::intel::fpga_reg(tmp[t].yy);
                });

                jtmp = ext::intel::fpga_reg(jtmp);
              });
            }

            float p_ii_x, i_r_ii_x;
            int qr_idx = l * kOutputMatrixSize / 2;

            ac_int<kRowsComponentBitSize+2, true> i = -1;
            ac_int<kjBitSize, true> j = kVariableIterations < 0
                          ? kVariableIterations
                          : 0;
            [[intel::initiation_interval(1)]] [[intel::ivdep(FIXED_ITERATIONS)]]
            for (int s = 0; s < ITERATIONS; s++) {
              MyComplex vector_t[ROWS_COMPONENT];
              MyComplex sori[kNumBanks];

              bool j_eq_i[kNumBanks], i_gt_0[kNumBanks],
                  i_ge_0_j_eq_i[kNumBanks], j_eq_i_plus_1[kNumBanks],
                  i_lt_0[kNumBanks];

              Unroller<0, kNumBanks>::Step([&](int k) {
                i_gt_0[k] = ext::intel::fpga_reg(i > 0);
                i_lt_0[k] = ext::intel::fpga_reg(i < 0);
                j_eq_i[k] = ext::intel::fpga_reg(j == i);
                i_ge_0_j_eq_i[k] = ext::intel::fpga_reg(i >= 0 && j >= i);
                j_eq_i_plus_1[k] = ext::intel::fpga_reg(j == i + 1);
                sori[k].xx = ext::intel::fpga_reg(s_or_i[j].xx);
                sori[k].yy = ext::intel::fpga_reg(s_or_i[j].yy);
              });

              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                vector_t[k].xx = aload_matrix[j].d[k].xx;
                vector_t[k].yy = aload_matrix[j].d[k].yy;
                if (i_gt_0[k / kNumElementsPerBank]) {
                  vector_t[k].xx = a_matrix[j].d[k].xx;
                  vector_t[k].yy = a_matrix[j].d[k].yy;
                }
                if (j_eq_i[k / kNumElementsPerBank]) {
                  vector_ai[k].xx = vector_t[k].xx;
                  vector_ai[k].yy = vector_t[k].yy;
                }
              });

              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                auto prod_lhs = vector_ai[k];
                auto prod_rhs = i_lt_0[k / kNumElementsPerBank] ?
                                  MyComplex(0.0, 0.0) :
                                  sori[k / kNumElementsPerBank];
                auto prod_added_term = j_eq_i[k / kNumElementsPerBank] ?
                                        MyComplex(0.0, 0.0) :
                                        vector_t[k];
                vector_t[k] = prod_lhs * prod_rhs + prod_added_term;

                if (i_ge_0_j_eq_i[k / kNumElementsPerBank]) {
                  ap_matrix[j].d[k].xx = a_matrix[j].d[k].xx = vector_t[k].xx;
                  ap_matrix[j].d[k].yy = a_matrix[j].d[k].yy = vector_t[k].yy;
                }
                if (j_eq_i_plus_1[k / kNumElementsPerBank]) {
                  vector_ti[k] = vector_t[k];
                }
              });

              MyComplex p_ij = MyComplex(0, 0);
              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                p_ij = p_ij + vector_t[k] * vector_ti[k];
              });

              if (j == i + 1) {
                p_ii_x = p_ij.xx;
                i_r_ii_x = rsqrt(p_ij.xx);
              }

              MyComplex s_ij =
                  MyComplex(0.0f - (p_ij.xx) / p_ii_x, p_ij.yy / p_ii_x);

              if (j >= 0) {
                s_or_i[j] = MyComplex(j == i + 1 ? i_r_ii_x : s_ij.xx,
                                      j == i + 1 ? 0.0f : s_ij.yy);
              }

              MyComplex r_ii = j == i + 1 ? MyComplex(sycl::sqrt(p_ii_x), 0.0)
                                          : MyComplex(i_r_ii_x * p_ij.xx,
                                                      i_r_ii_x * p_ij.yy);

              if (j >= i + 1 && i + 1 < N_VALUE) {
                out_matrix[qr_idx * 2] = r_ii.xx;
                out_matrix[qr_idx * 2 + 1] = r_ii.yy;
                qr_idx++;
              }

              if (j == N_VALUE - 1) {
                j = (kVariableIterations > i)
                        ? ac_int<kjBitSize, true>{i + 1}
                        : ac_int<kjBitSize, true>{kVariableIterations};
                i++;
              } else {
                j++;
              }
            }

            qr_idx *= 2;

            for (ac_int<kLoadIterBitSize+1, false> si = 0; si < kStoreIter; si++) {
              int desired = si % (kNumBanks);
              bool get[kNumBanks];
              Unroller<0, kNumBanks>::Step([&](int k) {
                get[k] = desired == k;
                desired = ext::intel::fpga_reg(desired);
              });

              MyComplex tmp[kNumElementsPerBank];
              Unroller<0, kNumBanks>::Step([&](int t) {
                Unroller<0, kNumElementsPerBank>::Step([&](int k) {
                  tmp[k].xx = get[t] ? ap_matrix[si / (kNumBanks)]
                                           .d[t * kNumElementsPerBank + k]
                                           .xx
                                     : ext::intel::fpga_reg(tmp[k].xx);
                  tmp[k].yy = get[t] ? ap_matrix[si / (kNumBanks)]
                                           .d[t * kNumElementsPerBank + k]
                                           .yy
                                     : ext::intel::fpga_reg(tmp[k].yy);
                });
              });

              Unroller<0, kNumElementsPerBank>::Step([&](int k) {
                out_matrix2[qr_idx + k * 2]     = tmp[k].xx;
                out_matrix2[qr_idx + k * 2 + 1] = tmp[k].yy;
              });

              qr_idx += kNumElementsPerBank*2;
            }
          }
        });
      });

      q.submit([&](handler &h) {
        accessor final_matrix(*output_matrix[b], h, read_only);
        h.copy(final_matrix, kPtr2);
      });
    }
  }

  for (short b = 0; b < kNumBuffers; b++) {
    delete input_matrix[b];
    delete output_matrix[b];
  }
}
