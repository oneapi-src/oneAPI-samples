//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <sycl/sycl.hpp>

// using joint_matrix = sycl::ext::oneapi::experimental::matrix;
using use = sycl::ext::oneapi::experimental::matrix::use;
using layout = sycl::ext::oneapi::experimental::matrix::layout;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#define SG_SZ 16

#define TM 8
#define TN SG_SZ
#define TK 16

#define BF16_EPSILON 0.00781250

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
private:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / 2, N * 2> &B) {
  // kernel begin
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  sycl::buffer<bfloat16, 2> bufA(A.get_data(), sycl::range<2>(M, K));
  sycl::buffer<bfloat16, 2> bufB(B.get_data(), sycl::range<2>(K, N));
  sycl::buffer<float, 2> bufC((float *)C.get_data(), sycl::range<2>(M, N));

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor accC(bufC, cgh, sycl::read_write, sycl::no_init);
     sycl::accessor accA(bufA, cgh, sycl::read_only);
     sycl::accessor accB(bufB, cgh, sycl::read_only);

     cgh.parallel_for(
         sycl::nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](sycl::nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The joint matrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sycl::sub_group sg = spmd_item.get_sub_group();
           sycl::ext::oneapi::experimental::matrix::joint_matrix<
               sycl::sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           // For B, we assume B has been already VNNIed.
           sycl::ext::oneapi::experimental::matrix::joint_matrix<
               sycl::sub_group, bfloat16, use::b, TK, TN,
               sycl::ext::intel::experimental::matrix::layout::packed>
               sub_b;
           sycl::ext::oneapi::experimental::matrix::joint_matrix<
               sycl::sub_group, float, use::accumulator, TM, TN>
               sub_c;

           joint_matrix_load(sg, sub_c,
                             accC.get_pointer() + (sg_startx * TM) * N +
                                 sg_starty / SG_SZ * TN,
                             N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(
                 sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(sg, sub_b,
                               accB.get_pointer() + (k * TK / 2) * (N * 2) +
                                   sg_starty / SG_SZ * TN * 2,
                               N * 2);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(sg, sub_c,
                              accC.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, layout::row_major);
         }); // parallel for
   }).wait();
  // kernel end
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
bfloat16 A[MATRIX_M][MATRIX_K];
bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
unsigned short Aref[MATRIX_M][MATRIX_K];
unsigned short Bref[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

float make_fp32(short x) {
  unsigned int y = x;
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

unsigned short make_bf16(float x) {
  int *res = reinterpret_cast<int *>(&x);
  *res = *res >> 16;
  return (unsigned short)*res;
}

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        short *va = (short *)(A_mem + m * K + k);
        short *vb = (short *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
        for (int i = 0; i < 2; i++) {
          acc += (make_fp32(va[i]) * make_fp32(vb[i]));
        }
        *((float *)(C_mem + m * N + n)) = acc;
      }
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      // bfloat16 is created using unsigned short since conversion from float to
      // bfloat16 is not supported on the host side yet
      A[i][j] = bfloat16(1.0f * (i + j));
      Aref[i][j] = make_bf16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = bfloat16(2.0f * i + 3.0f * j);
      Bref[i][j] = make_bf16(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1.0;
      D[i][j] = 1.0;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB((bfloat16 *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((int32_t *)Aref, (int32_t *)Bref, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 2);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if ((fabs(C[i][j]) - fabs(D[i][j])) > BF16_EPSILON)
        res = false;
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
