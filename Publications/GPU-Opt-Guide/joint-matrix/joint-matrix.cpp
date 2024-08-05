//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <sycl/sycl.hpp>

using use = sycl::ext::oneapi::experimental::matrix::use;
using layout = sycl::ext::oneapi::experimental::matrix::layout;
using bfloat16 = sycl::ext::oneapi::bfloat16;

constexpr float ALPHA = 2.0;
constexpr float C_INIT = 1.0;
constexpr float BF16_EPSILON = 0.00781250;

template <typename KernelName> size_t get_sg_size(sycl::queue q) {
  auto KernelID = sycl::get_kernel_id<KernelName>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q.get_context(), {KernelID});
  auto kernel = KB.get_kernel(KernelID);

  return kernel.template get_info<
      sycl::info::kernel_device_specific::max_sub_group_size>(q.get_device());
}
template <typename Tc, typename Ta, typename Tb, size_t TM, size_t TN,
          size_t TK, size_t M, size_t N, size_t K, class kernel_name>
void matrix_multiply(Tc *C, Ta *A, Tb *B, sycl::queue q) {
  // kernel begin
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<kernel_name>(
         sycl::nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](sycl::nd_item<2> spmd_item)

         {
           // The joint matrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sycl::sub_group sg = spmd_item.get_sub_group();
           auto pA = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::no>(A);
           auto pB = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::no>(B);
           auto pC = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::no>(C);
           sycl::ext::oneapi::experimental::matrix::joint_matrix<
               sycl::sub_group, Ta, use::a, TM, TK, layout::row_major>
               sub_a;
           sycl::ext::oneapi::experimental::matrix::joint_matrix<
               sycl::sub_group, Tb, use::b, TK, TN, layout::row_major>
               sub_b;
           sycl::ext::oneapi::experimental::matrix::joint_matrix<
               sycl::sub_group, Tc, use::accumulator, TM, TN>
               sub_c;

           joint_matrix_fill(sg, sub_c, C_INIT);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k * TK,
                               K);
             joint_matrix_load(sg, sub_b,
                               pB + (k * TK) * N + sg_starty / sg_size * TN, N);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_apply(sg, sub_c, [=](Tc &x) { x *= ALPHA; });
           joint_matrix_store(
               sg, sub_c, pC + (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
  // kernel end
}

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

template <typename Tc, typename Ta, typename Tb, size_t M, size_t N, size_t K>
void matrix_multiply_ref(Ta *A, Tb *B, Tc *C) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += make_fp32(A[m * K + k]) * make_fp32(B[k * N + n]);
      }
      C[m * N + n] *= ALPHA;
    }
}

template <typename Ta, typename Tb, typename Tc, size_t TM, size_t TN,
          size_t TK, class kernel_name>
int test() {

  static constexpr size_t M = TM * 2;
  static constexpr size_t N = TN * 2;
  static constexpr size_t K = TK * 2;
  sycl::queue q;
  Ta *A = sycl::malloc_shared<Ta>(M * K, q);
  Tb *B = sycl::malloc_shared<Tb>(K * N, q);
  Tc *C = sycl::malloc_shared<Tc>(M * N, q);
  Tc *D = sycl::malloc_shared<Tc>(M * N, q);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = Ta(1.0f * (i + j));
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = Tb(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = C_INIT;
      D[i * N + j] = C_INIT;
    }
  }

  matrix_multiply_ref<Tc, Ta, Tb, M, N, K>(A, B, D);
  matrix_multiply<Tc, Ta, Tb, TM, TN, TK, M, N, K, kernel_name>(C, A, B, q);
  bool res = true;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if constexpr (std::is_same_v<Tc, float>) {
        if (fabs(C[i * N + j] - D[i * N + j]) > BF16_EPSILON) {
          res = false;
          std::cout << "Incorrect result in matrix. "
                    << "i: " << i << ", j: " << j << ", Ref: " << D[i * N + j]
                    << ", Val: " << C[i * N + j] << "\n";
        }
      } else if (C[i * N + j] != D[i * N + j])
        res = false;
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}

int main() {
  sycl::queue q;
  // Snippet begin
  std::vector<sycl::ext::oneapi::experimental::matrix::combination>
      combinations = q.get_device()
                         .get_info<sycl::ext::oneapi::experimental::info::
                                       device::matrix_combinations>();

  bool passed = true;
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      passed &=
          test<int8_t, int8_t, int32_t, 16, 16, 64, class amx_int_16x16x64>();
      passed &= test<bfloat16, bfloat16, float, 16, 16, 32,
                     class amx_bf16_16x16x32>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      passed &=
          test<int8_t, int8_t, int32_t, 8, 16, 32, class pvc_int_8x16x32>();
      passed &=
          test<bfloat16, bfloat16, float, 8, 16, 16, class pvc_bf16_8x16x16>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      passed &= test<int8_t, int8_t, int32_t, 8, 8, 32, class dg2_int_8x8x32>();
      passed &=
          test<bfloat16, bfloat16, float, 8, 8, 16, class dg2_bf16_8x16x16>();
      break;
    }
  }
  // Snippet end

  return !passed;
}
