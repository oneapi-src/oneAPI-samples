//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "CL/sycl.hpp"
#include "mkl.h"
#include "oneapi/mkl/blas.hpp"
#include <cfloat>
#include <iostream>

int main(int argc, char **argv) {
  // Enter matrix dimensions
  unsigned int M, K, N;

  if (argc < 3) {
    std::cout << "Usage: matmul_onemkl N K" << std::endl;
    return -1;
  }
  M = N = std::stoi(argv[1]);
  K = std::stoi(argv[2]);

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector_v);
  auto sycl_device = Q.get_device();
  auto sycl_context = Q.get_context();
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Declare and allocate matrices in USM
  auto A = sycl::malloc_shared<float>(M * K, sycl_device, sycl_context);
  auto B = sycl::malloc_shared<float>(K * N, sycl_device, sycl_context);
  auto C = sycl::malloc_shared<float>(M * N, sycl_device, sycl_context);

  // Initialize matrices A and B
  for (unsigned int i = 0; i < M * K; i++)
    A[i] = 1.0;
  for (unsigned int i = 0; i < K * N; i++)
    B[i] = 1.0;

  auto start_time = std::chrono::system_clock::now(); // Start timer

  // Offload matrix multiplication
  float alpha = 1.0, beta = 0.0;
  oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;
  sycl::event gemm_done;
  std::vector<sycl::event> gemm_dependencies;
  gemm_done = oneapi::mkl::blas::gemm(Q, transA, transB, M, N, K, alpha, A, M,
                                      B, K, beta, C, M, gemm_dependencies);
  gemm_done.wait();

  auto end_time = std::chrono::system_clock::now(); // Stop timer
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  // Check for correctness and report run time
  bool errors(false);
  for (unsigned int i = 0; i < M * N; i++)
    if (std::abs(C[i] - K) > FLT_MIN) {
      errors = true;
      break;
    }
  if (errors)
    std::cout << "Program completed with errors." << std::endl;
  else {
    std::cout << "Program completed without errors." << std::endl;
    std::cout << "oneMKL SGEMM of " << M << " x " << K << " and " << K << " x "
              << N << " matrices took " << elapsed_time.count() << " seconds."
              << std::endl;
  }

  // Cleanup
  sycl::free(A, Q);
  sycl::free(B, Q);
  sycl::free(C, Q);
}
