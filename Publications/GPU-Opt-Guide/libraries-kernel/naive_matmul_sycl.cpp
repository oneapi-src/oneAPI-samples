//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "CL/sycl.hpp"
#include <cfloat>
#include <iostream>

int main(int argc, char **argv) {
  // Enter matrix dimensions
  unsigned int m, k, n, M, K, N;
  if (argc < 3) {
    std::cout << "Usage: naive_matmul_sycl M K" << std::endl;
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

  // Allocate matrices A, B, and C in USM
  auto A = sycl::malloc_shared<float *>(M, sycl_device, sycl_context);
  for (m = 0; m < M; m++)
    A[m] = sycl::malloc_shared<float>(K, sycl_device, sycl_context);

  auto B = sycl::malloc_shared<float *>(K, sycl_device, sycl_context);
  for (k = 0; k < K; k++)
    B[k] = sycl::malloc_shared<float>(N, sycl_device, sycl_context);

  auto C = sycl::malloc_shared<float *>(M, sycl_device, sycl_context);
  for (m = 0; m < M; m++)
    C[m] = sycl::malloc_shared<float>(N, sycl_device, sycl_context);

  // Initialize matrices A, B, and C
  for (m = 0; m < M; m++)
    for (k = 0; k < K; k++)
      A[m][k] = 1.0;

  for (k = 0; k < K; k++)
    for (n = 0; n < N; n++)
      B[k][n] = 1.0;

  for (m = 0; m < M; m++)
    for (n = 0; n < N; n++)
      C[m][n] = 0.0;

  auto start_time = std::chrono::system_clock::now(); // Start timer

  // Offload matrix multiplication kernel
  Q.parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> id) {
     unsigned int m = id[0];
     unsigned int n = id[1];

     float sum = 0.0;
     for (unsigned int k = 0; k < K; k++)
       sum += A[m][k] * B[k][n];

     C[m][n] = sum;
   }).wait(); // End matrix multiplication

  auto end_time = std::chrono::system_clock::now(); // Stop timer
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  // Check for correctness
  bool errors(false);
  for (m = 0; m < M; m++) {
    for (n = 0; n < N; n++) {
      if (std::abs(C[m][n] - K) > FLT_MIN) {
        errors = true;
        break;
      }
    }
  }
  if (errors)
    std::cout << "Program completed with errors." << std::endl;
  else {
    std::cout << "Program completed without errors." << std::endl;
    std::cout << "Naive DPC++ multiplication of " << M << " x " << K << " and "
              << K << " x " << N << " matrices took " << elapsed_time.count()
              << " seconds." << std::endl;
  }

  // Cleanup
  sycl::free(A, Q);
  sycl::free(B, Q);
  sycl::free(C, Q);
}
