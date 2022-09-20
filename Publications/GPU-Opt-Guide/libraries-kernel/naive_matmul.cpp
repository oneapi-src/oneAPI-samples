//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <cfloat>
#include <chrono>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <m> <k>\n";
    exit(1);
  }

  // Set matrix dimensions
  unsigned int m, k, n, M, K, N;
  M = N = std::stoi(argv[1]);
  K = std::stoi(argv[2]);

  // Allocate matrices A, B, and C
  float **A = new float *[M];
  for (m = 0; m < M; m++)
    A[m] = new float[K];

  float **B = new float *[K];
  for (k = 0; k < K; k++)
    B[k] = new float[N];

  float **C = new float *[M];
  for (m = 0; m < M; m++)
    C[m] = new float[N];

  // Initialize matrices A and B
  for (m = 0; m < M; m++)
    for (k = 0; k < K; k++)
      A[m][k] = 1.0;

  for (k = 0; k < K; k++)
    for (n = 0; n < N; n++)
      B[k][n] = 1.0;

  auto start_time = std::chrono::system_clock::now(); // Start timer

  // Multiply matrices A and B
  for (m = 0; m < M; m++) {
    for (n = 0; n < N; n++) {
      C[m][n] = 0.0;
      for (k = 0; k < K; k++) {
        C[m][n] += A[m][k] * B[k][n];
      }
    }
  } // End matrix multiplication

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
    std::cout << "Naive C++ multiplication of " << M << " x " << K << " and "
              << K << " x " << N << " matrices took " << elapsed_time.count()
              << " seconds." << std::endl;
  }

  // Cleanup
  delete[] A;
  delete[] B;
  delete[] C;
}
