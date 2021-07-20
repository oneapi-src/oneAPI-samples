// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

using matrixType = float;
extern const int matrixSize;

// This function must be implemented for each sample:
template <typename T>
double run_sycl(
    const std::vector<T>& vecA,
    const std::vector<T>& vecB,
    std::vector<T>& vecC);

template <typename T>
static T rand_uniform_01() {
  return T(std::rand()) / T(RAND_MAX);
}

template <typename T>
static std::vector<T> make_random_square_matrix() {
  std::vector<T> matrix(matrixSize * matrixSize);
  std::generate_n(matrix.data(), matrix.size(), rand_uniform_01<T>);
  return matrix;
}

template <typename T>
static void compute_reference(
    const std::vector<T>& matrixA,
    const std::vector<T>& matrixB,
    std::vector<T>& matrixC) {
  const int M = matrixSize;
  const int N = matrixSize;
  const int K = matrixSize;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += matrixA[m * K + k] * matrixB[k * N + n];
      }
      matrixC[m * N + n] = sum;
    }
  }
}

template <typename T>
int check_results(
    const std::vector<T>& matrixC, const std::vector<T>& referenceC) {
  const int M = matrixSize;
  const int N = matrixSize;

  float err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float localErr = std::fabs(matrixC[i] - referenceC[i]) /
        std::max(std::fabs(matrixC[i]), std::fabs(referenceC[i]));
    err = std::max(localErr, err);
    if (localErr >= 0.001f) {
      std::cerr << "Error at index " << i << ": Wanted " << referenceC[i]
                << ", got " << matrixC[i] << std::endl;
      break;
    }
  }

  return err < 0.001f;
}

int main() {
  auto matrixA = make_random_square_matrix<matrixType>();
  auto matrixB = make_random_square_matrix<matrixType>();
  auto referenceC = std::vector<matrixType>(matrixSize * matrixSize, 0);
  compute_reference(matrixA, matrixB, referenceC);

  auto matrixC = std::vector<matrixType>(matrixSize * matrixSize, 0);
  auto seconds = run_sycl(matrixA, matrixB, matrixC);

  if (!check_results(matrixC, referenceC)) {
    std::cerr << "Results did not validate!" << std::endl;
    return -1;
  }

  auto gflops = double(matrixSize) * matrixSize *
      (matrixSize +  // multiplications
       matrixSize) / // additions
      seconds /
      1e9;
  std::cout << "Success!\n";
  std::cout << "GFlops: " << gflops << std::endl;

  return 0;
}
