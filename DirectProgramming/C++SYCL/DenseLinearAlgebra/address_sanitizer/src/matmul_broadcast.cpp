// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <chrono>
#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

extern const int matrixSize = 128;
static const int iterations = 16;

// T is the type of data stored in the matrix
template <typename T>
double run_sycl(const std::vector<T>& vecA,
                const std::vector<T>& vecB,
                std::vector<T>& vecC) {
  using ns = std::chrono::nanoseconds;
  ns::rep best_time = std::numeric_limits<ns::rep>::max();

  const int M = matrixSize;
  const int N = matrixSize;
  const int K = matrixSize;

  std::fill(vecC.begin(), vecC.end(), (T)0);

  buffer<T, 2> bufA{vecA.data(), range<2>{M, K}};
  buffer<T, 2> bufB{vecB.data(), range<2>{K, N}};
  buffer<T, 2> bufC{vecC.data(), range<2>{M, N}};

  queue q;
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();

    q.submit([&](handler& h) {
      // Traditional accessors, representing matrices in
      // global memory:
      accessor matrixA{bufA, h};
      accessor matrixB{bufB, h};
      accessor matrixC{bufC, h};

      // Local accessor, for one matrix tile:
      constexpr int tile_size = 16;
      auto tileA = local_accessor<T, 1>(tile_size, h);

      // BEGIN CODE SNIP
      h.parallel_for(
          nd_range<2>{{M, N}, {1, tile_size}},
          [=](nd_item<2> item) {
            // Indices in the global index space:
            int m = item.get_global_id()[0];
            int n = item.get_global_id()[1];

            // Index in the local index space:
            int i = item.get_local_id()[1];

            // Template type T is the type of data stored in
            // the matrix
            T sum = 0;
            for (int kk = 0; kk < K; kk += tile_size) {
              // Load the matrix tile from matrix A, and
              // synchronize to ensure all work-items have a
              // consistent view of the matrix tile in local
              // memory.
              tileA[i+1] = matrixA[m][kk + i]; // BUG: i => i+1
              group_barrier(item.get_group());

              // Perform computation using the local memory
              // tile, and matrix B in global memory.
              for (int k = 0; k < tile_size; k++) {
                // Because the value of k is the same for
                // all work-items in the group, these reads
                // from tileA are broadcast operations.
                sum += tileA[k] * matrixB[kk + k][n];
              }

              // After computation, synchronize again, to
              // ensure all reads from the local memory tile
              // are complete.
              group_barrier(item.get_group());
            }

            // Write the final result to global memory.
            matrixC[m][n] = sum;
          });
      // END CODE SNIP
    });
    q.wait();

    auto duration =
        std::chrono::steady_clock::now() - start;
    auto time =
        std::chrono::duration_cast<ns>(duration).count();

    best_time = std::min(time, best_time);
  }

  double best_seconds = (double)best_time / 1e9;

  return best_seconds;
}

// This function must be implemented for each sample:
template <typename T>
double run_sycl(const std::vector<T>& vecA,
                const std::vector<T>& vecB,
                std::vector<T>& vecC);

template <typename T>
static T rand_uniform_01() {
  return T(std::rand()) / T(RAND_MAX);
}

template <typename T>
static std::vector<T> make_random_square_matrix() {
  std::vector<T> matrix(matrixSize * matrixSize);
  std::generate_n(matrix.data(), matrix.size(),
                  rand_uniform_01<T>);
  return matrix;
}

template <typename T>
static void compute_reference(const std::vector<T>& matrixA,
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
int check_results(const std::vector<T>& matrixC,
                  const std::vector<T>& referenceC) {
  const int M = matrixSize;
  const int N = matrixSize;

  float err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float localErr = fabs(matrixC[i] - referenceC[i]) /
                     std::max(fabs(matrixC[i]),
                              fabs(referenceC[i]));
    err = std::max(localErr, err);
    if (localErr >= 0.001f) {
      std::cerr << "Error at index " << i << ": Wanted "
                << referenceC[i] << ", got " << matrixC[i]
                << std::endl;
      break;
    }
  }

  return err < 0.001f;
}

using matrixType = float;

int main() {
  auto matrixA = make_random_square_matrix<matrixType>();
  auto matrixB = make_random_square_matrix<matrixType>();
  auto referenceC =
      std::vector<matrixType>(matrixSize * matrixSize, 0);
  compute_reference(matrixA, matrixB, referenceC);

  auto matrixC =
      std::vector<matrixType>(matrixSize * matrixSize, 0);
  auto seconds = run_sycl(matrixA, matrixB, matrixC);

  if (!check_results(matrixC, referenceC)) {
    std::cerr << "Results did not validate!" << std::endl;
    return -1;
  }

  auto gflops = double(matrixSize) * matrixSize *
                (matrixSize +   // multiplications
                 matrixSize) /  // additions
                seconds /
                1e9;
  std::cout << "Success!\n";
  std::cout << "GFlops: " << gflops << std::endl;

  return 0;
}
