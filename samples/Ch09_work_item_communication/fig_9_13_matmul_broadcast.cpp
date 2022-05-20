// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <chrono>
using namespace sycl;

extern const int matrixSize = 128;
static const int iterations = 16;

template <typename T>
double run_sycl(
    const std::vector<T>& vecA,
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

  queue Q;
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>() << "\n";

  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();

    Q.submit([&](handler& h) {
      // Traditional accessors, representing matrices in global memory:
      accessor matrixA{bufA, h};
      accessor matrixB{bufB, h};
      accessor matrixC{bufC, h};

      // Local accessor, for one matrix tile:
      constexpr int tile_size = 16;
      auto tileA =
          accessor<T, 1, access::mode::read_write, access::target::local>(
              tile_size, h);

// BEGIN CODE SNIP
      h.parallel_for(
          nd_range<2>{{M, N}, {1, tile_size}}, [=](nd_item<2> item) {
            // Indices in the global index space:
            int m = item.get_global_id()[0];
            int n = item.get_global_id()[1];

            // Index in the local index space:
            int i = item.get_local_id()[1];

            T sum = 0;
            for (int kk = 0; kk < K; kk += tile_size) {
              // Load the matrix tile from matrix A, and synchronize
              // to ensure all work-items have a consistent view
              // of the matrix tile in local memory.
              tileA[i] = matrixA[m][kk + i];
              group_barrier(item.get_group());

              // Perform computation using the local memory tile, and
              // matrix B in global memory.
              for (int k = 0; k < tile_size; k++) {
                // Because the value of k is the same for all work-items
                // in the group, these reads from tileA are broadcast
                // operations.
                sum += tileA[k] * matrixB[kk + k][n];
              }

              // After computation, synchronize again, to ensure all
              // reads from the local memory tile are complete.
              group_barrier(item.get_group());
            }

            // Write the final result to global memory.
            matrixC[m][n] = sum;
          });
// END CODE SNIP
    });
    Q.wait();

    auto duration = std::chrono::steady_clock::now() - start;
    auto time = std::chrono::duration_cast<ns>(duration).count();

    best_time = std::min(time, best_time);
  }

  double best_seconds = (double)best_time / 1e9;

  return best_seconds;
}

template double run_sycl<float>(
    const std::vector<float>& vecA,
    const std::vector<float>& vecB,
    std::vector<float>& vecC);
