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

  buffer<T, 2> bufA{vecA.data(), range<2>{M, K}};
  buffer<T, 2> bufB{vecB.data(), range<2>{K, N}};
  buffer<T, 2> bufC{vecC.data(), range<2>{M, N}};

  queue Q;  // Choose any available device
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>() << "\n";

  for (int i = 0; i < iterations; ++i) {
    // Because the hierarchical kernel accumulates directly into the
    // result matrix C, we need to clear the result matrix before each
    // iteration
    Q.submit([&](handler& h) {
      accessor matrixC{bufC, h};
      h.fill(matrixC, (T)0);
    });
    Q.wait();

    auto start = std::chrono::steady_clock::now();

    Q.submit([&](handler& h) {
      accessor matrixA{bufA, h};
      accessor matrixB{bufB, h};
      accessor matrixC{bufC, h};

// BEGIN CODE SNIP
      const int tileSize = 16;
      range group_size{1, tileSize};
      range num_groups{M, N / tileSize};

      h.parallel_for_work_group(num_groups, group_size, [=](group<2> group) {
        // Because this array is declared at work-group scope
        // it is in local memory
        T tileA[16];

        for (int kk = 0; kk < K; kk += tileSize) {
          // A barrier may be inserted between scopes here
          // automatically, unless the compiler can prove it is
          // not required

          // Load the matrix tile from matrix A
          group.parallel_for_work_item([&](h_item<2> item) {
            int m = item.get_global_id()[0];
            int i = item.get_local_id()[1];
            tileA[i] = matrixA[m][kk + i];
          });

          // A barrier gets inserted here automatically, so all
          // work items have a consistent view of memory

          group.parallel_for_work_item([&](h_item<2> item) {
            int m = item.get_global_id()[0];
            int n = item.get_global_id()[1];
            for (int k = 0; k < tileSize; k++) {
              matrixC[m][n] += tileA[k] * matrixB[kk + k][n];
            }
          });

          // A barrier gets inserted here automatically, too
        }
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
