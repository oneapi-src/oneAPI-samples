// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
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
      accessor matrixA{bufA, h};
      accessor matrixB{bufB, h};
      accessor matrixC{bufC, h};

// BEGIN CODE SNIP
      h.parallel_for(range{M, N}, [=](id<2> id) {
        int m = id[0];
        int n = id[1];

        T sum = 0;
        for (int k = 0; k < K; k++) {
          sum += matrixA[m][k] * matrixB[k][n];
        }

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
