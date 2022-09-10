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

  const int M = matrixSize;
  const int N = matrixSize;
  const int K = matrixSize;

  using ns = std::chrono::nanoseconds;
  ns::rep best_time = std::numeric_limits<ns::rep>::max();

  std::fill(vecC.begin(), vecC.end(), (T)0);

  buffer<T> bufA{vecA};  // M * K elements
  buffer<T> bufB{vecB};  // K * N elements
  buffer<T> bufC{vecC};  // M * N elements

  queue Q;  // Choose any available device
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>() << "\n";

  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();

    Q.submit([&](handler& h) {
      accessor matrixA{bufA, h};
      accessor matrixB{bufB, h};
      accessor matrixC{bufC, h};

      // A work-group consisting of a single work-item is inefficient!
      h.parallel_for(nd_range<1>{M, 1}, [=](nd_item<1> idx) {
        int m = idx.get_global_id(0);

        for (int n = 0; n < N; n++) {
          T sum = 0;
          for (int k = 0; k < K; k++) {
            sum += matrixA[m * K + k] * matrixB[k * N + n];
          }
          matrixC[m * N + n] = sum;
        }
      });
    });

    Q.wait();  // So that we know the kernel has finished before checking time
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
