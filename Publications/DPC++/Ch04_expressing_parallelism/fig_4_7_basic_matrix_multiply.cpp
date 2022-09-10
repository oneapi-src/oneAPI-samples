// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <random>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue Q;

  // Initialize input and output memory on the host
  constexpr size_t N = 256;
  std::vector<float> a(N * N), b(N * N), c(N * N);
  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto rng = [&]() {
    return dist(gen);
  };
  std::generate(a.begin(), a.end(), rng);
  std::generate(b.begin(), b.end(), rng);
  std::fill(c.begin(), c.end(), 0);

  {
    // Create buffers associated with inputs and output
    buffer<float, 2> a_buf(a.data(), range<2>(N, N)),
        b_buf(b.data(), range<2>(N, N)), c_buf(c.data(), range<2>(N, N));

    // Submit the kernel to the queue
    Q.submit([&](handler& h) {
      accessor a{a_buf, h};
      accessor b{b_buf, h};
      accessor c{c_buf, h};

// START CODE SNIP
      h.parallel_for(range{N, N}, [=](id<2> idx) {
        int j = idx[0];
        int i = idx[1];
        for (int k = 0; k < N; ++k) {
          c[j][i] += a[j][k] * b[k][i]; // or c[idx] += a[id(j,k)] * b[id(k,i)];
        }
      });
// END CODE SNIP
    });
  }

  // Check that all outputs match serial execution
  bool passed = true;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      float gold = 0;
      for (int k = 0; k < N; ++k) {
        gold += a[j * N + k] * b[k * N + i];
      }
      if (std::abs(gold - c[j * N + i]) / gold > 1.0E-06) {
        passed = false;
      }
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}
