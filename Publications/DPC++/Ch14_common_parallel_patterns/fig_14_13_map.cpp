// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>

using namespace sycl;

int main() {
  queue Q;

  const size_t N = 64;
  float* input = malloc_shared<float>(N, Q);
  float* output = malloc_shared<float>(N, Q);
  std::iota(input, input + N, 1);
  std::fill(output, output + N, 0);

  // Compute the square root of each input value
  Q.parallel_for(N, [=](id<1> i) {
     output[i] = sqrt(input[i]);
   }).wait();

  // Check that all outputs match serial execution.
  bool passed = true;
  for (int i = 0; i < N; ++i) {
    float gold = std::sqrt(input[i]);
    if (std::abs(output[i] - gold) >= 1.0E-06) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(output, Q);
  free(input, Q);
  return (passed) ? 0 : 1;
}
