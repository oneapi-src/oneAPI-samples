//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <cmath>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  const size_t N = 64;
  float* input = malloc_shared<float>(N, q);
  float* output = malloc_shared<float>(N, q);
  std::iota(input, input + N, 1);
  std::fill(output, output + N, 0);

  // BEGIN CODE SNIP
  // Compute the square root of each input value
  q.parallel_for(N, [=](id<1> i) {
     output[i] = std::sqrt(input[i]);
   }).wait();
  // END CODE SNIP

  // Check that all outputs match serial execution.
  bool passed = true;
  for (int i = 0; i < N; ++i) {
    float gold = std::sqrt(input[i]);
    if (std::abs(output[i] - gold) >= 1.0E-06) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(output, q);
  free(input, q);
  return (passed) ? 0 : 1;
}
