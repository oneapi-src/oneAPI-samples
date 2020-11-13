//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "kernel.hpp"

using namespace sycl;

int main() {
  std::vector<float> vec_a(kArraySize);
  std::vector<float> vec_b(kArraySize);
  std::vector<float> vec_r(kArraySize);

  // Fill vectors a and b with random float values
  for (size_t i = 0; i < kArraySize; i++) {
    vec_a[i] = rand() / (float)RAND_MAX;
    vec_b[i] = rand() / (float)RAND_MAX;
  }

  // The definition of this function is in a different compilation unit,
  // so host and device code can be separately compiled.
  RunKernel(vec_a, vec_b, vec_r);

  // Test the results
  size_t correct = 0;
  for (size_t i = 0; i < kArraySize; i++) {
    float tmp = vec_a[i] + vec_b[i] - vec_r[i];
    if (tmp * tmp < kTol * kTol) {
      correct++;
    }
  }

  // Summarize results
  if (correct == kArraySize) {
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }

  return !(correct == kArraySize);
}
