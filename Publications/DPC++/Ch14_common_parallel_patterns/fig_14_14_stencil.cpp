// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;

int main() {
  queue Q;

  const size_t N = 16;
  const size_t M = 16;
  range<2> stencil_range(N, M);
  range<2> alloc_range(N + 2, M + 2);
  std::vector<float> input(alloc_range.size()), output(alloc_range.size());
  std::iota(input.begin(), input.end(), 1);
  std::fill(output.begin(), output.end(), 0);

  {
    buffer<float, 2> input_buf(input.data(), alloc_range);
    buffer<float, 2> output_buf(output.data(), alloc_range);

    Q.submit([&](handler& h) {
      accessor input{ input_buf, h };
      accessor output{ output_buf, h };

      // Compute the average of each cell and its immediate neighbors
      h.parallel_for(stencil_range, [=](id<2> idx) {
        int i = idx[0] + 1;
        int j = idx[1] + 1;

        float self = input[i][j];
        float north = input[i - 1][j];
        float east = input[i][j + 1];
        float south = input[i + 1][j];
        float west = input[i][j - 1];
        output[i][j] = (self + north + east + south + west) / 5.0f;
      });
    });
  }

  // Check that all outputs match serial execution.
  bool passed = true;
  for (int i = 1; i < N + 1; ++i) {
    for (int j = 1; j < M + 1; ++j) {
      float self = input[i * (M + 2) + j];
      float north = input[(i - 1) * (M + 2) + j];
      float east = input[i * (M + 2) + (j + 1)];
      float south = input[(i + 1) * (M + 2) + j];
      float west = input[i * (M + 2) + (j - 1)];
      float gold = (self + north + east + south + west) / 5.0f;
      if (std::abs(output[i * (M + 2) + j] - gold) >= 1.0E-06) {
        passed = false;
      }
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";
  return (passed) ? 0 : 1;
}
