// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {
  // Set up queue on any available device
  queue Q;

  // Initialize input and output memory on the host
  constexpr size_t N = 256;
  constexpr size_t M = 256;
  std::vector<int> a(N * M), b(N * M), c(N * M);
  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 2);
  std::fill(c.begin(), c.end(), 0);

  {
    // Create buffers associated with inputs and output
    buffer<int, 2> a_buf(a.data(), range<2>(N, M)),
        b_buf(b.data(), range<2>(N, M)), c_buf(c.data(), range<2>(N, M));

    // Submit the kernel to the queue
    Q.submit([&](handler& h) {
      accessor a{a_buf, h};
      accessor b{b_buf, h};
      accessor c{c_buf, h};

// START CODE SNIP
      h.parallel_for(range{N, M}, [=](id<2> idx) {
        c[idx] = a[idx] + b[idx];
      });
// END CODE SNIP
    });
  }

  // Check that all outputs match expected value
  bool passed = std::all_of(c.begin(), c.end(), [](int i) {
    return (i == 3);
  });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}
