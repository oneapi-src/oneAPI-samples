// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {
// BEGIN CODE SNIP

  constexpr size_t N = 1024;

  // Set up queue on any available device
  queue q;

  // Create host containers to initialize on the host
  std::vector<int> in_vec(N), out_vec(N);

  // Initialize input and output vectors
  for (int i=0; i < N; i++) in_vec[i] = i;
  std::fill(out_vec.begin(), out_vec.end(), 0);

  // Create buffers using host allocations (vector in this case)
  buffer in_buf{in_vec}, out_buf{out_vec};

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) {
        out[idx] = in[idx];
        });
    });

  // BUG!!! We're using the host allocation out_vec, but the buffer out_buf
  // is still alive and owns that allocation!  We will probably see the
  // initialiation value (zeros) printed out, since the kernel probably
  // hasn't even run yet, and the buffer has no reason to have copied
  // any output back to the host even if the kernel has run.
  for (int i=0; i<N; i++) std::cout << "out_vec[" << i << "]=" << out_vec[i] << "\n";

// END CODE SNIP
  return 0;
}
