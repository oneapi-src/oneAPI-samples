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

  // Create buffers of size N
  buffer<int> in_buf{N}, out_buf{N};

  // Use host accessors to initialize the data
  { // CRITICAL: Begin scope for host_accessor lifetime!
    host_accessor in_acc{ in_buf }, out_acc{ out_buf };
    for (int i=0; i < N; i++) {
      in_acc[i] = i;
      out_acc[i] = 0;
    }
  } // CRITICAL: Close scope to make host accessors go out of scope!

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) {
      out[idx] = in[idx];
    });
  });

  // Check that all outputs match expected value
  // Use host accessor!  Buffer is still in scope / alive
  host_accessor A{out_buf};

  for (int i=0; i<N; i++) std::cout << "A[" << i << "]=" << A[i] << "\n";

// END CODE SNIP
  return 0;
}
