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

  // Nuance: Create new scope so that we can easily cause buffers to go out
  // of scope and be destroyed
  {

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

  // Close the scope that buffer is alive within!  Causes buffer destruction
  // which will wait until the kernels writing to buffers have completed, and
  // will copy the data from written buffers back to host allocations (our
  // std::vectors in this case).  After the buffer destructor runs, caused by this
  // closing of scope, then it is safe to access the original in_vec and out_vec
  // again!
  }

  // Check that all outputs match expected value
  // WARNING: The buffer destructor must have run for us to safely use in_vec
  // and out_vec again in our host code.  While the buffer is alive it owns those
  // allocations, and they are not safe for us to use!  At the least they will
  // contain values that are not up to date.  This code is safe and correct
  // because the closing of scope above has caused the buffer to be destroyed
  // before this point where we use the vectors again.
  for (int i=0; i<N; i++) std::cout << "out_vec[" << i << "]=" << out_vec[i] << "\n";

// END CODE SNIP

  return 0;
}
