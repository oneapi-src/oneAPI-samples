// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {
// BEGIN CODE SNIP

  constexpr size_t N = 1024;

  // Set up queue on any available device
  queue q;

  // Create buffers using host allocations (vector in this case)
  buffer<int> in_buf{N}, out_buf{N};

  // Use host accessors to initialize the data
  host_accessor in_acc{ in_buf }, out_acc{ out_buf };
  for (int i=0; i < N; i++) {
    in_acc[i] = i;
    out_acc[i] = 0;
  }

  // BUG: Host accessors in_acc and out_acc are still alive! Later q.submits
  // will never start on a device, because the runtime doesn't know that we've
  // finished accessing the buffers via the host accessors.  The device kernels
  // can't launch until the host finishes updating the buffers, since the host
  // gained access first (before the queue submissions).
  // This program will appear to hang!  Use a debugger in that case.

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) {
      out[idx] = in[idx];
    });
  });

  std::cout <<
    "This program will deadlock here!!!  Our host_accessors used\n" <<
    " for data initialization are still in scope, so the runtime won't\n" <<
    " allow our kernel to start executing on the device (the host could\n" <<
    " still be initializing the data that is used by the kernel). The next line\n" <<
    " of code is acquiring a host accessor for the output, which will wait for\n" <<
    " the kernel to run first.  Since in_acc and out_acc have not been\n" <<
    " destructed, the kernel is not safe for the runtime to run, and we deadlock.\n";

  // Check that all outputs match expected value
  // Use host accessor!  Buffer is still in scope / alive
  host_accessor A{out_buf};

  for (int i=0; i<N; i++) std::cout << "A[" << i << "]=" << A[i] << "\n";

// END CODE SNIP
  return 0;
}
