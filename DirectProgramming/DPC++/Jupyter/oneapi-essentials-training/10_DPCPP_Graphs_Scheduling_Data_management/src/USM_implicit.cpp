
// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;
  int *host_array = malloc_host<int>(N, Q);
  int *shared_array = malloc_shared<int>(N, Q);

  for (int i = 0; i < N; i++) {
    // Initialize hostArray on host
    host_array[i] = i;
  }

  // We will learn how to simplify this example later
  Q.submit([&](handler &h) {
      h.parallel_for(N, [=](id<1> i) {
          // access sharedArray and hostArray on device
          shared_array[i] = host_array[i] + 1;
        });
    });
  Q.wait();

  for (int i = 0; i < N; i++) {
    // access sharedArray on host
    host_array[i] = shared_array[i];
  }

  free(shared_array, Q);
  free(host_array, Q);
  return 0;
}
