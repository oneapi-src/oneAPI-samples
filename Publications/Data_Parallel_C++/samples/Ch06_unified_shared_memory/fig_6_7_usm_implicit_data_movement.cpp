// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  int* host_array = malloc_host<int>(N, Q);
  int* shared_array = malloc_shared<int>(N, Q);
  for (int i = 0; i < N; i++)
    host_array[i] = i;

  Q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) {
      // access sharedArray and hostArray on device
      shared_array[i] = host_array[i] + 1;
    });
  });
  Q.wait();

  free(shared_array, Q);
  free(host_array, Q);
  return 0;
}
