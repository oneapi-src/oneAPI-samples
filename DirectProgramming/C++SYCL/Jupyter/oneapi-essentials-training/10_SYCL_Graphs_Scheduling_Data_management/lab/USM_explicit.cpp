//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include<array>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  std::array<int,N> host_array;
  int *device_array = malloc_device<int>(N, Q);

  for (int i = 0; i < N; i++)
    host_array[i] = N;

  // Submit the queue
  Q.submit([&](handler &h) {
      // copy hostArray to deviceArray
      h.memcpy(device_array, &host_array[0], N * sizeof(int));
    });
  Q.wait();

  Q.submit([&](handler &h) {
      h.parallel_for(N, [=](id<1> i) { device_array[i]++; });
    });
  Q.wait();

  Q.submit([&](handler &h) {
      // copy deviceArray back to hostArray
      h.memcpy(&host_array[0], device_array, N * sizeof(int));
    });
  Q.wait();

  free(device_array, Q);
  return 0;
}
