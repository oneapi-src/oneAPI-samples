//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 16;

int main() {
  queue q;
  int *data = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) data[i] = i;

  //# Explicit USM allocation using malloc_device
  int *data_device = static_cast<int *>(malloc_device(N * sizeof(int), q));

  //# copy mem from host to device
  auto e1 = q.memcpy(data_device, data, sizeof(int) * N);

  //# update device memory
  auto e2 = q.submit([&](handler &h) {
    h.depends_on(e1);
    h.parallel_for(range<1>(N), [=](id<1> i) { data_device[i] *= 2; });
  });

  //# copy mem from device to host
  q.submit([&](handler &h) {
    h.depends_on(e2);
    h.memcpy(data, data_device, sizeof(int) * N);
  }).wait();

  //# print output
  for (int i = 0; i < N; i++) std::cout << data[i] << std::endl;
  free(data_device, q);
  free(data);
  return 0;
}
