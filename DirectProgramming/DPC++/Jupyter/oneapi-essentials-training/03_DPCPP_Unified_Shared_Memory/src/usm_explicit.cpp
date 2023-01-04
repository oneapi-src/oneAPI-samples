//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static const int N = 16;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data on host
  int *data = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) data[i] = i;

  //# Explicit USM allocation using malloc_device
  int *data_device = malloc_device<int>(N, q);

  //# copy mem from host to device
  q.memcpy(data_device, data, sizeof(int) * N).wait();

  //# update device memory
  q.parallel_for(range<1>(N), [=](id<1> i) { data_device[i] *= 2; }).wait();

  //# copy mem from device to host
  q.memcpy(data, data_device, sizeof(int) * N).wait();

  //# print output
  for (int i = 0; i < N; i++) std::cout << data[i] << "\n";
  free(data_device, q);
  free(data);
  return 0;
}
