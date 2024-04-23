//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  //# initialize data on host
  constexpr int N = 16;
  int host_data[N];
  for (int i = 0; i < N; i++) host_data[i] = 10;

  //# Explicit USM allocation using malloc_device
  int *device_data = sycl::malloc_device<int>(N, q);

  //# copy mem from host to device
  q.memcpy(device_data, host_data, sizeof(int) * N).wait();

  //# update device memory
  q.parallel_for(N, [=](auto i) { device_data[i] += 1; }).wait();

  //# copy mem from device to host
  q.memcpy(host_data, device_data, sizeof(int) * N).wait();

  //# print output
  for (int i = 0; i < N; i++) std::cout << host_data[i] << " ";std::cout <<"\n";
  sycl::free(device_data, q);
  return 0;
}