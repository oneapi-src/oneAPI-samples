//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  //# USM allocation using malloc_shared
  constexpr int N = 16;
  int *data = sycl::malloc_shared<int>(N, q);

  //# Initialize data array
  for (int i = 0; i < N; i++) data[i] = 10;

  //# Modify data array on device
  q.parallel_for(N, [=](auto i) { data[i] += 1; }).wait();

  //# print output
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";std::cout << "\n";
  sycl::free(data, q);
  return 0;
}
