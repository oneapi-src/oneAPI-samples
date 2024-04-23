//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

int main() {
  constexpr int N = 16;
  std::vector<int> host_data(N, 10);

  sycl::queue q;
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  //# Modify data array on device
  sycl::buffer buffer_data(host_data);
  q.submit([&](sycl::handler& h) {
    sycl::accessor device_data(buffer_data, h);
    h.parallel_for(N, [=](auto i) { device_data[i] += 1; });
  });
  sycl::host_accessor ha(buffer_data, sycl::read_only);

  //# print output
  for (int i = 0; i < N; i++) std::cout << ha[i] << " ";std::cout << "\n";
  return 0;
}
