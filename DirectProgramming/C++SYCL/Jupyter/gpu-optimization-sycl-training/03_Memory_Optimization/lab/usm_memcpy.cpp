//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int N = 1024000000;
  //# host allocation using malloc
  auto host_data = static_cast<int *>(malloc(N * sizeof(int)));
  //# USM host allocation using malloc_host
  auto host_data_usm = sycl::malloc_host<int>(N, q);

  //# USM device allocation using malloc_device
  auto device_data_usm = sycl::malloc_device<int>(N, q);

  //# copy mem from host (malloc) to device
  auto e1 = q.memcpy(device_data_usm, host_data, sizeof(int) * N);
    
  //# copy mem from host (malloc_host) to device
  auto e2 = q.memcpy(device_data_usm, host_data_usm, sizeof(int) * N);

  q.wait();

  //# free allocations
  sycl::free(device_data_usm, q);
  sycl::free(host_data_usm, q);
  free(host_data);

  std::cout << "memcpy Time (malloc-to-malloc_device)     : " << (e1.template get_profiling_info<sycl::info::event_profiling::command_end>() - e1.template get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e+9 << " seconds\n";

  std::cout << "memcpy Time (malloc_host-to-malloc_device : " << (e2.template get_profiling_info<sycl::info::event_profiling::command_end>() - e2.template get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e+9 << " seconds\n";

  return 0;
}
