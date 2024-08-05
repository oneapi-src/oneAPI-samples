//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q{sycl::property::queue::enable_profiling{}};

  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int N = 1024 * 1024;
  int *data = sycl::malloc_shared<int>(N, q);
  int *data2 = sycl::malloc_shared<int>(N, q);
  memset(data2, 0xFF, sizeof(int) * N);

  auto e = q.submit([&](auto &h) {
    h.parallel_for(sycl::nd_range(sycl::range{N / 16}, sycl::range{32}),
                   [=](sycl::nd_item<1> it) {
                     int i = it.get_global_linear_id();
                     i = i * 16;
                     for (int j = i; j < (i + 16); j++) {
                       data[j] = data2[j];
                     }
                   });
  });

  q.wait();
  std::cout << "Kernel time = " << (e.template get_profiling_info< sycl::info::event_profiling::command_end>() - e.template get_profiling_info< sycl::info::event_profiling::command_start>())<< " ns\n";
  return 0;
}
