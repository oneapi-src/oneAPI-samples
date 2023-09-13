//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int N = 32;
  auto data = sycl::malloc_shared<int>(N, q);

  auto e = q.submit([&](auto &h) {
    sycl::local_accessor<int, 1> slm(sycl::range(32 * 64), h);
    h.parallel_for(sycl::nd_range(sycl::range{N}, sycl::range{32}), [=](sycl::nd_item<1> it) {
     int i = it.get_global_linear_id();
     int j = it.get_local_linear_id();

     slm[j * 16] = 0;
     sycl::group_barrier(it.get_group());

     for (int m = 0; m < 1024 * 1024; m++) {
       slm[j * 16] += i * m;
       sycl::group_barrier(it.get_group());
     }

     data[i] = slm[j * 16];
   });
  });
  q.wait();
  std::cout << "Kernel time = "
            << (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>())
            << " ns\n";
  return 0;
}
