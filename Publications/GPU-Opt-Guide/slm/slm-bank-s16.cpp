//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>

int main() {
  sycl::queue q{sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  // Snippet begin
  constexpr int N = 32;
  int *data = sycl::malloc_shared<int>(N, q);

  auto e = q.submit([&](auto &h) {
    sycl::local_accessor<int, 1> slm(sycl::range(32 * 64), h);
    h.parallel_for(sycl::nd_range(sycl::range{N}, sycl::range{32}),
                   [=](sycl::nd_item<1> it) {
                     int i = it.get_global_linear_id();
                     int j = it.get_local_linear_id();

                     slm[j * 16] = 0;
                     it.barrier(sycl::access::fence_space::local_space);

                     for (int m = 0; m < 1024 * 1024; m++) {
                       slm[j * 16] += i * m;
                       it.barrier(sycl::access::fence_space::local_space);
                     }

                     data[i] = slm[j * 16];
                   });
  });
  // Snippet end
  q.wait();
  std::cout << "Kernel time = "
            << (e.template get_profiling_info<
                    sycl::info::event_profiling::command_end>() -
                e.template get_profiling_info<
                    sycl::info::event_profiling::command_start>())
            << " ns" << std::endl;
  return 0;
}
