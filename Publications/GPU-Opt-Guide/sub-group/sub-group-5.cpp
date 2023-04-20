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
  constexpr int N = 1024 * 1024;
  int *data = sycl::malloc_shared<int>(N, q);
  int *data2 = sycl::malloc_shared<int>(N, q);
  memset(data2, 0xFF, sizeof(int) * N);

  auto e = q.submit([&](auto &h) {
    h.parallel_for(sycl::nd_range(sycl::range{N / 16}, sycl::range{32}),
                   [=](sycl::nd_item<1> it) {
                     int i = it.get_global_linear_id();
                     auto sg = it.get_sub_group();
                     int sgSize = sg.get_local_range()[0];
                     i = (i / sgSize) * sgSize * 16 + (i % sgSize);
                     for (int j = 0; j < sgSize * 16; j += sgSize) {
                       data[i + j] = data2[i + j];
                     }
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
