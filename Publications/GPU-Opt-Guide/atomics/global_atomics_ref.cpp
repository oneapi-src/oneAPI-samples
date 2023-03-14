//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
int main() {
  constexpr int N = 256 * 256;
  constexpr int M = 512;
  int total = 0;
  int *a = static_cast<int *>(malloc(sizeof(int) * N));
  for (int i = 0; i < N; i++)
    a[i] = 1;
  sycl::queue q({sycl::property::queue::enable_profiling()});
  sycl::buffer<int> buf(&total, 1);
  sycl::buffer<int> bufa(a, N);
  auto e = q.submit([&](sycl::handler &h) {
    sycl::accessor acc(buf, h);
    sycl::accessor acc_a(bufa, h, sycl::read_only);
    h.parallel_for(sycl::nd_range<1>(N, M), [=](auto it) {
      auto i = it.get_global_id();
      sycl::atomic_ref<int, sycl::memory_order_relaxed,
                       sycl::memory_scope_device,
                       sycl::access::address_space::global_space>
          atomic_op(acc[0]);
      atomic_op += acc_a[i];
    });
  });
  sycl::host_accessor h_a(buf);
  std::cout << "Reduction Sum : " << h_a[0] << "\n";
  std::cout
      << "Kernel Execution Time of Global Atomics Ref: "
      << e.get_profiling_info<sycl::info::event_profiling::command_end>() -
             e.get_profiling_info<sycl::info::event_profiling::command_start>()
      << "\n";
  return 0;
}
