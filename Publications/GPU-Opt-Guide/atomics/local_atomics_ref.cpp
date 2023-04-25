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
  constexpr int NUM_WG = N / M;
  int total = 0;
  int *a = static_cast<int *>(malloc(sizeof(int) * N));
  for (int i = 0; i < N; i++)
    a[i] = 1;
  sycl::queue q({sycl::property::queue::enable_profiling()});
  sycl::buffer<int> global(&total, 1);
  sycl::buffer<int> bufa(a, N);
  auto e1 = q.submit([&](sycl::handler &h) {
    sycl::accessor b(global, h);
    sycl::accessor acc_a(bufa, h, sycl::read_only);
    auto acc = sycl::local_accessor<int, 1>(NUM_WG, h);
    h.parallel_for(sycl::nd_range<1>(N, M), [=](auto it) {
      auto i = it.get_global_id(0);
      auto group_id = it.get_group(0);
      sycl::atomic_ref<int, sycl::memory_order_relaxed,
                       sycl::memory_scope_device,
                       sycl::access::address_space::local_space>
          atomic_op(acc[group_id]);
      sycl::atomic_ref<int, sycl::memory_order_relaxed,
                       sycl::memory_scope_device,
                       sycl::access::address_space::global_space>
          atomic_op_global(b[0]);
      atomic_op += acc_a[i];
      it.barrier(sycl::access::fence_space::local_space);
      if (it.get_local_id() == 0)
        atomic_op_global += acc[group_id];
    });
  });
  sycl::host_accessor h_global(global);
  std::cout << "Reduction Sum : " << h_global[0] << "\n";
  int total_time =
      (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
       e1.get_profiling_info<sycl::info::event_profiling::command_start>());
  std::cout << "Kernel Execution Time of Local Atomics : " << total_time
            << "\n";
  return 0;
}
