//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  constexpr int N = 1024 * 1000 * 1000;
  constexpr int M = 256;
  int sum = 0;
  int *data = static_cast<int *>(malloc(sizeof(int) * N));
  for (int i = 0; i < N; i++) data[i] = 1;

  sycl::queue q({sycl::property::queue::enable_profiling()});
  sycl::buffer<int> buf_sum(&sum, 1);
  sycl::buffer<int> buf_data(data, N);

  auto e = q.submit([&](sycl::handler &h) {
    sycl::accessor acc_sum(buf_sum, h);
    sycl::accessor acc_data(buf_data, h, sycl::read_only);
    h.parallel_for(sycl::nd_range<1>(N, M), [=](auto it) {
      auto i = it.get_global_id();
      sycl::atomic_ref<int, sycl::memory_order_relaxed,
        sycl::memory_scope_device, sycl::access::address_space::global_space>
        atomic_op(acc_sum[0]);
      atomic_op += acc_data[i];
    });
  });
  sycl::host_accessor h_a(buf_sum);

  std::cout << "Reduction Sum : " << sum << "\n";
  auto total_time = (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9;
  std::cout << "Kernel Execution Time of Global Atomics : " << total_time << "seconds\n";
  return 0;
}
