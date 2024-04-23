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
    h.parallel_for(
        sycl::nd_range(sycl::range{N / 16}, sycl::range{32}), [=
    ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
          sycl::ext::oneapi::sub_group sg = it.get_sub_group();
          sycl::vec<int, 8> x;

          using global_ptr =
              sycl::multi_ptr<int, sycl::access::address_space::global_space>;
          int base = (it.get_group(0) * 32 +
                      sg.get_group_id()[0] * sg.get_local_range()[0]) *
                     16;
          x = sg.load<8>(global_ptr(&(data2[base + 0])));
          sg.store<8>(global_ptr(&(data[base + 0])), x);
          x = sg.load<8>(global_ptr(&(data2[base + 128])));
          sg.store<8>(global_ptr(&(data[base + 128])), x);
        });
  });

  q.wait();
  std::cout << "Kernel time = " << (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) << " ns\n";
  return 0;
}
