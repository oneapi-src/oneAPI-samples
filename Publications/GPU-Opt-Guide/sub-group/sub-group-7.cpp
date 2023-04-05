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
  memset(data2, 0, sizeof(int) * N);

  auto e = q.submit([&](auto &h) {
    h.parallel_for(
        sycl::nd_range(sycl::range{N / 16}, sycl::range{32}),
        [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          sycl::vec<int, 4> x;

          int base = (it.get_group(0) * 32 +
                      sg.get_group_id()[0] * sg.get_local_range()[0]) *
                     16;

          x = sg.load<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data2[base + 0])));
          sg.store<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data[base + 0])),
              x);
          x = sg.load<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data2[base + 64])));
          sg.store<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data[base + 64])),
              x);
          x = sg.load<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data2[base + 128])));
          sg.store<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data[base + 128])),
              x);
          x = sg.load<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data2[base + 192])));
          sg.store<4>(
              (sycl::
                   multi_ptr<int, sycl::access::address_space::global_space>)(&(
                  data[base + 192])),
              x);
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
