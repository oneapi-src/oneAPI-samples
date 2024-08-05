//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>

template <typename T>
auto get_multi_ptr(T *raw_ptr) {
  auto multi_ptr =
    sycl::address_space_cast<
      sycl::access::address_space::global_space,
      sycl::access::decorated::yes>(raw_ptr);
  return multi_ptr;
}

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

	  auto load_ptr0 = get_multi_ptr(&(data2[base + 0*64]));
          x = sg.load<4>(load_ptr0);

	  auto store_ptr0 = get_multi_ptr(&(data[base + 0*64]));
          sg.store<4>(store_ptr0, x);

	  auto load_ptr1 = get_multi_ptr(&(data2[base + 1*64]));
          x = sg.load<4>(load_ptr1);

	  auto store_ptr1 = get_multi_ptr(&(data[base + 1*64]));
          sg.store<4>(store_ptr1, x);

	  auto load_ptr2 = get_multi_ptr(&(data2[base + 2*64]));
          x = sg.load<4>(load_ptr2);

	  auto store_ptr2 = get_multi_ptr(&(data[base + 2*64]));
          sg.store<4>(store_ptr2, x);

	  auto load_ptr3 = get_multi_ptr(&(data2[base + 3*64]));
          x = sg.load<4>(load_ptr3);

	  auto store_ptr3 = get_multi_ptr(&(data[base + 3*64]));
          sg.store<4>(store_ptr3, x);

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
