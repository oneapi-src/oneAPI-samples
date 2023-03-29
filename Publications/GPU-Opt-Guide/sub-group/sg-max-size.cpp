//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <CL/sycl.hpp>

constexpr int N = 7;

// DPC++ asynchronous exception handler
static auto exception_handler = [](sycl::exception_list eList) {
  for (std::exception_ptr const &e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};

int main() {
  sycl::queue q{sycl::gpu_selector_v, exception_handler,
                sycl::property::queue::enable_profiling{}};

  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  std::cout << "Max Compute Units: "
            << q.get_device().get_info<sycl::info::device::max_compute_units>()
            << std::endl;
  std::cout << "Max Work Item Size: "
            << q.get_device()
                   .get_info<sycl::info::device::max_work_item_sizes<3>>()[0]
            << " "
            << q.get_device()
                   .get_info<sycl::info::device::max_work_item_sizes<3>>()[1]
            << " "
            << q.get_device()
                   .get_info<sycl::info::device::max_work_item_sizes<3>>()[2]
            << std::endl;
  std::cout
      << "Max Work Group Size: "
      << q.get_device().get_info<sycl::info::device::max_work_group_size>()
      << std::endl;
  std::cout << "Preffered Vector Width Float: "
            << q.get_device()
                   .get_info<sycl::info::device::preferred_vector_width_float>()
            << std::endl;
  std::cout << "Native Vector Width Float: "
            << q.get_device()
                   .get_info<sycl::info::device::native_vector_width_float>()
            << std::endl;
  std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;

  int *data = sycl::malloc_shared<int>(N + N + 2, q);

  for (int i = 0; i < N + N + 2; i++) {
    data[i] = i;
  }

  // Snippet begin
  auto e = q.submit([&](auto &h) {
    sycl::stream out(65536, 128, h);
    h.parallel_for(sycl::nd_range<1>(7, 7),
                   [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
                     int i = it.get_global_linear_id();
                     auto sg = it.get_sub_group();
                     int sgSize = sg.get_local_range()[0];
                     int sgMaxSize = sg.get_max_local_range()[0];
                     int sId = sg.get_local_id()[0];
                     int j = data[i];
                     int k = data[i + sgSize];
                     out << "globalId = " << i << " sgMaxSize = " << sgMaxSize
                         << " sgSize = " << sgSize << " sId = " << sId
                         << " j = " << j << " k = " << k << sycl::endl;
                   });
  });
  q.wait();
  // Snippet end
  std::cout << "Kernel time = "
            << (e.template get_profiling_info<
                    sycl::info::event_profiling::command_end>() -
                e.template get_profiling_info<
                    sycl::info::event_profiling::command_start>())
            << " ns" << std::endl;
  return 0;
}
