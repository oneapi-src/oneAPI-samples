//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

constexpr int N = 1024000000;

int main() {
  sycl::queue q{sycl::property::queue::enable_profiling()};
    
  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  auto e = q.parallel_for(N, [=](auto id) {
    /* NOP */
  });
  e.wait();
    
  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "Kernel Duration  : " << duration / 1e+9 << " seconds\n";

  auto startK = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto endK = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Kernel Execturion: " << (endK - startK) / 1e+9 << " seconds\n";
    
  return 0;
}
