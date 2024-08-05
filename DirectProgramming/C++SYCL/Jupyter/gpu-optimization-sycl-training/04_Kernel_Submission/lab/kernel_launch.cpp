//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

constexpr int N = 1024000000;

int main() {
  sycl::queue q;
    
  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  q.parallel_for(N, [=](auto id) {
    /* NOP */
  });
  
  auto k_subm = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    
  q.wait();
    
  auto k_exec = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "Kernel Submission Time: " << k_subm / 1e+9 << " seconds\n";
  std::cout << "Kernel Submission + Execution Time: " << k_exec / 1e+9 << " seconds\n";

  return 0;
}
