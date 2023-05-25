//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

constexpr int N = 1024000000;

int main() {

  std::vector<int> a(N, 1);
  std::vector<int> b(N, 2);
  std::vector<int> c(N);

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
  sycl::queue q;
  {
    sycl::buffer<int> a_buf(a);
    sycl::buffer<int> b_buf(b);
    sycl::buffer<int> c_buf(c);

    q.submit([&](auto &h) {
      // Create device accessors.
      sycl::accessor a_acc(a_buf, h);
      sycl::accessor b_acc(b_buf, h);
      sycl::accessor c_acc(c_buf, h);

      h.parallel_for(N, [=](auto i) {
        c_acc[i] = a_acc[i] + b_acc[i];
      });
    });
  }
  std::cout << "C = " << c[N/2] << "\n";
    
  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "Compute Duration: " << duration / 1e+9 << " seconds\n";

  return 0;
}
