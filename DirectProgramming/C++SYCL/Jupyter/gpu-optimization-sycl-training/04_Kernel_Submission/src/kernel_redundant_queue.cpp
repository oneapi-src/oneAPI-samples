//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

constexpr int N = 1024000;
constexpr int ITER = 1000;

int main() {

  std::vector<int> data(N);
  sycl::buffer<int> data_buf(data);

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  //# kernel to initialize data 
  sycl::queue q1;
  q1.submit([&](auto &h) {
    sycl::accessor data_acc(data_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(N, [=](auto i) { data_acc[i] = i; });
  }).wait();

  //# for-loop with kernel computation
  for (int i = 0; i < ITER; i++) {

    sycl::queue q2;

    q2.submit([&](auto &h) {
      sycl::accessor data_acc(data_buf, h);
      h.parallel_for(N, [=](auto i) {
        data_acc[i] += 1;
      });
    });
    sycl::host_accessor ha(data_buf);

  }
  std::cout << "data[0] = " << data[0] << "\n";
    
  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "Compute Duration: " << duration / 1e+9 << " seconds\n";

  return 0;
}
