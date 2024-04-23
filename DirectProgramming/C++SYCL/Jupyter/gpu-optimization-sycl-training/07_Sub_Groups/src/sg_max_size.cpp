//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

constexpr int N = 15;

int main() {
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()<< "\n";

  int *data = sycl::malloc_shared<int>(N + N + 2, q);

  for (int i = 0; i < N + N + 2; i++) {
    data[i] = i;
  }

  // Snippet begin
  auto e = q.submit([&](auto &h) {
    sycl::stream out(65536, 128, h);
    h.parallel_for(
        sycl::nd_range<1>(15, 15), [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
          int i = it.get_global_linear_id();
          auto sg = it.get_sub_group();
          int sgSize = sg.get_local_range()[0];
          int sgMaxSize = sg.get_max_local_range()[0];
          int sId = sg.get_local_id()[0];
          int j = data[i];
          int k = data[i + sgSize];
          out << "globalId = " << i << " sgMaxSize = " << sgMaxSize
              << " sgSize = " << sgSize << " sId = " << sId << " j = " << j
              << " k = " << k << sycl::endl;
        });
  });
  q.wait();
  return 0;
}
