//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024; // global size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  std::vector<int> data(N);
  for (int i = 0; i < N; i++) data[i] = i;
  int sum = 0;
  {
    //# create buffers for data and sum
    buffer buf_data(data);
    buffer buf_sum(&sum, range(1));

    //# Reduction Kernel using atomics 
    q.submit([&](auto &h) {
      accessor data_acc(buf_data, h, sycl::read_only);
      accessor sum_acc(buf_sum, h);

      h.parallel_for(N, [=](auto i) {
        auto sum_atomic = atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(sum_acc[0]);
        sum_atomic += data_acc[i];
      });
    });
  }
  std::cout << "Sum = " << sum << "\n";

  return 0;
}
