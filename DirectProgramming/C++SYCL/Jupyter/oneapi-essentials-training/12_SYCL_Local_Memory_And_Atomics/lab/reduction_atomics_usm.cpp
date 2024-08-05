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

  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;
  auto sum = malloc_shared<int>(1, q);
  sum[0] = 0;

  //# Reduction Kernel using atomics 
  q.parallel_for(N, [=](auto i) {
    auto sum_atomic = atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(sum[0]);
    sum_atomic += data[i];
  }).wait();

  std::cout << "Sum = " << sum[0] << "\n";
  return 0;
}
