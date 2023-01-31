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
  auto min = malloc_shared<int>(1, q);
  auto max = malloc_shared<int>(1, q);
  min[0] = 0;
  max[0] = 0;

  //# Reduction Kernel using atomics 
  q.parallel_for(N, [=](auto i) {
    //# STEP 1: create atomic reference for min and max
    auto min_atomic = atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(min[0]);
    auto max_atomic = atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(max[0]);
    
    //# STEP 2: add atomic operation for min and max computation  
    min_atomic.fetch_min(data[i]);
    max_atomic.fetch_max(data[i]);
  }).wait();

  auto mid = 0.0;
  //# STEP 3: Compute mid-range using the min and max 
  mid = (min[0] + max[0]) / 2.0;
  
  std::cout << "Minimum   = " << min[0] << "\n";
  std::cout << "Maximum   = " << max[0] << "\n";
  std::cout << "Mid-Range = " << mid << "\n";

  return 0;
}
