//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

using namespace sycl;
using namespace oneapi::dpl::execution;

const int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    
  //# USM allocator 
  usm_allocator<int, usm::alloc::shared> alloc(q);
  std::vector<int, decltype(alloc)> v(N, alloc);
    
  //# Parallel STL algorithm with USM allocator
  oneapi::dpl::fill(make_device_policy(q), v.begin(), v.end(), 20);
  q.wait();
    
  for (int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
