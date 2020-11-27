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
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
    
  //# USM allocator 
  usm_allocator<int, usm::alloc::shared> alloc(q);
  std::vector<int, decltype(alloc)> v(N, alloc);
    
  //# Parallel STL algorithm with USM allocator
  std::fill(make_device_policy(q), v.begin(), v.end(), 20);
  q.wait();
    
  for (int i = 0; i < v.size(); i++) std::cout << v[i] << std::endl;
  return 0;
}
