//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

using namespace sycl;
constexpr int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
  std::vector<int> v(N);
    
  //# Parallel STL fill function with device policy
  oneapi::dpl::fill(oneapi::dpl::execution::make_device_policy(q), v.begin(), v.end(), 20);
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
