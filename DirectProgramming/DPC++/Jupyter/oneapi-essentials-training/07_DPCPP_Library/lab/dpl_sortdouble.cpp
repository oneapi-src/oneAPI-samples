//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include<CL/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
  std::vector<int> v{2,3,1,4};
    
  std::for_each(make_device_policy(q), v.begin(), v.end(), [](int &a){ a *= 2; });
  std::sort(make_device_policy(q), v.begin(), v.end());
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << std::endl;
  return 0;
}
