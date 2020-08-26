//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <dpstd/execution>
#include <dpstd/algorithm>
#include <dpstd/iterators.h>
using namespace sycl;
using namespace dpstd::execution;

int main(){
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
  std::vector<int> v{2,3,1,4};
    
  //# Create a buffer and use buffer iterators in Parallel STL algorithms
  {
    buffer<int> buf{v.data(), v.size()};

    std::for_each(make_device_policy(q), dpstd::begin(buf), dpstd::end(buf), [](int &a){ a *= 2; });
    std::sort(make_device_policy(q), dpstd::begin(buf), dpstd::end(buf));
  }
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << std::endl;
  return 0;
}
