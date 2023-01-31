//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace sycl;
using namespace oneapi::dpl::execution;


int main(){
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
  std::vector<int> v{2,3,1,4};
    
  //# Create a buffer and use buffer iterators in Parallel STL algorithms
  {
    buffer buf(v);
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);

    oneapi::dpl::for_each(make_device_policy(q), buf_begin, buf_end, [](int &a){ a *= 3; });
    oneapi::dpl::sort(make_device_policy(q), buf_begin, buf_end);
  }
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
