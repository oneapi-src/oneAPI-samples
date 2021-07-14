// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

int main(){
  sycl::queue Q;
  const int n = 10;
  sycl::usm_allocator<int, sycl::usm::alloc::shared> 
                         alloc(Q.get_context(), Q.get_device());
  std::vector<int, decltype(alloc)> vec(n, alloc);

  std::fill(dpstd::execution::make_device_policy(Q), 
                              vec.begin(), vec.end(), 78);
  Q.wait();

  return 0;
}
