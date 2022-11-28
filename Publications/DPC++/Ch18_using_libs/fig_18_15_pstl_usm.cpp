// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
// old naming dpstd:: is now oneapi::dpl::
// -------------------------------------------------------

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>

int main(){
  sycl::queue Q;
  const int n = 10;
  sycl::usm_allocator<int, sycl::usm::alloc::shared> 
                         alloc(Q.get_context(), Q.get_device());
  std::vector<int, decltype(alloc)> vec(n, alloc);

  std::fill(oneapi::dpl::execution::make_device_policy(Q), 
                              vec.begin(), vec.end(), 78);
  Q.wait();

  return 0;
}
