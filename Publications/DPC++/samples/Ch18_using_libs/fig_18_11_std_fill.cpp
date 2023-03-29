// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
// old naming dpstd:: is now oneapi::dpl::
// -------------------------------------------------------

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>

int main(){
  sycl::queue Q;
  sycl::buffer<int> buf { 1000 };

  auto buf_begin = oneapi::dpl::begin(buf);
  auto buf_end   = oneapi::dpl::end(buf);

  auto policy = oneapi::dpl::execution::make_device_policy<class fill>( Q );
  std::fill(policy, buf_begin, buf_end, 42);

  return 0;
}
