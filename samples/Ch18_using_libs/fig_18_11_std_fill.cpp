// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <dpstd/execution>
#include <dpstd/algorithm>
#include <dpstd/iterators>
int main(){
  sycl::queue Q;
  sycl::buffer<int> buf { 1000 };

  auto buf_begin = dpstd::begin(buf);
  auto buf_end   = dpstd::end(buf);

  auto policy = dpstd::execution::make_device_policy<class fill>( Q );
  std::fill(policy, buf_begin, buf_end, 42);

  return 0;
}
