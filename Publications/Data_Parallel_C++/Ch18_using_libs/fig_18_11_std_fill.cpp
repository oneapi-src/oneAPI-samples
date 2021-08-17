// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <CL/sycl.hpp>

int main(){
  sycl::queue Q;
  sycl::buffer<int> buf { 1000 };

  //auto buf_begin = dpstd::begin(buf); // original line from book, valid for toolkits 2021.1-2021.22
  auto buf_begin = oneapi::dpl::begin(buf); //updated for oneAPI Toolkits 2021.3+  
  //auto buf_end   = dpstd::end(buf); // original line from book, valid for toolkits 2021.1-2021.22
  auto buf_end   = oneapi::dpl::end(buf);   //updated for oneAPI Toolkits 2021.3+  
  
  //auto policy = dpstd::execution::make_device_policy<class fill>( Q );      // original line from book, valid for toolkits 2021.1-2021.22
  auto policy = oneapi::dpl::execution::make_device_policy<class fill>( Q );  //updated for oneAPI Toolkits 2021.3+  
  std::fill(policy, buf_begin, buf_end, 42);

  return 0;
}