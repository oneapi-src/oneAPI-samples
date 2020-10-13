//==============================================================
// Copyright © 2020, Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <CL/sycl.hpp>
using namespace cl::sycl;


int main() {
  
  // GPU Device selector
  gpu_selector device_selector;
  
  // Buffer creation
  constexpr int num = 16; 
  auto R = range<1>{ num };
  buffer<int> A{ R };
  
  // Kernel
  class ExampleKernel;
  queue q{ device_selector };
  q.submit([&](handler& h) {
    auto out = A.get_access<access::mode::write>(h);
    h.parallel_for<ExampleKernel>(R, [=](id<1> idx) { out[idx] = idx[0]; }); 
  });
  
  // Consume result
  auto result = A.get_access<access::mode::read>();
  for (int i = 0; i < num; ++i)
    std::cout << result[i] << "\n";

  return 0;
}