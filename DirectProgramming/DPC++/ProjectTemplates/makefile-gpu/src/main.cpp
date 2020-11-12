//==============================================================
// Copyright © 2020, Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: MIT
// =============================================================

// located in $ONEAPI_ROOT/compiler/latest/linux/include/sycl/
#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>

using namespace cl::sycl;

int main() {
  // create GPU device selector
  gpu_selector device_selector;

  // create a buffer
  constexpr int num = 16;
  auto R = range<1>{ num };
  buffer<int> A{ R };

  // create a kernel
  class ExampleKernel;
  queue q{device_selector };
  q.submit([&](handler& h) {
    auto out = A.get_access<access::mode::write>(h);
    h.parallel_for<ExampleKernel>(R, [=](id<1> idx) { out[idx] = idx[0]; });
  });

  // consume result
  auto result = A.get_access<access::mode::read>();
  for (int index = 0; index < num; ++index) {
    std::cout << result[index] << "\n";
  }

  return (EXIT_SUCCESS);
}
