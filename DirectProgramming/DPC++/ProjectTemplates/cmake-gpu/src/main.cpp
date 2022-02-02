//==============================================================
// Copyright © 2020, Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: MIT
// =============================================================

// located in $ONEAPI_ROOT/compiler/latest/linux/include/sycl/
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

static const int num = 16;

int main() {
  // create GPU device selector
  gpu_selector device_selector;

  // create a kernel
  queue q{device_selector};

  // USM allocation using malloc_shared
  int *data = malloc_shared<int>(num, q);

  q.submit([&](handler& h) {
    h.parallel_for(range<1>{num}, [=](id<1> idx) {data[idx] = idx[0];});
  }).wait();

  // print output
  for (int i = 0; i < num; i++) std::cout << data[i] << std::endl;
  free(data, q);

  return (EXIT_SUCCESS);
}
