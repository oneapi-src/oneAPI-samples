// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
  queue Q;

  bool *resArray = malloc_shared<bool>(1, Q);
  resArray[0] = true;

  Q.single_task([=]() {
    sycl::vec<int, 4> old_v = sycl::vec<int, 4>(000, 100, 200, 300);
    sycl::vec<int, 4> new_v = sycl::vec<int, 4>();

    new_v.rgba() = old_v.abgr();
    int vals[] = {300, 200, 100, 000};

    if (new_v.r() != vals[0] || new_v.g() != vals[1] ||
        new_v.b() != vals[2] || new_v.a() != vals[3]) {
      resArray[0] = false;
    }
  }) .wait();

  if (resArray[0])
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
  free(resArray, Q);
}
