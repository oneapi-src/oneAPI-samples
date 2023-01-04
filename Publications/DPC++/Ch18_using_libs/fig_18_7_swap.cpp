// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <utility>
using namespace sycl;

class KernelSwap;

int main() {
  std::array <int,2> arr{8,9};
  buffer<int> buf{arr};

  {
    host_accessor host_A(buf);
    std::cout << "Before: " << host_A[0] << ", " << host_A[1] << "\n";
  }  // End scope of host_A so that upcoming kernel can operate on buf

  queue Q;
  Q.submit([&](handler &h) {
      accessor A{buf, h};
      h.single_task([=]() {
        // Call std::swap!
        std::swap(A[0], A[1]);
      });
  });

  host_accessor host_B(buf);
  std::cout << "After:  " << host_B[0] << ", " << host_B[1] << "\n";
  return 0;
}
