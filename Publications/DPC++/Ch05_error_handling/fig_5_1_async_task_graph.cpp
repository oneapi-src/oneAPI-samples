// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr int size=16;
  buffer<int> B { range{ size } };

  // Create queue on any available device
  queue Q;

  Q.submit([&](handler& h) {
    accessor A{B, h};

      h.parallel_for(size , [=](auto& idx) {
            A[idx] = idx;
          });
      });

  // Obtain access to buffer on the host
  // Will wait for device kernel to execute to generate data
  host_accessor A{B};
  for (int i = 0; i < size; i++)
    std::cout << "data[" << i << "] = " << A[i] << "\n";

  return 0;
}

