// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <cmath>
#include <iostream>
using namespace sycl;

int main() {
  constexpr int size = 9;
  std::array<float, size> A;
  std::array<float, size> B;

  bool pass = true;

  for (int i = 0; i < size; ++i) { A[i] = i; B[i] = i; }

  queue Q;

  range sz{size};

  buffer<float> bufA(A);
  buffer<float> bufB(B);
  buffer<bool>   bufP(&pass, 1);

  Q.submit([&](handler &h) {
    accessor accA{ bufA, h};
    accessor accB{ bufB, h};
    accessor accP{ bufP, h};

    h.parallel_for(size, [=](id<1> idx) {
      accA[idx] = std::log(accA[idx]);
      accB[idx] = sycl::log(accB[idx]);
      if (!sycl::isequal( accA[idx], accB[idx]) ) {
        accP[0] = false;
      }
    });
  });

  host_accessor host_A(bufA);
  host_accessor host_P(bufP);

  if (host_P[0] && host_A[4] == std::log(4.00f)) {
    std::cout << "Passed\n";
  } else {
    std::cout << "Failed\n";
  }

  return 0;
}
