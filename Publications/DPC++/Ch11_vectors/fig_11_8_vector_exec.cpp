// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include<array>
#include<CL/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int size = 8;

  std::array<float, size> fpData;
  std::array<float4, size> fp4Data;
  for (int i = 0; i < size; i++) {
    fpData[i] = i;
    float b = i*4.0f;
    fp4Data[i] = float4(b, b+1, b+2, b+3);
  }

  buffer fpBuf(fpData);
  buffer fp4Buf(fp4Data);

  queue Q;
  Q.submit([&](handler& h) {
    accessor a{fpBuf, h};
    accessor b{fp4Buf, h};

// BEGIN CODE SNIP
    h.parallel_for(8, [=](id<1> i) {
      float x = a[i];
      float4 y4 = b[i];
      a[i] = x + sycl::length(y4);
    });
// END CODE SNIP
  });

  host_accessor A(fpBuf);
  for (int i = 0; i < size; i++) {
    float b = 4*i;
    if ( 1 < A[i] - (i + std::sqrt(std::pow(b,2)
        + std::pow(b+1,2) + std::pow(b+2,2) + std::pow(b+3,2)))) {
      std::cout << "Failed\n";
      return -1;
    }
  }

  std::cout << "Passed\n";
  return 0;
}

