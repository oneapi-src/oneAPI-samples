// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include<array>
#include<sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int workers = 64;
  constexpr int size = workers * 16;

// BEGIN CODE SNIP
  std::array<float, size> fpData;
  for (int i = 0; i < size; i++) {
    fpData[i] = 8.0f;
  }

  buffer fpBuf(fpData);

  queue Q;
  Q.submit([&](handler& h) {
    accessor buf{fpBuf, h};

    h.parallel_for(workers, [=](id<1> idx) {
      float16 inpf16;
      inpf16.load(idx, buf.get_pointer());
      float16 result = inpf16 * 2.0f;
      result.store(idx, buf.get_pointer());
    });
  });
// END CODE SNIP

  host_accessor hostAcc(fpBuf);
  if ( fpData[0] != 16.0f ) {
    std::cout << "Failed\n";
    return -1;
  }

  std::cout << "Passed\n";
  return 0;
}

