// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <cassert>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  // Create 3 buffers of 42 ints
  buffer<int> A{range{N}};
  buffer<int> B{range{N}};
  buffer<int> C{range{N}};

  accessor pC{C};

  Q.submit([&](handler &h) {
      accessor aA{A, h, write_only, noinit};
      accessor aB{B, h, write_only, noinit};
      accessor aC{C, h, write_only, noinit};
      h.parallel_for(N, [=](id<1> i) {
          aA[i] = 1;
          aB[i] = 40;
          aC[i] = 0;
        });
    });
  Q.submit([&](handler &h) {
      accessor aA{A, h, read_only};
      accessor aB{B, h, read_only};
      accessor aC{C, h, read_write};
      h.parallel_for(N, [=](id<1> i) { aC[i] += aA[i] + aB[i]; });
    });
  Q.submit([&](handler &h) {
      h.require(pC);
      h.parallel_for(N, [=](id<1> i) { pC[i]++; });
    });

  host_accessor result{C, read_only};

  for (int i = 0; i < N; i++) {
    assert(result[i] == N);
  }
  return 0;
}
