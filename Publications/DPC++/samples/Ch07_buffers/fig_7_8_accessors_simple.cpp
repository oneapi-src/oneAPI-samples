// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cassert>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;
  // create 3 buffers of 42 ints
  buffer<int> A{range{N}};
  buffer<int> B{range{N}};
  buffer<int> C{range{N}};
  accessor pC{C};

  Q.submit([&](handler &h) {
      accessor aA{A, h};
      accessor aB{B, h};
      accessor aC{C, h};
      h.parallel_for(N, [=](id<1> i) {
          aA[i] = 1;
          aB[i] = 40;
          aC[i] = 0;
        });
    });
  Q.submit([&](handler &h) {
      accessor aA{A, h};
      accessor aB{B, h};
      accessor aC{C, h};
      h.parallel_for(N, [=](id<1> i) { aC[i] += aA[i] + aB[i]; });
    });
  Q.submit([&](handler &h) {
      h.require(pC);
      h.parallel_for(N, [=](id<1> i) { pC[i]++; });
    });

  host_accessor result{C};
  for (int i = 0; i < N; i++) {
    assert(result[i] == N);
  }
  return 0;
}
