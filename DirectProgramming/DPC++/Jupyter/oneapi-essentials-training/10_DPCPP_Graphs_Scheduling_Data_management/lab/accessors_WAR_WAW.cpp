//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int,N> a, b;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = 0;
  }

  queue Q;
  buffer A{a};
  buffer B{b};

  Q.submit([&](handler &h) {
      accessor accA(A, h, read_only);
      accessor accB(B, h, write_only);
      h.parallel_for( // computeB
          N, [=](id<1> i) {
          accB[i] = accA[i] + 1;
          });
      });

  Q.submit([&](handler &h) {
      // WAR of buffer A
      accessor accA(A, h, write_only);
      h.parallel_for( // rewriteA
          N, [=](id<1> i) {
          accA[i] = 21 + 21;
          });
      });

  Q.submit([&](handler &h) {
      // WAW of buffer B
      accessor accB(B, h, write_only);
      h.parallel_for( // rewriteB
          N, [=](id<1> i) {
          accB[i] = 30 + 12;
          });
      });

  host_accessor host_accA(A, read_only);
  host_accessor host_accB(B, read_only);
  for (int i = 0; i < N; i++) {
    std::cout << host_accA[i] << " " << host_accB[i] << " ";
  }
  std::cout << "\n";
  return 0;
}
