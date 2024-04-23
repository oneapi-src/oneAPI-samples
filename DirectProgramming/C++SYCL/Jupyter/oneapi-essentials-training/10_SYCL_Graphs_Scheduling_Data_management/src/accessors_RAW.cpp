//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <sycl/sycl.hpp>
#include <array>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int,N> a, b, c;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = c[i] = 0;
  }

  queue Q;

  //Create Buffers
  buffer A{a};
  buffer B{b};
  buffer C{c};

  Q.submit([&](handler &h) {
      accessor accA(A, h, read_only);
      accessor accB(B, h, write_only);
      h.parallel_for( // computeB
        N,
        [=](id<1> i) { accB[i] = accA[i] + 1; });
    });

  Q.submit([&](handler &h) {
      accessor accA(A, h, read_only);
      h.parallel_for( // readA
        N,
        [=](id<1> i) {
          // Useful only as an example
          int data = accA[i];
        });
    });

  Q.submit([&](handler &h) {
      // RAW of buffer B
      accessor accB(B, h, read_only);
      accessor accC(C, h, write_only);
      h.parallel_for( // computeC
        N,
        [=](id<1> i) { accC[i] = accB[i] + 2; });
    });

  // read C on host
  host_accessor host_accC(C, read_only);
  for (int i = 0; i < N; i++) {
    std::cout << host_accC[i] << " ";
  }
  std::cout << "\n";
  return 0;
}
