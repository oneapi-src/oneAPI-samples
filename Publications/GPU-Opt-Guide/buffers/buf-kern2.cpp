//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <stdio.h>

constexpr int N = 25;
constexpr int STEPS = 100000;

int main() {

  int AData[N];
  int BData[N];
  int CData[N];

  sycl::queue Q;

  // Create 3 buffers, each holding N integers
  sycl::buffer<int> ABuf(&AData[0], N);
  sycl::buffer<int> BBuf(&BData[0], N);
  sycl::buffer<int> CBuf(&CData[0], N);

  Q.submit([&](auto &h) {
    // Create device accessors.
    // The property no_init lets the runtime know that the
    // previous contents of the buffer can be discarded.
    sycl::accessor aA(ABuf, h, sycl::write_only, sycl::no_init);
    sycl::accessor aB(BBuf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(N, [=](auto i) {
      aA[i] = 10;
      aB[i] = 20;
    });
  });

  for (int j = 0; j < STEPS; j++) {
    Q.submit([&](auto &h) {
      // Create device accessors.
      sycl::accessor aA(ABuf, h);
      sycl::accessor aB(BBuf, h);
      sycl::accessor aC(CBuf, h);
      h.parallel_for(N, [=](auto i) {
        aC[i] = (aA[i] < aB[i]) ? -1 : 1;
        aA[i] += aC[i];
        aB[i] -= aC[i];
      });
    });
  } // end for

  // Create host accessors.
  const sycl::host_accessor haA(ABuf);
  const sycl::host_accessor haB(BBuf);
  printf("%d %d\n", haA[N / 2], haB[N / 2]);

  return 0;
}
