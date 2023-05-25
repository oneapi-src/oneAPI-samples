//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>
#include <stdio.h>

constexpr int N = 100;
constexpr int iters = 100;

int main() {

  int AData[N];
  int BData[N];
  int CData[N];

  sycl::queue Q;
  sycl::buffer<int> CBuf(&CData[0], N);

  {
    // Create 2 buffers, each holding N integers
    sycl::buffer<int> ABuf(&AData[0], N);
    sycl::buffer<int> BBuf(&BData[0], N);

    // Kernel1
    Q.submit([&](auto &h) {
      // Create device accessors.
      // The property no_init lets the runtime know that the
      // previous contents of the buffer can be discarded.
      sycl::accessor aA(ABuf, h, sycl::write_only, sycl::no_init);
      sycl::accessor aB(BBuf, h, sycl::write_only, sycl::no_init);
      sycl::accessor aC(CBuf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(N, [=](auto i) {
        aA[i] = 11;
        aB[i] = 22;
        aC[i] = 0;
      });
    });
  }

  for (int it = 0; it < iters; it++) {
    // Kernel2
    Q.submit([&](auto &h) {
      // Create device accessors
      sycl::accessor aC(CBuf, h);
      h.parallel_for(N, [=](auto i) { aC[i] += AData[i] + BData[i]; });
    });
  }

  sycl::host_accessor h_acc(CBuf);
  for (int i = 0; i < N; i++) {
    printf("%d\n", h_acc[i]);
  }

  return 0;
}
// Snippet end
