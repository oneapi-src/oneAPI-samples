//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>
#include <stdio.h>

constexpr int N = 100;

int main() {

  int AData[N];
  int BData[N];
  int CData[N];

  sycl::queue Q;

  {
    // Create 3 buffers, each holding N integers
    sycl::buffer<int> ABuf(&AData[0], N);
    sycl::buffer<int> BBuf(&BData[0], N);
    sycl::buffer<int> CBuf(&CData[0], N);

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

    // Kernel2
    Q.submit([&](auto &h) {
      // Create device accessors
      sycl::accessor aA(ABuf, h, sycl::read_only);
      sycl::accessor aB(BBuf, h, sycl::read_only);
      sycl::accessor aC(CBuf, h);
      h.parallel_for(N, [=](auto i) { aC[i] += aA[i] + aB[i]; });
    });
  }
  // Since the buffers are going out of scope, they will have to be
  // copied back from device to host and this will require a wait for
  // all the kernels to finish and so no explicit wait is needed
  for (int i = 0; i < N; i++) {
    printf("%d\n", CData[i]);
  }

  return 0;
}
// Snippet end
