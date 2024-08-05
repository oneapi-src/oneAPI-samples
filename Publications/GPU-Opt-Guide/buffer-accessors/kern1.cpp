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

  // Kernel1
  {
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
      sycl::accessor aC(CBuf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(N, [=](auto i) {
        aA[i] = 11;
        aB[i] = 22;
        aC[i] = 0;
      });
    });
  } // end Kernel1

  // Kernel2
  {
    // Create 3 buffers, each holding N integers
    sycl::buffer<int> ABuf(&AData[0], N);
    sycl::buffer<int> BBuf(&BData[0], N);
    sycl::buffer<int> CBuf(&CData[0], N);

    Q.submit([&](auto &h) {
      // Create device accessors
      sycl::accessor aA(ABuf, h, sycl::read_only);
      sycl::accessor aB(BBuf, h, sycl::read_only);
      sycl::accessor aC(CBuf, h);
      h.parallel_for(N, [=](auto i) { aC[i] += aA[i] + aB[i]; });
    });
  } // end Kernel2

  // Buffers are destroyed and so CData is updated and can be accessed
  for (int i = 0; i < N; i++) {
    printf("%d\n", CData[i]);
  }

  return 0;
}
// Snippet end
