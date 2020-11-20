//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int N = 16;
  auto R = range<1>(N);
  std::vector<int> v(N, 10);
  queue q;
  // Buffer takes ownership of the data stored in vector.  
  buffer buf(v);
  q.submit([&](handler& h) {
    accessor a(buf,h);
    h.parallel_for(R, [=](auto i) { a[i] -= 2; });
  });
  // Creating host accessor is a blocking call and will only return after all
  // enqueued DPC++ kernels that modify the same buffer in any queue completes
  // execution and the data is available to the host via this host accessor.
  host_accessor b(buf,read_only);
  for (int i = 0; i < N; i++) std::cout << b[i] << " ";
  return 0;
}
