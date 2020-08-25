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
  // Buffer takes ownership of the data stored in vector.__
  buffer<int, 1> buf(v.data(), R);
  q.submit([&](handler& h) {
    auto a = buf.get_access<access::mode::read_write>(h);
    h.parallel_for(R, [=](id<1> i) { a[i] -= 2; });
  });
  // Creating host accessor is a blocking call and will only return after all
  // enqueued DPC++ kernels that modify the same buffer in any queue completes
  // execution and the data is available to the host via this host accessor.
  auto b = buf.get_access<access::mode::read>();
  for (int i = 0; i < N; i++) std::cout << v[i] << "\n";
  return 0;
}
