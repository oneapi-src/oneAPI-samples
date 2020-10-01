//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
constexpr int N = 16;
using namespace sycl;

// Buffer creation happens within a separate function scope.
void dpcpp_code(std::vector<int> &v, queue &q) {
  auto R = range<1>(N);
  buffer<int, 1> buf(v.data(), R);
  q.submit([&](handler &h) {
    auto a = buf.get_access<access::mode::read_write>(h);
    h.parallel_for(R, [=](id<1> i) { a[i] -= 2; });
  });
}
int main() {
  std::vector<int> v(N, 10);
  queue q;
  dpcpp_code(v, q);
  // When execution advances beyond this function scope, buffer destructor is
  // invoked which relinquishes the ownership of data and copies back the data to
  // the host memory.
  for (int i = 0; i < N; i++) std::cout << v[i] << "\n";
  return 0;
}
