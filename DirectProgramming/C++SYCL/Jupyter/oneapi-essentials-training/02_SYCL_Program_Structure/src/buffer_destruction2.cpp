//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
constexpr int N = 16;
using namespace sycl;

// Buffer creation happens within a separate function scope.
void SYCL_code(std::vector<int> &v, queue &q) {
  auto R = range<1>(N);
  buffer buf(v);
  q.submit([&](handler &h) {
    accessor a(buf,h);
    h.parallel_for(R, [=](auto i) { a[i] -= 2; });
  });
}
int main() {
  std::vector<int> v(N, 10);
  queue q;
  SYCL_code(v, q);
  // When execution advances beyond this function scope, buffer destructor is
  // invoked which relinquishes the ownership of data and copies back the data to
  // the host memory.
  for (int i = 0; i < N; i++) std::cout << v[i] << "\n";
  return 0;
}
