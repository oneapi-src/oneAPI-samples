//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  int *data = malloc_shared<int>(N, Q);

  auto e = Q.parallel_for(N, [=](id<1> i) { data[i] = 1; });

  Q.submit([&](handler &h) {
      h.depends_on(e);
      h.single_task([=]() {
          for (int i = 1; i < N; i++)
            data[0] += data[i];
        });
    });
  Q.wait();

  assert(data[0] == N);
  return 0;
}
