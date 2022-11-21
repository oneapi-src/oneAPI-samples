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
 
  int *data1 = malloc_shared<int>(N, Q);
  int *data2 = malloc_shared<int>(N, Q);

  auto e1 = Q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });

  auto e2 = Q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });

  auto e3 = Q.parallel_for(range{N}, {e1, e2},
                           [=](id<1> i) { data1[i] += data2[i]; });

  Q.single_task(e3, [=]() {
      for (int i = 1; i < N; i++)
        data1[0] += data1[i];

      data1[0] /= 3;
    });
  Q.wait();

  assert(data1[0] == N);
  return 0;
}
