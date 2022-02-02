// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q{property::queue::in_order()};

  int *data = malloc_shared<int>(N, Q);

  Q.parallel_for(N, [=](id<1> i) { data[i] = 1; });

  Q.single_task([=]() {
      for (int i = 1; i < N; i++)
        data[0] += data[i];
    });
  Q.wait();

  assert(data[0] == N);
  return 0;
}
