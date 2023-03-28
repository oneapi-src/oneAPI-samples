// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <iostream>

using namespace sycl;

int main() {
  queue Q;

  const size_t N = 32;
  const size_t M = 4;
  std::vector<int> data(N);
  std::fill(data.begin(), data.end(), 0);

  {
    buffer buf(data);

    Q.submit([&](handler& h) {
      accessor acc{buf, h};
      h.parallel_for(N, [=](id<1> i) {
        int j = i % M;
        atomic_ref<int, memory_order::relaxed, memory_scope::system,
                   access::address_space::global_space> atomic_acc(acc[j]);
        atomic_acc += 1;
      });
    });
  }

  for (int i = 0; i < N; ++i) {
    std::cout << "data [" << i << "] = " << data[i] << "\n";
  }

  bool passed = true;
  int* gold = (int*) malloc(N * sizeof(int));
  std::fill(gold, gold + N, 0);
  for (int i = 0; i < N; ++i) {
    int j = i % M;
    gold[j] += 1;
  }
  for (int i = 0; i < N; ++i) {
    if (data[i] != gold[i]) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS\n" : "FAILURE\n");
  free(gold);
  return (passed) ? 0 : 1;
}
