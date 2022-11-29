// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;

int main() {
  constexpr size_t N = 16;

  queue Q;
  int* data = malloc_shared<int>(N, Q);
  int* sum = malloc_shared<int>(1, Q);
  std::iota(data, data + N, 1);
  *sum = 0;

  Q.parallel_for(N, [=](id<1> i) {
     atomic_ref<
         int,
         memory_order::relaxed,
         memory_scope::system,
         access::address_space::global_space>(*sum) += data[i];
   }).wait();

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
