// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <cstdio>
#include <numeric>

using namespace sycl;
using namespace sycl::ONEAPI;

int main() {

  using memory_order = sycl::ONEAPI::memory_order;
  using memory_scope = sycl::ONEAPI::memory_scope;

  constexpr size_t N = 16;
  constexpr size_t B = 4;

  queue Q;
  int* data = malloc_shared<int>(N, Q);
  int* sum = malloc_shared<int>(1, Q);
  std::iota(data, data + N, 1);
  *sum = 0;

  Q.parallel_for(nd_range<1>{N, B}, [=](nd_item<1> it) {
     int i = it.get_global_id(0);
     int group_sum = reduce(it.get_group(), data[i], plus<>());
     if (it.get_local_id(0) == 0) {
       atomic_ref<
           int,
           memory_order::relaxed,
           memory_scope::system,
           access::address_space::global_space>(*sum) += group_sum;
     }
   }).wait();

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
