// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cstdio>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t N = 16;
  constexpr size_t B = 4;

  queue q;
  int* data = malloc_shared<int>(N, q);
  int* sum = malloc_shared<int>(1, q);
  std::iota(data, data + N, 1);
  *sum = 0;

  // BEGIN CODE SNIP
  q.parallel_for(nd_range<1>{N, B}, [=](nd_item<1> it) {
     int i = it.get_global_id(0);
     auto grp = it.get_group();
     int group_sum =
         reduce_over_group(grp, data[i+1], plus<>()); // BUG: i => i+1
     if (grp.leader()) {
       atomic_ref<int, memory_order::relaxed,
                  memory_scope::system,
                  access::address_space::global_space>(
           *sum) += group_sum;
     }
   }).wait();
  // END CODE SNIP

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, q);
  free(data, q);
  return (passed) ? 0 : 1;
}
