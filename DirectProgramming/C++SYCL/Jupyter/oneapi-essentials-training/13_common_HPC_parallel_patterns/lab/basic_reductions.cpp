//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t N = 16;

  queue q;
  int* data = malloc_shared<int>(N, q);
  int* sum = malloc_shared<int>(1, q);
  std::iota(data, data + N, 1);
  *sum = 0;

  q.submit([&](handler& h) {
     // BEGIN CODE SNIP
     h.parallel_for(
         range<1>{N}, reduction(sum, plus<>()),
         [=](id<1> i, auto& sum) { sum += data[i]; });
     // END CODE SNIP
   }).wait();

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, q);
  free(data, q);
  return (passed) ? 0 : 1;
}
