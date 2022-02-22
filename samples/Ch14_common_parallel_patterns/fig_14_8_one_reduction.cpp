// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
//   dropped 'using namespace sycl::ONEAPI'
//   this allows reduction to use the sycl::reduction,
//   added sycl::ONEAPI:: to plus
// -------------------------------------------------------

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;

int main() {

  constexpr size_t N = 16;
  constexpr size_t B = 4;

  queue Q;
  int* data = malloc_shared<int>(N, Q);
  int* sum = malloc_shared<int>(1, Q);
  std::iota(data, data + N, 1);
  *sum = 0;

  Q.submit([&](handler& h) {
// BEGIN CODE SNIP
     h.parallel_for(
         nd_range<1>{N, B},
         reduction(sum, sycl::plus<>()),
         [=](nd_item<1> it, auto& sum) {
           int i = it.get_global_id(0);
           sum += data[i];
         });
// END CODE SNIP
   }).wait();

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
