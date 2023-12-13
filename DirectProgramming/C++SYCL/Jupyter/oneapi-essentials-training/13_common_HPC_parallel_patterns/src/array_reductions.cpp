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
  constexpr size_t B = 4;

  queue q;
  int* data = malloc_shared<int>(N, q);
  int* histogram = malloc_shared<int>(B, q);
  std::iota(data, data + N, 1);
  std::fill(histogram, histogram + B, 0);

  q.submit([&](handler& h) {
     // BEGIN CODE SNIP
     h.parallel_for(
         range{N},
         reduction(span<int, 16>(histogram, 16), plus<>()),
         [=](id<1> i, auto& histogram) {
           histogram[i % B]++;
         });
     // END CODE SNIP
   }).wait();

  bool passed = true;
  std::cout << "Histogram:" << std::endl;
  for (int b = 0; b < B; ++b) {
    std::cout << "bin[" << b << "]: " << histogram[b]
              << std::endl;
    passed &= (histogram[b] == N / B);
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(histogram, q);
  free(data, q);
  return (passed) ? 0 : 1;
}
