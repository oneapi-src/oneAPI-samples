// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;

int main() {

  constexpr size_t N = 16;
  constexpr size_t M = 16;

  queue Q;
  float* data = malloc_shared<float>(N * M, Q);
  float* max = malloc_shared<float>(1, Q);
  std::iota(data, data + N * M, 1);
  *max = 0;

  Q.submit([&](handler& h) {
     h.parallel_for_work_group(N, reduction(max, maximum<>()),
         [=](group<1> g, auto& max) {
           float sum = 0.0f;
           g.parallel_for_work_item(M, reduction(sum, plus<>()),
           [=](h_item<1> it, auto& sum) {
             sum += data[it.get_global_id()];
           });
           max.combine(sum);
         });
   }).wait();

  std::cout << "max sum = " << *max << "\n";
  float gold = 0;
  for (int g = 0; g < N; ++g) {
    float sum = 0.0f;
    for (int i = 0; i < M; ++i) {
      sum += data[g * M + i];
    }
    gold = std::max(gold, sum);
  }
  bool passed = (std::abs(*max - gold) < 1.0E-06);
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(max, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
