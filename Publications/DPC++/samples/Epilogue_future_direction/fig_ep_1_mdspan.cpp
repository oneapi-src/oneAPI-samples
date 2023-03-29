// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <experimental/mdspan>

using namespace sycl;
namespace stdex = std::experimental;

int main() {

  queue Q;
  constexpr int N = 4;
  constexpr int M = 2;
  int* data = malloc_shared<int>(N * M, Q);

  stdex::mdspan<int, N, M> view{data};
  Q.parallel_for(range<2>{N, M}, [=](id<2> idx) {
    int i = idx[0];
    int j = idx[1];
    view(i, j) = i * M + j;
  }).wait();

  bool passed = true;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (data[i * M + j] != i * M + j) {
        passed = false;
      }
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(data, Q);
  return (passed) ? 0 : 1;
}
