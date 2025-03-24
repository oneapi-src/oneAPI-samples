// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>

constexpr size_t N = 64;

int main() {
  sycl::queue Q;
  auto *data = new int[N];
  auto *array = sycl::malloc_device<long long>(N, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();

  sycl::free(data, Q);
  return 0;
}
