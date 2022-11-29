//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Compile:
// dpcpp -D{HOST|CPU|GPU} -std=c++17 -fsycl external_rand.cpp -o external_rand

#include <sycl/sycl.hpp>
#include <iostream>
#include <random>

#define N 5

extern SYCL_EXTERNAL int rand(void);

int main(void) {
#if defined HOST
  sycl::queue Q(sycl::host_selector{});
#elif defined CPU
  sycl::queue Q(sycl::cpu_selector{});
#elif defined GPU
  sycl::queue Q(sycl::gpu_selector{});
#endif

  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Attempt to use rand() inside a DPC++ kernel
  auto test1 = sycl::malloc_shared<float>(N, Q.get_device(), Q.get_context());

  srand((unsigned)time(NULL));
  Q.parallel_for(N, [=](auto idx) {
     test1[idx] = (float)rand() / (float)RAND_MAX;
   }).wait();

  // Show the random number sequence
  for (int i = 0; i < N; i++)
    std::cout << test1[i] << std::endl;

  // Cleanup
  sycl::free(test1, Q.get_context());
}
