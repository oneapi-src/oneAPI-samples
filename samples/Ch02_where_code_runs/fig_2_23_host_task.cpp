// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;
constexpr int N = 4;

int main() {

  queue Q{gpu_selector_v};
  int *A = malloc_shared<int>(N, Q);

  std::cout << "Selected device: " <<
    Q.get_device().get_info<info::device::name>() << "\n";

  // Initialize values in the buffer
  auto eA = Q.submit([&](handler& h) {
      h.parallel_for(N , [=](auto& idx) {
	  A[idx] = idx;
	  });
      });

  // Use a host task to output values on the host as part of
  // task graph.  depends_on is used to define a dependence on
  // previous device code having completed.
  // Here the host task is defined as a lambda expression.
  Q.submit([&](handler& h) {
      h.depends_on( eA );
      h.host_task( [&](){
	  for (int i = 0; i < N; i++)
	  std::cout << "host_task @ " << i << " = " << A[i] << "\n";
	  });
      });

  // Wait for work to be completed in the queue before accessing
  // the shared data in the host program.
  Q.wait();

  for (int i = 0; i < N; i++)
    std::cout << "main @ " << i << " = " << A[i] << "\n";

  free(A, Q);

  return 0;
}
