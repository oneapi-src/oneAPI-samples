//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// The program solves the linear equation Ax=b, where matrix A is a
// n x n sparse matrix with diagonals [1 1 4 1 1],
// vector b is set such that the solution is a [1 1 ... 1]^T.
// The linear system is solved via Jacobi iteration.

// Each call of the kernel computes one iteration of the Jacobi method.

#include <CL/sycl.hpp>
#include <iostream>
// Location of file: <oneapi-root>/dev-utilities/<version>/include
#include "dpc_common.hpp"
#include "selector.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
  constexpr size_t n = 64;
  int b[n];
  float x_k[n];
  float x_k1[n];

  // Initialize the input.
  for (int i = 0; i < n; i++) {
    b[i] = 8;
    x_k[i] = 0;
  }

  b[0] = 6;
  b[1] = 7;
  b[n - 2] = 7;
  b[n - 1] = 6;

  try {
    CustomSelector selector(GetDeviceType(argc, argv));
    queue q(selector, dpc_common::exception_handler);
    cout << "[SYCL] Using device: ["
         << q.get_device().get_info<info::device::name>() << "] from ["
         << q.get_device().get_platform().get_info<info::platform::name>()
         << "]\n";

    // Create 1D buffers for b vector and iteration vectors.
    buffer buffer_b(b, range{n});
    buffer buffer_x_k(x_k, range{n});
    buffer buffer_x_k1(x_k1, range{n});

    for (int k = 0; k < 15000; k++) {
      // k-th iteration of Jacobi.
      q.submit([&](auto &h) {
        auto acc_b = buffer_b.get_access<access::mode::read>(h);
        auto acc_x_k = buffer_x_k.get_access<access::mode::read>(h);
        auto acc_x_k1 = buffer_x_k1.get_access<access::mode::write>(h);

        // kernel-start
        h.parallel_for(range{n}, [=](id<1> index) {
          int i = index[0];
          acc_x_k1[i] = acc_b[i] / 4 - (acc_x_k[i - 2] + acc_x_k[i - 1] +
                                        acc_x_k[i + 1] + acc_x_k[i + 2]) /
                                           4;
        });
        // kernel-end
      });
      q.wait_and_throw();
      for (int i = 0; i < n; i++)
        x_k[i] = x_k1[i];
    }
  } catch (sycl::exception const &e) {
    cout << "fail; synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }

  bool correct = true;
  // Verify the output, we expect a vector whose components are close to 1.0.
  for (int i = 0; i < n; i++) {
    if ((x_k1[i] - 1.0f) * (x_k1[i] - 1.0f) > 1e-6)
      correct = false;
  }

  if (correct)
    cout << "success; result is correct.\n";
  else
    cout << "fail; components of x_k are not close to 1.0\n";

  return 0;
}
