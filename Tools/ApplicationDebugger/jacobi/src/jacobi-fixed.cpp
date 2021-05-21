//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// The program solves the linear equation Ax=b, where matrix A is a
// n x n sparse matrix with diagonals [1 1 5 1 1],
// vector b is set such that the solution is [1 1 ... 1]^T.
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
  constexpr size_t number_of_iterations = 100;
  constexpr float tolerance = 1e-6;

  int b[n];
  float x_k[n];
  float x_k1[n];

  // Initialize the input.
  for (int i = 0; i < n; i++) {
    b[i] = 9;
    x_k[i] = 0;
  }

  b[0] = 7;
  b[1] = 8;
  b[n - 2] = 8;
  b[n - 1] = 7;

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

    for (int k = 0; k < number_of_iterations; k++) {
      // k-th iteration of Jacobi.
      q.submit([&](auto &h) {
        accessor acc_b (buffer_b, h, read_only);
        accessor acc_x_k (buffer_x_k, h, read_only);
        accessor acc_x_k1 (buffer_x_k1, h, write_only);

        // kernel-start
        h.parallel_for(range{n}, [=](id<1> index) {
          int i = index[0];
          // Bug 1: out-of-bounds access.
          // float sum = acc_x_k[i - 2] + acc_x_k[i - 1] + acc_x_k[i + 1] + acc_x_k[i + 2];
          //
          // Bug 2: b is an integer, so the division gives you 0.
          // acc_x_k1[i] = acc_b[i] / 4 - sum / 4;

          /* Fix bug 1. */
          float x_k1 = acc_b[i];
          if (i > 1)
            x_k1 -= acc_x_k[i - 2];
          if (i > 0)
            x_k1 -= acc_x_k[i - 1];
          if (i < n - 1)
            x_k1 -= acc_x_k[i + 1];
          if (i < n - 2)
            x_k1 -= acc_x_k[i + 2];

          /* Fix bug 2. */
          x_k1 *= 0.2;
          acc_x_k1[index] = x_k1;
        });
        // kernel-end
      });
      // Bug 3: the host buffers are not yet updated at this moment.
      // Note, with this code, you get correct results on CPU.
      // q.wait_and_throw();
      // for (int i = 0; i < n; i++)
      //   x_k[i] = x_k1[i];

      // Fix bug 3.
      q.submit([&](auto &h) {
        accessor acc_x_k (buffer_x_k, h, write_only);
        accessor acc_x_k1 (buffer_x_k1, h, read_only);

        h.parallel_for(range{n},
                       [=](id<1> index) { acc_x_k[index] = acc_x_k1[index]; });
      });
    }
  } catch (sycl::exception const &e) {
    cout << "fail; synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }

  bool correct = true;
  // Verify the output, we expect a vector whose components are close to 1.0.
  for (int i = 0; i < n; i++) {
    if ((x_k[i] - 1.0f) * (x_k[i] - 1.0f) > tolerance)
      correct = false;
  }

  if (correct)
    cout << "success; result is correct.\n";
  else
    cout << "fail; components of x_k are not close to 1.0\n";

  return 0;
}
