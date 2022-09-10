//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// The program solves the linear equation Ax=b, where matrix A is a
// n x n sparse matrix with diagonals [1 1 5 1 1],
// vector b is set such that the solution is [1 1 ... 1]^T.
// The linear system is solved via Jacobi iteration.
// The algorithm converges, as the matrix A is strictly diagonally dominant.

#include <CL/sycl.hpp>
#include <iostream>
#include <cmath>
// Location of file: <oneapi-root>/dev-utilities/<version>/include
#include "dpc_common.hpp"
#include "selector.hpp"

using namespace std;
using namespace sycl;

// Helper structure to initialize and hold all our SYCL buffers.
// Note: no bugs here.
struct buffer_args {
  buffer<float, 1> b;
  buffer<float, 1> x_k;
  buffer<float, 1> x_k1;
  buffer<float, 1> l1_norm_x_k1;
  buffer<float, 1> abs_error;
  buffer_args(size_t n, float *b,
              float *x_k, float *x_k1,
              float *l1_norm_x_k1,
              float *abs_error):
    // Right hand side vector b;
    b(buffer(b, range{n})),
    // Iteration vectors x_k and x_k1;
    x_k(buffer(x_k, range{n})),
    x_k1(buffer(x_k1, range{n})),
    // Sum of absolute values of x_k1 elements.
    l1_norm_x_k1(buffer(l1_norm_x_k1, range{1})),
    // Absolute error.
    abs_error(buffer(abs_error, range{1}))
  {}
};

// Initialize right hand side vector b and the initial guess for x_k.
void initialize_input(float *b, float *x_k, size_t n);

// At each iteration the computation of the resulting vector x happens here.
void main_computation (queue &q, buffer_args &buffers, size_t n) {
  // Here, we compute the updated vector x_k1.
  q.submit([&](auto &h) {
    accessor acc_b(buffers.b, h, read_only);
    accessor acc_x_k(buffers.x_k, h, read_only);
    accessor acc_x_k1(buffers.x_k1, h, write_only);

    h.parallel_for(range{n}, [=](id<1> index) {
      // Current index.
      int i = index[0];

      // The vector x_k1 should be computed as:
      //     x_k1 = D^{-1}(b - (A - D)x_k),
      // where A is our matrix, D is its diagonal, b is right hand
      // side vector, and x_k is the result of the previous iteration.
      //
      // Matrices (A - D) and D are hardcoded as:
      //     (A - D) is a stencil matrix [1 1 0 1 1];
      //     D is a diagonal matrix with all elements equal to 5.

      float x_k1 = acc_b[i];

      // Non-diagonal elements of matrix A are all 1s, so to substract
      // i-th element of (A - D)x_k, we need to substract the sum of elements
      // of x_k with indices i - 2, i - 1, i + 1, i + 2. We do not substract
      // the i-th element, as it gets multiplied by 0 in (A - D)x_k.
      //
      // Fix bug 1: out-of-bounds access: all indices below can trigger
      // out-of-bounds access and thus garbage values will be read.
      // Fix it by adding checks that the index exists:
      if (i > 1)
        x_k1 -= acc_x_k[i - 2];
      if (i > 0)
        x_k1 -= acc_x_k[i - 1];
      if (i < n - 1)
        x_k1 -= acc_x_k[i + 1];
      if (i < n - 2)
        x_k1 -= acc_x_k[i + 2];

      // In our case the diagonal matrix has only 5s on the diagonal, so
      // division by 5 gives us its invert.
      x_k1 /= 5;

      // Save the value to the output buffer.
      acc_x_k1[index] = x_k1;
    });
  });
}

// Here we compute values which are used for relative error computation
// and copy the vector x_k1 over the vector x_k.
void prepare_for_next_iteration (queue &q, buffer_args &buffers,size_t n) {
  constexpr size_t l = 16;

  // To compute the relative error we need to prepare two values:
  // abs_error and l1_norm_x_k1.
  //
  // First, we need to compute the L1-norm of (x_k - x_k1). To do that,
  // we sum absolute values of its elements into abs_error buffer.
  // We use a reduction algorithm here.
  q.submit([&](auto &h) {
    accessor acc_abs_error(buffers.abs_error, h, read_write);
    accessor acc_x_k(buffers.x_k, h, read_only);
    accessor acc_x_k1(buffers.x_k1, h, read_only);

    h.parallel_for(nd_range<1>{n, l},
                   reduction(acc_abs_error, std::plus<>()),
                   [=](nd_item<1> index, auto& acc_abs_error) {
      auto gid = index.get_global_id();

      // Compute the sum.
      acc_abs_error += abs(acc_x_k[gid] - acc_x_k1[gid]);
    });
  });

  // Second, we need to compute the L1 norm of x_k1.
  // For that, we sum absolute values of all elements of vector x_k1 into
  // l1_norm_x_k1 buffer. Again we use the reduction algorithm here.
  q.submit([&](auto &h) {
    accessor acc_x_k1(buffers.x_k1, h, read_only);
    accessor acc_l1_norm_x_k1(buffers.l1_norm_x_k1, h, read_write);

    h.parallel_for(nd_range<1>{n, l},
                   reduction(acc_l1_norm_x_k1, std::plus<>()),
                   [=](nd_item<1> index, auto& acc_l1_norm_x_k1) {
      auto gid = index.get_global_id();
      // Compute the sum.
      acc_l1_norm_x_k1 += abs(acc_x_k1[gid]); // Bug 2 challenge: breakpoint here.
    });
  });

  // We copy the values of vector x_k1 to x_k in order to setup
  // the next iteration.
  q.submit([&](auto &h) {
    accessor acc_x_k(buffers.x_k, h, write_only);
    accessor acc_x_k1(buffers.x_k1, h, read_only);

    h.parallel_for(range{n}, [=](id<1> index) {
      auto gid = index[0];

      // Copy the vector x_k1 over x_k.
      acc_x_k[gid] = acc_x_k1[gid];
    });
  });
}

int main(int argc, char *argv[]) {
  // The size of the problem.
  // The matrix A is n x n matrix, the length of the vector x is n.
  constexpr size_t n = 64;

  // The maximum number of iterations the algorithm is going
  // to take.
  constexpr size_t max_number_of_iterations = 100;

  // We expect each element of vector x to be that close
  // to the analitycal solution.
  constexpr float tolerance = 1e-4;

  // The right hand side vector.
  float b[n];

  // We store an intermediate result after every iteration here.
  float x_k[n];

  // At each iteration we compute a new value of x here. We need
  // both buffers x_k and x_k1 due to the data dependency: to compute
  // one element at the iteration k + 1 we need several elements
  // from the iteration k.
  float x_k1[n];

  // We will compute the relative error at each iteration as:
  //
  //              ||x_k - x_k1||_1       abs_error
  // rel_error = ------------------- = --------------
  //                 ||x_k1||_1         l1_norm_x_k1

  // Absolute error, ||x_k - x_k1||_1, L1-norm of (x_k - x_k1).
  float abs_error = 0;
  // ||x_k1||_1, L1-norm of x_k1.
  float l1_norm_x_k1 = 0;
  // Relative error.
  float rel_error;

  // Initialize the input.
  // Note: the matrix A is hardcoded as a stencil matrix [1 1 5 1 1]
  // into the kernel.
  initialize_input(b, x_k, n);

  // Iteration counter.
  int k = 0;
  try {
    CustomSelector selector(GetDeviceType(argc, argv));
    queue q(selector, dpc_common::exception_handler);
    cout << "[SYCL] Using device: ["
         << q.get_device().get_info<info::device::name>() << "] from ["
         << q.get_device().get_platform().get_info<info::platform::name>()
         << "]\n";

    // Jacobi iteration begins.
    do {// k-th iteration of Jacobi.
      // Fix bug 2: we have to reset the error values at each iteration, otherwise
      // the relative error accumulates through iterations and does not fall
      // below the tolerance.
      abs_error = 0;
      l1_norm_x_k1 = 0;

      { // Fix bug 3: the host values of abs_error and l1_norm_x_k1
        // were not synchronised with their new values on device.
        // Open new scope for buffers. Once the scope is ended, the destructors
        // of buffers will write the data back from device to host.

        // Create SYCL buffers.
        buffer_args buffers {n, b, x_k, x_k1, &l1_norm_x_k1, &abs_error};

        main_computation(q, buffers, n);
        prepare_for_next_iteration(q, buffers, n);
      }

      // Compute relative error based on reduced values from this iteration.
      rel_error = abs_error / (l1_norm_x_k1 + 1e-32);

      if (abs_error < 0 || l1_norm_x_k1 < 0
          || (abs_error + l1_norm_x_k1) < 1e-32) {
        cout << "\nfail; Bug 3. Fix it on GPU. The relative error has invalid value "
             << "after iteration " << k << ".\n"
             << "Hint 1: inspect reduced error values. With the challenge scenario\n"
             << "    from bug 2 you can verify that reduction algorithms compute\n"
             << "    the correct values inside kernel on GPU. Take into account\n"
             << "    SIMD lanes: on GPU each thread processes several work items\n"
             << "    at once, so you need to modify your commands and update\n"
             << "    the convenience variable for each SIMD lane.\n"
             << "Hint 2: why don't we get the correct values at the host part of\n"
             << "    the application?\n";
        return 0;
      }

      // Periodically print out how the algorithm behaves.
      if (k % 20 == 0) {
        std::cout << "Iteration " << k << ", relative error = "
                  << rel_error << "\n";
      }

      k++;
    } while (rel_error > tolerance && k < max_number_of_iterations);
    // Jacobi iteration ends.

  } catch (sycl::exception const &e) {
    cout << "fail; synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }

  // Verify the output, we expect a vector whose components are close to 1.0.
  bool correct = true;
  for (int i = 0; i < n; i++) {
    if ((x_k[i] - 1.0f) * (x_k[i] - 1.0f) > tolerance)
      correct = false;
  }

  if (correct)
    cout << "\nsuccess; all elements of the resulting vector are close to 1.0.\n";
  else {
    cout << "\nfail; Bug 1. Fix this on CPU: components of x_k are not close to 1.0.\n"
         << "Hint: figure out which elements are farthest from 1.0.\n";
    return 0;
  }

  // Check whether the algorithm converged.
  if (k < max_number_of_iterations) {
    cout << "success; the relative error (" << rel_error
         << ") is below the desired tolerance "
         << tolerance <<" after " << k <<" iterations.\n\n";
  } else {
    cout << "\nfail; Bug 2. Fix this on CPU: the relative error (" << rel_error
         << ") is greater than\n"
         << "    the desired tolerance "
         << tolerance <<" after " << max_number_of_iterations
         << " iterations.\n"
         << "Hint: check the reduction results at several iterations.\n"
         << "Challenge: in the debugger you can simmulate the computation of a reduced\n"
         << "    value by putting a BP inside the corresponding kernel and defining\n"
         << "    a convenience variable. We will compute the reduced value at this\n"
         << "    convenience variable: at each BP hit we update it with a help of \"commands\"\n"
         << "    command. After the reduction kernel is finished, the convenience\n"
         << "    variable should contain the reduced value.\n"
         << "    See README for details.\n";
    return 0;
  }

  return 0;
}

// Note: no bugs here.
void initialize_input(float *b, float *x_k, size_t n) {
  constexpr float main_b = 9;

  // Vector b and the matrix A are hardcoded
  // such that the analytical solution of the system Ax=b is a vector
  // whose elements are 1s.
  for (int i = 0; i < n; i++)
    b[i] = main_b;

  // Boundary values of the vector b.
  b[0] = main_b - 2;
  b[1] = main_b - 1;
  b[n - 2] = main_b - 1;
  b[n - 1] = main_b -2;

  // Initial guess of x is b.
  for (int i = 0; i < n; i++)
    x_k[i] = b[i];
}
