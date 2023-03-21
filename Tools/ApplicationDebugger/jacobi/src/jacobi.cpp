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

#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
// Location of file: <oneapi-root>/dev-utilities/<version>/include
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

// The size of the problem.
// The matrix A is n x n matrix, the length of the vector x is n.
constexpr size_t n = 64;

// The maximum number of iterations the algorithm is going
// to take.
constexpr size_t max_number_of_iterations = 100;

// We expect each element of vector x to be that close
// to the analitycal solution.
constexpr float tolerance = 1e-4;

// Helper structure to initialize and hold all our SYCL buffers.
// Note: no bugs here.
struct buffer_args {
  buffer<float, 1> b;
  buffer<float, 1> x_k;
  buffer<float, 1> x_k1;
  buffer<float, 1> l1_norm_x_k1;
  buffer<float, 1> abs_error;
  buffer_args(float *b, float *x_k, float *x_k1,
              float *l1_norm_x_k1, float *abs_error):
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

// Depending on whether FIXED is set, select fixed or bugged versions
// of computation methods.
#ifdef FIXED
#include "fixed.cpp"
#else
#include "bugged.cpp"
#endif // FIXED

// Initialize right hand side vector b and the initial guess for x_k.
void initialize_input(float *b, float *x_k);

int main(int argc, char *argv[]) {
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

  // Relative error.
  float rel_error;

  // Initialize the input.
  // Note: the matrix A is hardcoded as a stencil matrix [1 1 5 1 1]
  // into the kernel.
  initialize_input(b, x_k);

  // Iteration counter.
  int k;
  try {
    queue q(default_selector_v, dpc_common::exception_handler);
    cout << "[SYCL] Using device: ["
         << q.get_device().get_info<info::device::name>() << "] from ["
         << q.get_device().get_platform().get_info<info::platform::name>()
         << "]\n";

    k = iterate (q, b, x_k, x_k1, rel_error);
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
void initialize_input(float *b, float *x_k) {
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
  b[n - 1] = main_b - 2;

  // Initial guess of x is b.
  for (int i = 0; i < n; i++)
    x_k[i] = b[i];
}
