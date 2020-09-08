//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include "mkl.h"
#include "mkl_blas_sycl.hpp"

using namespace std;
using namespace cl::sycl;

// Matrix size constants
#define SIZE 4800  // Must be a multiple of 8.
#define M SIZE / 8
#define N SIZE / 4
#define P SIZE / 2

/**
 * Perform the matrix multiplication on host to verify results from mkl.
 */
int VerifyResult(double *c_back);

int main() {
  //
  // Initialize data for Gemm
  //
  // C = alpha * op(A) * op(B)  + beta * C
  //
  mkl::transpose transA = mkl::transpose::nontrans;
  mkl::transpose transB = mkl::transpose::nontrans;

  // matrix data sizes
  int m = M;
  int n = P;
  int k = N;

  // leading dimensions of data
  int ldA = m;
  int ldB = k;
  int ldC = m;

  // set scalar fp values
  double alpha = 1.0;
  double beta = 0.0;

  // 1D arrays on host side
  double *A;
  double *B;
  double *C;

  A = new double[M * N];
  B = new double[N * P];
  C = new double[M * P];

  // prepare matrix data with column-major style
  int i, j;
  // A(M, N) is a matrix whose values are column number plus one
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++) A[i * M + j] = i + 1.0;

  // B(N, P) is matrix whose values are row number plus one
  for (i = 0; i < P; i++)
    for (j = 0; j < N; j++) B[i * N + j] = j + 1.0;

  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
       << ") * b(" << N << "," << P << ")" << std::endl;

  //
  // Execute Gemm
  //
  auto asyncHandler = [&](cl::sycl::exception_list eL) {
    for (auto &e : eL) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << "fail" << std::endl;
        // std::terminate() will exit the process, return non-zero, and output a
        // message to the user about the exception
        std::terminate();
      }
    }
  };

  try {
    // Initializing the devices queue with the default selector
    // The device queue is used to enqueue the kernels and encapsulates
    // all the states needed for execution
    default_selector device_selector;
    queue device_queue(device_selector, asyncHandler);

    std::cout << "Device: "
              << device_queue.get_device().get_info<info::device::name>()
              << std::endl;

    // Creating 1D buffers for matrices which are bound to host memory array
    buffer<double, 1> a{A, range<1>{M * N}};
    buffer<double, 1> b{B, range<1>{N * P}};
    buffer<double, 1> c{C, range<1>{M * P}};

    mkl::blas::gemm(device_queue, transA, transB, m, n, k, alpha, a, ldA, b,
                    ldB, beta, c, ldC);
  } catch (cl::sycl::exception const &e) {
    std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
  }

  int result;
  result = VerifyResult(C);

  delete[] A;
  delete[] B;
  delete[] C;

  return result;
}

bool ValueSame(double a, double b) {
  return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}

int VerifyResult(double *c_back) {
  // Check that the results are correct by comparing with host computing
  int i, j, k;

  // 2D arrays on host side
  double(*a_host)[N];
  double(*b_host)[P];
  double(*c_host)[P];

  a_host = new double[M][N];
  b_host = new double[N][P];
  c_host = new double[M][P];

  // a_host is a matrix whose values are column number plus one
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a_host[i][j] = j + 1.0;

  // b_host is a matrix whose values are row number plus one
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b_host[i][j] = i + 1.0;

  // c_host is initialized to zero
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = 0;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      for (j = 0; j < P; j++) {
        c_host[i][j] += a_host[i][k] * b_host[k][j];
      }
    }
  }

  bool MismatchFound = false;

  // compare host side results with the result buffer from device side: print
  // fail data 5 times only.
  int printf_count = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i + j * M], c_host[i][j])) {
        cout << "fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << " , but got: " << c_back[i + j * M] << std::endl;
        MismatchFound = true;
        printf_count++;
        if (printf_count >= 5) break;
      }
    }
    if (printf_count >= 5) break;
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  if (!MismatchFound) {
    cout << "success - The results are correct!" << std::endl;
    return 0;
  } else {
    cout << "fail - The results mis-match!" << std::endl;
    return -1;
  }
}
