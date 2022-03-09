//=========================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
//=========================================================

#include <CL/sycl.hpp>
#include <chrono>

#include "jacobi.h"

using Time = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;
using float_ms = std::chrono::duration<float, ms::period>;

using namespace sycl;

// Run the Jacobi method for A*x = b on Device.
extern double JacobiMethodGpu(const float *A, const double *b,
                              const float conv_threshold, const int max_iter,
                              double *x, double *x_new, queue stream);

/* creates N_ROWS x N_ROWS matrix A with N_ROWS+1 on the diagonal and 1
elsewhere. The elements of the right hand side b all equal 2*n, hence the
exact solution x to A*x = b is a vector of ones. */
void createLinearSystem(float *A, double *b);

// Run the Jacobi method for A*x = b on Host.
void JacobiMethodCPU(float *A, double *b, float conv_threshold, int max_iter,
                     int *numit, double *x);

int main(int argc, char **argv) {
  // Host variable declaration and allocation
  double *b = NULL;
  float *A = NULL;

  queue q{default_selector(), property::queue::in_order()};

  b = (double *)malloc(N_ROWS * sizeof(double));
  memset(b, 0, N_ROWS * sizeof(double));

  A = (float *)malloc(N_ROWS * N_ROWS * sizeof(float));
  memset(A, 0, N_ROWS * N_ROWS * sizeof(float));

  createLinearSystem(A, b);
  double *x = NULL;

  // start with array of all zeroes
  x = (double *)calloc(N_ROWS, sizeof(double));

  float conv_threshold = 1.0e-2;
  int max_iter = 4 * N_ROWS * N_ROWS;
  int cnt = 0;

  // start Host Timer
  auto startHostTime = Time::now();
  JacobiMethodCPU(A, b, conv_threshold, max_iter, &cnt, x);

  double sum = 0.0;

  // Compute error
  for (int i = 0; i < N_ROWS; i++) {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }

  // stop Host timer
  auto stopHostTime = Time::now();

  printf("\nSerial Implementation : \n");
  printf("Iterations : %d\n", cnt);
  printf("Error : %.3e\n", sum);

  auto Host_duration =
      std::chrono::duration_cast<float_ms>(stopHostTime - startHostTime)
          .count();
  printf("Processing time : %f (ms)\n", Host_duration);

  // Device variable allocation and declaration
  float *d_A;
  double *d_b, *d_x, *d_x_new;

  d_b = malloc_device<double>(N_ROWS, q);
  d_A = malloc_device<float>(N_ROWS * N_ROWS, q);
  d_x = malloc_device<double>(N_ROWS, q);
  d_x_new = malloc_device<double>(N_ROWS, q);

  q.memset(d_x, 0, sizeof(double) * N_ROWS);
  q.memset(d_x_new, 0, sizeof(double) * N_ROWS);

  // Copy from Host variables to Device variables
  q.memcpy(d_A, A, N_ROWS * N_ROWS * sizeof(float));
  q.memcpy(d_b, b, N_ROWS * sizeof(double));

  // wait for the memcpy to complete
  q.wait();

  std::cout << "\nRunning on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  // start Device Timer
  auto startDeviceTime = Time::now();

  double sumDevice = 0.0;
  sumDevice =
      JacobiMethodGpu(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, q);

  // stop Device Timer
  auto stopDeviceTime = Time::now();

  auto Device_duration =
      std::chrono::duration_cast<float_ms>(stopDeviceTime - startDeviceTime)
          .count();
  printf("Processing time : %f (ms)\n", Device_duration);

  // Free up allocated memory
  free(d_b, q);
  free(d_A, q);
  free(d_x, q);
  free(d_x_new, q);
  free(A);
  free(b);

  printf("JacobiSYCL %s\n",
         (fabs(sum - sumDevice) < conv_threshold) ? "PASSED" : "FAILED");

  return (fabs(sum - sumDevice) < conv_threshold) ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Fill the arrays
void createLinearSystem(float *A, double *b) {
  int i, j;
  for (i = 0; i < N_ROWS; i++) {
    b[i] = 2.0 * N_ROWS;
    for (j = 0; j < N_ROWS; j++) A[i * N_ROWS + j] = 1.0;
    A[i * N_ROWS + i] = N_ROWS + 1.0;
  }
}

// Jacobi method for serial computation
void JacobiMethodCPU(float *A, double *b, float conv_threshold, int max_iter,
                     int *num_iter, double *x) {
  double *x_new;
  x_new = (double *)calloc(N_ROWS, sizeof(double));
  int k;

  for (k = 0; k < max_iter; k++) {
    double sum = 0.0;
    for (int i = 0; i < N_ROWS; i++) {
      double temp_dx = b[i];
      for (int j = 0; j < N_ROWS; j++) temp_dx -= A[i * N_ROWS + j] * x[j];
      temp_dx /= A[i * N_ROWS + i];
      x_new[i] += temp_dx;
      sum += fabs(temp_dx);
    }

    for (int i = 0; i < N_ROWS; i++) x[i] = x_new[i];

    if (sum <= conv_threshold) break;
  }
  *num_iter = k + 1;

  free(x_new);
}
