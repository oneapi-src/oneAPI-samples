//=========================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
//=========================================================

#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "jacobi.h"

// Run the Jacobi method for A*x = b on GPU.
extern double JacobiMethodGpu(const float *A, const double *b,
                              const float conv_threshold, const int max_iter,
                              double *x, double *x_new, sycl::queue *stream);

// creates N_ROWS x N_ROWS matrix A with N_ROWS+1 on the diagonal and 1
// elsewhere. The elements of the right hand side b all equal 2*n, hence the
// exact solution x to A*x = b is a vector of ones.
void createLinearSystem(float *A, double *b);

// Run the Jacobi method for A*x = b on CPU.
void JacobiMethodCPU(float *A, double *b, float conv_threshold, int max_iter,
                     int *numit, double *x);

int main(int argc, char **argv) {
  // Host variable declaration and allocation
  double *b = NULL;
  float *A = NULL;

  b = sycl::malloc_host<double>(N_ROWS, dpct::get_default_queue());
  memset(b, 0, N_ROWS * sizeof(double));

  A = sycl::malloc_host<float>(N_ROWS * N_ROWS, dpct::get_default_queue());
  memset(A, 0, N_ROWS * N_ROWS * sizeof(float));

  createLinearSystem(A, b);
  double *x = NULL;

  // start with array of all zeroes
  x = (double *)calloc(N_ROWS, sizeof(double));

  float conv_threshold = 1.0e-2;
  int max_iter = 4 * N_ROWS * N_ROWS;
  int cnt = 0;

  // create timer
  StopWatchInterface *timerCPU = NULL, *timerGpu = NULL;
  sdkCreateTimer(&timerCPU);

  // start Host Timer
  sdkStartTimer(&timerCPU);
  JacobiMethodCPU(A, b, conv_threshold, max_iter, &cnt, x);

  double sum = 0.0;

  // Compute error
  for (int i = 0; i < N_ROWS; i++) {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }

  // stop Host timer
  sdkStopTimer(&timerCPU);
  printf("CPU iterations : %d\n", cnt);
  printf("CPU error : %.3e\n", sum);
  printf("CPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerCPU));

  sycl::queue *q;
  q = dpct::get_current_device().create_queue();

  // Device variable allocation and declaration
  float *d_A;
  double *d_b, *d_x, *d_x_new;

  d_b = sycl::malloc_device<double>(N_ROWS, dpct::get_default_queue());
  d_A = sycl::malloc_device<float>(N_ROWS * N_ROWS, dpct::get_default_queue());
  d_x = sycl::malloc_device<double>(N_ROWS, dpct::get_default_queue());
  d_x_new = sycl::malloc_device<double>(N_ROWS, dpct::get_default_queue());

  q->memset(d_x, 0, sizeof(double) * N_ROWS);
  q->memset(d_x_new, 0, sizeof(double) * N_ROWS);

  // Copy from Host variables to Device variables
  q->memcpy(d_A, A, sizeof(float) * N_ROWS * N_ROWS);
  q->memcpy(d_b, b, sizeof(double) * N_ROWS);

  // wait for the memcpy to complete
  q->wait();

  sdkCreateTimer(&timerGpu);
  // start Device Timer
  sdkStartTimer(&timerGpu);

  double sumGPU = 0.0;
  sumGPU = JacobiMethodGpu(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, q);

  // stop Device Timer
  sdkStopTimer(&timerGpu);
  printf("GPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerGpu));

  // Free up allocated memory
  free(d_b, dpct::get_default_queue());
  free(d_A, dpct::get_default_queue());
  free(d_x, dpct::get_default_queue());
  free(d_x_new, dpct::get_default_queue());
  free(A, dpct::get_default_queue());
  free(b, dpct::get_default_queue());

  printf("&&&& jacobiCuda %s\n",
         (fabs(sum - sumGPU) < conv_threshold) ? "PASSED" : "FAILED");

  return (fabs(sum - sumGPU) < conv_threshold) ? EXIT_SUCCESS : EXIT_FAILURE;
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
