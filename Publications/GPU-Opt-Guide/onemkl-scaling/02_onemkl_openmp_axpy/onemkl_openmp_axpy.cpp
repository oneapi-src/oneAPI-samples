//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include "mkl.h"
#include "mkl_omp_offload.h"
#include <omp.h>
#include <stdio.h>

int main() {

  double *x, *y, alpha;
  MKL_INT n, incx, incy, i;

  // Initialize data for AXPY
  alpha = 1.0;
  n = 8192;
  incx = 1;
  incy = 1;

  // Allocate and initialize arrays for vectors
  x = (double *)mkl_malloc(n * sizeof(double), 128);
  y = (double *)mkl_malloc(n * sizeof(double), 128);
  if ((x == NULL) || (y == NULL)) {
    printf("Error in vector allocation\n");
    return 1;
  }
  for (i = 0; i < n; i++) {
    x[i] = rand() / (double)RAND_MAX - .5;
    y[i] = rand() / (double)RAND_MAX - .5;
  }

  printf("First 10 elements of the output vector Y before AXPY:\n");
  for (i = 0; i < 10; i++) {
    printf("%lf ", y[i]);
  }
  printf("\n\n");

  // Detect number of available devices
  int nb_devices = omp_get_num_devices();

  // Copy data to device and perform computation
  if (nb_devices > 1) {
    printf("2 devices are detected. AXPY operation is divided into 2 to take "
           "advantage of explicit scaling\n");
    printf("Copy x[0..%lld] and y[0..%lld] to device 0\n", n / 2 - 1,
           n / 2 - 1);
#pragma omp target data map(to : x[0 : n / 2]) map(tofrom : y[0 : n / 2])      \
    device(0)
    {
      printf("Copy x[%lld..%lld] and y[%lld..%lld] to device 1\n", n / 2, n - 1,
             n / 2, n - 1);
#pragma omp target data map(to : x[n / 2 : n - n / 2])                         \
    map(tofrom : y[n / 2 : n - n / 2]) device(1)
      {
        double *x1 = &x[n / 2];
        double *y1 = &y[n / 2];
#pragma omp dispatch device(0) nowait
        cblas_daxpy(n / 2, alpha, x, incx, y, incy);
#pragma omp dispatch device(1)
        cblas_daxpy(n / 2, alpha, x1, incx, y1, incy);
#pragma omp taskwait
      }
    }
  } else {
    printf("1 device is detected. Entire AXPY operation is performed on that "
           "device\n");
    printf("Copy x[0..%lld] and y[0..%lld] to device 0\n", n - 1, n - 1);
#pragma omp target data map(to : x[0 : n]) map(tofrom : y[0 : n]) device(0)
    {
#pragma omp dispatch device(0)
      cblas_daxpy(n, alpha, x, incx, y, incy);
    }
  }
  // End of computation

  printf("\nFirst 10 elements of the output vector Y after AXPY:\n");
  for (i = 0; i < 10; i++) {
    printf("%lf ", y[i]);
  }
  printf("\n");

  mkl_free(x);
  mkl_free(y);
  return 0;
}
// Snippet end
