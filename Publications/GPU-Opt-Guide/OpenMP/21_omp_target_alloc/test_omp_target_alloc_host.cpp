//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#pragma omp requires unified_address

#define iterations 100
#define length     64*1024*1024

int main(void)
{
  int device_id = omp_get_default_device();
  size_t bytes = length*sizeof(double);
  double * __restrict A;
  double * __restrict B;
  double * __restrict C;
  double scalar = 3.0;
  double nstream_time = 0.0;

  // Allocate arrays in host memory

  A = (double *) omp_target_alloc_host(bytes, device_id);
  if (A == NULL){
     printf(" ERROR: Cannot allocate space for A using omp_target_alloc_host().\n");
     exit(1);
  }

  B = (double *) omp_target_alloc_host(bytes, device_id);
  if (B == NULL){
     printf(" ERROR: Cannot allocate space for B using omp_target_alloc_host().\n");
     exit(1);
  }

  C = (double *) omp_target_alloc_host(bytes, device_id);
  if (C == NULL){
     printf(" ERROR: Cannot allocate space for C using omp_target_alloc_host().\n");
     exit(1);
  }

  // Initialize the arrays

  #pragma omp parallel for
  for (size_t i=0; i<length; i++) {
      A[i] = 2.0;
      B[i] = 2.0;
      C[i] = 0.0;
  }

  // Perform the computation

  nstream_time = omp_get_wtime();
  for (int iter = 0; iter<iterations; iter++) {
      #pragma omp target teams distribute parallel for
      for (size_t i=0; i<length; i++) {
          C[i] += A[i] + scalar * B[i];
      }
  }
  nstream_time = omp_get_wtime() - nstream_time;

  // Validate and output results

  double ar = 2.0;
  double br = 2.0;
  double cr = 0.0;
  for (int iter = 0; iter<iterations; iter++) {
      for (int i=0; i<length; i++) {
          cr += ar + scalar * br;
      }
  }

  double asum = 0.0;
  #pragma omp parallel for reduction(+:asum)
  for (size_t i=0; i<length; i++) {
      asum += fabs(C[i]);
  }

  omp_target_free(A, device_id);
  omp_target_free(B, device_id);
  omp_target_free(C, device_id);

  double epsilon=1.e-8;
  if (fabs(cr - asum)/asum > epsilon) {
      printf("Failed Validation on output array\n"
             "       Expected checksum: %lf\n"
             "       Observed checksum: %lf\n"
             "ERROR: solution did not validate\n", cr, asum);
      return 1;
  } else {
      printf("Solution validates\n");
      double avgtime = nstream_time/iterations;
      printf("Checksum = %lf; Avg time (s): %lf\n", asum, avgtime);
  }

  return 0;
}
