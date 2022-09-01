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

#define iterations 100
#define length     64*1024*1024

int main(void)
{
  int device_id = omp_get_default_device();
  int host_id = omp_get_initial_device();
  size_t bytes = length*sizeof(double);
  double * __restrict h_A;
  double * __restrict h_B;
  double * __restrict h_C;
  double * __restrict d_A;
  double * __restrict d_B;
  double * __restrict d_C;
  double scalar = 3.0;
  double nstream_time = 0.0;

  // Allocate arrays h_A, h_B, and h_C on the host using plain malloc()

  h_A = (double *) malloc(bytes);
  if (h_A == NULL){
     printf(" ERROR: Cannot allocate space for h_A using plain malloc().\n");
     exit(1);
  }

  h_B = (double *) malloc(bytes);
  if (h_B == NULL){
     printf(" ERROR: Cannot allocate space for h_B using plain malloc().\n");
     exit(1);
  }

  h_C = (double *) malloc(bytes);
  if (h_C == NULL){
     printf(" ERROR: Cannot allocate space for h_C using plain malloc().\n");
     exit(1);
  }

  // Allocate arrays d_A, d_B, and d_C on the device using omp_target_alloc()

  d_A = (double *) omp_target_alloc(bytes, device_id);
  if (d_A == NULL){
     printf(" ERROR: Cannot allocate space for d_A using omp_target_alloc().\n");
     exit(1);
  }

  d_B = (double *) omp_target_alloc(bytes, device_id);
  if (d_B == NULL){
     printf(" ERROR: Cannot allocate space for d_B using omp_target_alloc().\n");
     exit(1);
  }

  d_C = (double *) omp_target_alloc(bytes, device_id);
  if (d_C == NULL){
     printf(" ERROR: Cannot allocate space for d_C using omp_target_alloc().\n");
     exit(1);
  }

  // Initialize the arrays on the host

  #pragma omp parallel for
  for (size_t i=0; i<length; i++) {
      h_A[i] = 2.0;
      h_B[i] = 2.0;
      h_C[i] = 0.0;
  }

  // Call omp_target_memcpy() to copy values from host to device

  int rc = 0;
  rc = omp_target_memcpy(d_A, h_A, bytes, 0, 0, device_id, host_id);
  if (rc) {
     printf("ERROR: omp_target_memcpy(A) returned %d\n", rc);
     exit(1);
  }

  rc = omp_target_memcpy(d_B, h_B, bytes, 0, 0, device_id, host_id);
  if (rc) {
     printf("ERROR: omp_target_memcpy(B) returned %d\n", rc);
     exit(1);
  }

  rc = omp_target_memcpy(d_C, h_C, bytes, 0, 0, device_id, host_id);
  if (rc) {
     printf("ERROR: omp_target_memcpy(C) returned %d\n", rc);
     exit(1);
  }

  // Perform the computation

  nstream_time = omp_get_wtime();
  for (int iter = 0; iter<iterations; iter++) {
      #pragma omp target teams distribute parallel for \
        is_device_ptr(d_A,d_B,d_C)
      for (size_t i=0; i<length; i++) {
          d_C[i] += d_A[i] + scalar * d_B[i];
      }
  }
  nstream_time = omp_get_wtime() - nstream_time;

  // Call omp_target_memcpy() to copy values from device to host

  rc = omp_target_memcpy(h_C, d_C, bytes, 0, 0, host_id, device_id);
  if (rc) {
     printf("ERROR: omp_target_memcpy(A) returned %d\n", rc);
     exit(1);
  }

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
      asum += fabs(h_C[i]);
  }

  free(h_A);
  free(h_B);
  free(h_C);
  omp_target_free(d_A, device_id);
  omp_target_free(d_B, device_id);
  omp_target_free(d_C, device_id);

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
