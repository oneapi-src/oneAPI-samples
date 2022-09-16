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

#define length 65536

int main(void)
{
  int device_id = omp_get_default_device();
  size_t bytes = length*sizeof(double);
  double * __restrict A;
  double * __restrict B;
  double * __restrict C;
  double scalar = 3.0;
  double ar;
  double br;
  double cr;
  double asum;

  // Allocate arrays in device memory

  A = (double *) omp_target_alloc_device(bytes, device_id);
  if (A == NULL){
     printf(" ERROR: Cannot allocate space for A using omp_target_alloc_device().\n");
     exit(1);
  }

  B = (double *) omp_target_alloc_device(bytes, device_id);
  if (B == NULL){
     printf(" ERROR: Cannot allocate space for B using omp_target_alloc_device().\n");
     exit(1);
  }

  C = (double *) omp_target_alloc_device(bytes, device_id);
  if (C == NULL){
     printf(" ERROR: Cannot allocate space for C using omp_target_alloc_device().\n");
     exit(1);
  }

  #pragma omp target data use_device_ptr(A,B,C)
  {
      // Initialize the arrays

      #pragma omp target teams distribute parallel for
      for (size_t i=0; i<length; i++) {
          A[i] = 2.0;
          B[i] = 2.0;
          C[i] = 0.0;
      }

      // Perform the computation

      #pragma omp target teams distribute parallel for
      for (size_t i=0; i<length; i++) {
          C[i] += A[i] + scalar * B[i];
      }

      // Validate and output results

      ar = 2.0;
      br = 2.0;
      cr = 0.0;
      for (int i=0; i<length; i++) {
          cr += ar + scalar * br;
      }

      asum = 0.0;
      #pragma omp target teams distribute parallel for reduction(+:asum)
      for (size_t i=0; i<length; i++) {
          asum += fabs(C[i]);
      }

  } // end target data

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
      printf("Solution validates. Checksum = %lf\n", asum);
  }

  return 0;
}
