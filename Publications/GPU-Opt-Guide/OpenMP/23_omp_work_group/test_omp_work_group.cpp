//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <stdio.h>
#include <omp.h>

double * make_array(int n, double value) {
  double* array = static_cast<double*>(malloc(n * sizeof(double)));
  for (int i = 0; i < n; i++) {
    array[i] = value / (100.0 + i);
  }
  return array;
}

int main() {

  // begin
  int N = 2048;

  double* A = make_array(N, 0.8);
  double* B = make_array(N, 0.65);
  double* C = make_array(N*N, 2.5);

  int i, j;
  double val = 0.0;

  #pragma omp target map(to:A[0:N],B[0:N],C[0:N*N]) map(tofrom:val)
  {

  #pragma omp teams distribute parallel for collapse(2) reduction(+ : val)
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        val += C[i * N + j] * A[i] * B[j];
      }
    }
  }

  printf("Reduced val[%f10.3]", val);

  free(A);
  free(B);
  free(C);
  // end

  return 0;
}
