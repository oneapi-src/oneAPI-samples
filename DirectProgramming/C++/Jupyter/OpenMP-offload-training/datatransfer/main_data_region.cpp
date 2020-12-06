//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <stdio.h>

constexpr int ARRAY_SIZE = 256;

void init1(float *x, int N) {
  for (int i = 0; i < N; i++) x[i] = 1.0;
}
void init2(float *x, int N) {
  for (int i = 0; i < N; i++) x[i] = 2.0;
}
int main() {
  float x[ARRAY_SIZE], y[ARRAY_SIZE];
  double tb, te;
  int correct_count = 0;

  init1(x, ARRAY_SIZE);
  init1(y, ARRAY_SIZE);

  printf("Number of OpenMP Devices: %d\n", omp_get_num_devices());

  tb = omp_get_wtime();

#include "lab/target_data_region.cpp"

  te = omp_get_wtime();

  printf("Time of kernel: %lf seconds\n", te - tb);

  for (int i = 0; i < ARRAY_SIZE; i++)
    if (x[i] == 4.0) correct_count++;

  printf("Test: %s\n", (correct_count == ARRAY_SIZE) ? "PASSED!" : "Failed");
}
