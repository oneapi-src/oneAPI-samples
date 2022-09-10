//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

constexpr int ARRAY_SIZE = 256;

void init1(float *x, int N) {
  for (int i = 0; i < N; i++) x[i] = 1.0;
}
void init2(float *x, int N) {
  for (int i = 0; i < N; i++) x[i] = 2.0;
}
int main() {
  int deviceId = (omp_get_num_devices() > 0) ? omp_get_default_device()
                                             : omp_get_initial_device();

#include "lab/alloc_func.cpp"

  double tb, te;
  int correct_count = 0;

  init1(x, ARRAY_SIZE);
  init1(y, ARRAY_SIZE);

  printf("Number of OpenMP Devices: %d\n", omp_get_num_devices());

  tb = omp_get_wtime();

#pragma omp target
  {
    for (int i = 0; i < ARRAY_SIZE; i++) x[i] += y[i];
  }

  init2(y, ARRAY_SIZE);

#pragma omp target
  {
    for (int i = 0; i < ARRAY_SIZE; i++) x[i] += y[i];
  }

  te = omp_get_wtime();

  printf("Time of kernel: %lf seconds\n", te - tb);

  for (int i = 0; i < ARRAY_SIZE; i++)
    if (x[i] == 4.0) correct_count++;

  printf("Test: %s\n", (correct_count == ARRAY_SIZE) ? "PASSED!" : "Failed");

  omp_target_free(x, deviceId);
  omp_target_free(y, deviceId);
}
