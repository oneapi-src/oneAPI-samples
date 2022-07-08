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

  // Allocate memory on host
  float *x = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *y = (float *)malloc(ARRAY_SIZE * sizeof(float));

  double tb, te;
  int correct_count = 0;

  init1(x, ARRAY_SIZE);
  init1(y, ARRAY_SIZE);

  printf("Number of OpenMP Devices: %d\n", omp_get_num_devices());

  tb = omp_get_wtime();

  // Allocate memory on device
  float *x_dev =
      (float *)omp_target_alloc_device(ARRAY_SIZE * sizeof(float), deviceId);
  float *y_dev =
      (float *)omp_target_alloc_device(ARRAY_SIZE * sizeof(float), deviceId);

  // Explicit data movement from Host to device
  int error = omp_target_memcpy(x_dev, x, ARRAY_SIZE * sizeof(float), 0, 0,
                                deviceId, 0);
  error = omp_target_memcpy(y_dev, y, ARRAY_SIZE * sizeof(float), 0, 0,
                            deviceId, 0);

#pragma omp target
  {
    for (int i = 0; i < ARRAY_SIZE; i++) x_dev[i] += y_dev[i];
  }

  // Explicit Data Movement from Device to Host
  error = omp_target_memcpy(x, x_dev, ARRAY_SIZE * sizeof(float), 0, 0, 0,
                            deviceId);
  error = omp_target_memcpy(y, y_dev, ARRAY_SIZE * sizeof(float), 0, 0, 0,
                            deviceId);

  init2(y, ARRAY_SIZE);

  // Explicit data movement from Host to device
  error = omp_target_memcpy(x_dev, x, ARRAY_SIZE * sizeof(float), 0, 0,
                            deviceId, 0);
  error = omp_target_memcpy(y_dev, y, ARRAY_SIZE * sizeof(float), 0, 0,
                            deviceId, 0);

#pragma omp target
  {
    for (int i = 0; i < ARRAY_SIZE; i++) x_dev[i] += y_dev[i];
  }
  // Explicit Data Movement from Device to Host
  error = omp_target_memcpy(x, x_dev, ARRAY_SIZE * sizeof(float), 0, 0, 0,
                            deviceId);
  error = omp_target_memcpy(y, y_dev, ARRAY_SIZE * sizeof(float), 0, 0, 0,
                            deviceId);

  te = omp_get_wtime();

  printf("Time of kernel: %lf seconds\n", te - tb);

  for (int i = 0; i < ARRAY_SIZE; i++)
    if (x[i] == 4.0) correct_count++;

  printf("Test: %s\n", (correct_count == ARRAY_SIZE) ? "PASSED!" : "Failed");

  omp_target_free(x_dev, deviceId);
  omp_target_free(y_dev, deviceId);
  free(x);
  free(y);

  return EXIT_SUCCESS;
}
