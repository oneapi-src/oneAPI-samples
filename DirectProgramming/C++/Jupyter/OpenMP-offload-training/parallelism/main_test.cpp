//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <stdio.h>

constexpr int ARRAY_SIZE = 256;
constexpr int NUM_BLOCKS = 9;

int main(int argc, char *argv[]) {
  int i, ib, is_cpu = 1, num_teams = 0;
  double tstart, tstop;
  float x[ARRAY_SIZE], y[ARRAY_SIZE];

  float a = 1.0f;
  float tolerance = 0.01f;
  int correct_count = 0;

  // Initialize some data
  for (i = 0; i < ARRAY_SIZE; i++) {
    x[i] = (float)i;
    y[i] = (float)i;
  }

  tstart = omp_get_wtime();

#include "saxpy_func_parallel_solution.cpp"

  tstop = omp_get_wtime();

  printf("Number of OpenMP Devices Available: %d\n", omp_get_num_devices());
  printf("Running on %s.\n", is_cpu ? "CPU" : "GPU");
  printf("Work took %f seconds.\n", tstop - tstart);
  printf("Number of Teams Created: %d\n", num_teams);
  for (int i = 0; i < ARRAY_SIZE; i++)
    if (abs(y[i] - (a * i + i)) < tolerance)
      correct_count++;
    else {
      printf("Incorrect Result at Element [%d] : %f\n", i, y[i]);
      break;
    }
  printf("Test: %s\n", (correct_count == ARRAY_SIZE) ? "PASSED!" : "Failed");
}
