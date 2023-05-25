//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <math.h>
#include <omp.h>

#define P 16
#define BLOCKS 8
#define SIZE (BLOCKS * P * P * P)

#define MAX 100
#define scaled_rand() ((rand() % MAX) / (1.0 * MAX))

#define IDX2(i, j) (i * P + j)
#define IDX4(b, i, j, k) (b * P * P * P + i * P * P + j * P + k)

int main(void) {
  double w[SIZE];            /* output */
  double u[SIZE], dx[P * P]; /* input */
  int b, i, j, k, l;         /* loop counters */
  double start, end;         /* timers */

  omp_set_default_device(0);

  /* dummy target region, so as not to measure startup time. */
  #pragma omp target
  { ; }

  /* initialize input with random values */
  srand(0);
  for (int i = 0; i < SIZE; i++)
    u[i] = scaled_rand();

  for (int i = 0; i < P * P; i++)
    dx[i] = scaled_rand();

  /* map data to device */
  #pragma omp target enter data map(to: u[0:SIZE], dx[0:P * P])

  start = omp_get_wtime();

  /* offload the kernel with no collapse clause */
  #pragma omp target teams distribute parallel for \
    private(b, i, j, k, l)
  for (b = 0; b < BLOCKS; b++) {
    for (i = 0; i < P; i++) {
      for (j = 0; j < P; j++) {
        for (k = 0; k < P; k++) {
          double ur = 0.;
          double us = 0.;
          double ut = 0.;

          for (l = 0; l < P; l++) {
            ur += dx[IDX2(i, l)] * u[IDX4(b, l, j, k)];
            us += dx[IDX2(k, l)] * u[IDX4(b, i, l, k)];
            ut += dx[IDX2(j, l)] * u[IDX4(b, i, j, l)];
          }

          w[IDX4(b, i, j, k)] = ur * us * ut;
        }
      }
    }
  }

  end = omp_get_wtime();

  #pragma omp target exit data map(from: w[0:SIZE])

  /* print result */
  printf("no-collapse-clause: w[0]=%lf time=%lf\n", w[0], end - start);

  return 0;
}
// Snippet end
