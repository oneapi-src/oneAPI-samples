//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
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
  double w[SIZE];                      /* output */
  double u[SIZE], dx[P * P];           /* input */
  double ur[SIZE], us[SIZE], ut[SIZE]; /* work arrays */
  int b, i, j, k, l;                   /* loop counters */
  double start, end;                   /* timers */

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

  start = omp_get_wtime();

  /* offload the kernel */
  #pragma omp target teams distribute parallel for simd simdlen(16) collapse(4) \
    map(to:u[0:SIZE],dx[0:P*P]) \
    map(from:w[0:SIZE])         \
    map(to:ur[0:SIZE])          \
    map(to:us[0:SIZE])          \
    map(to:ut[0:SIZE])          \
    private(b,i,j,k,l)
  for (b = 0; b < BLOCKS; b++) {
    for (i = 0; i < P; i++) {
      for (j = 0; j < P; j++) {
        for (k = 0; k < P; k++) {
          w[IDX4(b, i, j, k)] = 0.;
          ur[IDX4(b, i, j, k)] = 0.;
          us[IDX4(b, i, j, k)] = 0.;
          ut[IDX4(b, i, j, k)] = 0.;

          for (l = 0; l < P; l++) {
            ur[IDX4(b, i, j, k)] += dx[IDX2(i, l)] * u[IDX4(b, l, j, k)];
            us[IDX4(b, i, j, k)] += dx[IDX2(k, l)] * u[IDX4(b, i, l, k)];
            ut[IDX4(b, i, j, k)] += dx[IDX2(j, l)] * u[IDX4(b, i, j, l)];
          }

          w[IDX4(b, i, j, k)] = ur[IDX4(b, i, j, k)] * us[IDX4(b, i, j, k)] *
                                ut[IDX4(b, i, j, k)];
        }
      }
    }
  }

  end = omp_get_wtime();

  /* print result */
  printf("collapse-clause: w[0]=%lf time=%lf\n", w[0], end - start);

  return 0;
}
