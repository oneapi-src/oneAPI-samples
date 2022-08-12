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

#define N 100

int main(void)
{
  int *arr_host = NULL;
  int *arr_device = NULL;

  arr_host = (int *) malloc(N * sizeof(int));
  arr_device = (int *) omp_target_alloc_device(N * sizeof(int),
					       omp_get_default_device());

  #pragma omp target is_device_ptr(arr_device) map(from: arr_host[0:N])
  {
    for (int i = 0; i < N; ++i) {
      arr_device[i] = i;
      arr_host[i] = arr_device[i];
    }
  }

  printf ("%d, %d, %d \n", arr_host[0], arr_host[N/2], arr_host[N-1]);
}
