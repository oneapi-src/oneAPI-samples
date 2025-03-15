//==============================================================
// Copyright © 2024 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define SIZE 320

int num_devices = omp_get_num_devices();
int chunksize = SIZE/num_devices;

int main(void)
{
  int *A;
  A = new int[sizeof(int) * SIZE];

  printf ("num_devices = %d\n", num_devices);

  for (int i = 0; i < SIZE; i++)
      A[i] = -9;

  #pragma omp parallel for
  for (int id = 0; id < num_devices; id++) {
      #pragma omp target teams distribute parallel for device(id) \
	      map(tofrom: A[id * chunksize : chunksize])
      for (int i = id * chunksize; i < (id + 1) * chunksize; i++) {
          A[i] = i;
      }
  }

  for (int i = 0; i < SIZE; i++)
    if (A[i] != i)
      printf ("Error in: %d\n", A[i]);
    else
      printf ("%d\n", A[i]);
}

// Snippet end
