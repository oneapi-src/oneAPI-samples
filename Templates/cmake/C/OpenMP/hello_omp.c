//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  printf("Hello, OpenMP C World!\n");
#pragma omp parallel
  { printf("  I am thread %d\n", omp_get_thread_num()); }
  printf("All done, bye.\n");
  return 0;
}
