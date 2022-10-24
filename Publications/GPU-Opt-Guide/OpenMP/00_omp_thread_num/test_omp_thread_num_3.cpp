//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <stdio.h>
#include <omp.h>

void foo() {

  #pragma omp target teams distribute parallel for simd simdlen(64)
  for (int i = 0; i < 100; ++i) {

    printf ("team_num=%d num_threads=%d thread_id=%d \n",
	    omp_get_team_num(),
	    omp_get_num_threads(),
	    omp_get_thread_num());
  }
}

int main(void) {

  foo();
  return 0;
}
