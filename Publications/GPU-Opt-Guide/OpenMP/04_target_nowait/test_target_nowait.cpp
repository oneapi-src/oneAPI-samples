//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
/*
 * This test is taken from OpenMP API 5.0.1 Examples (June 2020)
 * https://www.openmp.org/wp-content/uploads/openmp-examples-5-0-1.pdf
 * (4.13.2 nowait Clause on target Construct)
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 100000 // N must be even

void init(int n, float *v1, float *v2) {
  int i;

  for(i=0; i<n; i++){
    v1[i] = i * 0.25;
    v2[i] = i - 1.25;
  }
}

int main() {
  int i, n=N;
  float v1[N],v2[N],vxv[N];
  double start,end; // timers

  init(n, v1,v2);

  /* Dummy parallel and target (nowait) regions, so as not to measure
     startup time. */
  #pragma omp parallel
  {
     #pragma omp master
     #pragma omp target nowait
       {;}
  }

  start=omp_get_wtime();

  #pragma omp parallel
  {
     #pragma omp master
     #pragma omp target teams distribute parallel for nowait \
       map(to: v1[0:n/2])                                    \
       map(to: v2[0:n/2])                                    \
       map(from: vxv[0:n/2])
     for(i=0; i<n/2; i++){
       vxv[i] = v1[i]*v2[i];
     }

     #pragma omp for
     for(i=n/2; i<n; i++) {
       vxv[i] = v1[i]*v2[i];
     }
     /* Implicit barrier at end of worksharing for. Target region is
        guaranteed to be completed by this point. */
  }

  end=omp_get_wtime();

  printf("vxv[1]=%f, vxv[n-1]=%f, time=%lf\n", vxv[1], vxv[n-1], end-start);
  return 0;
}
