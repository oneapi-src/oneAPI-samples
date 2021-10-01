//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <stdio.h>
#include <string.h>

#pragma omp declare target
int foo(int y) {
  printf("called from device, y = %d\n", y);
  return y;
}
#pragma omp end declare target

int main() {
  int x = 0;
  int y = 100;
  int (*fptr)(int) = foo;
#pragma omp target teams \
        distribute parallel for \
        firstprivate(y) reduction(+: x) map(to: fptr)
  for (int k = 0; k < 16; k++) {
    fptr = foo;
    x = x + fptr(y + k);
  }
  printf("Output x = %d\n", x);
}
