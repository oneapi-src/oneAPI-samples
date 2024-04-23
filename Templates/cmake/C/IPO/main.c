//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "ipo_c_lib.h"

#include <stdio.h>

int main(int argc, char *argv[]) {
  int ans = plus3(argc);
  printf("%d + 3 = %d\n", argc, ans);
  return 0;
}
