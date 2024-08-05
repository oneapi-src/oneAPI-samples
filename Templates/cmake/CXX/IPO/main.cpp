//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "ipo_cxx_lib.h"
#include <iostream>

int main(int argc, char *argv[]) {
  int ans = plus3(argc);
  std::cout << argc << " + 3 = " << ans << std::endl;

  return 0;
}
