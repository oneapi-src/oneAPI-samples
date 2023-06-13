//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// The main program has no SYCL calls. It just implements the main program
// logic and calls routines defined elsewhere to do the real work.
//

#include "offload.h"

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  int n_items = 16;
  if (argc > 1) {
    n_items = atoi(argv[1]);
  }

  std::vector<int> ans(n_items);
  do_work(ans);

  // # Print Output
  for (auto &&x : ans) {
    std::cout << x << std::endl;
  }

  return 0;
}