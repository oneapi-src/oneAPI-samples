//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <omp.h>
#include <sstream>

int main(int argc, char *argv[]) {
  std::cout << "Hello, OpenMP C World!" << std::endl;
#pragma omp parallel
  {
    std::ostringstream msg;
    msg << "  I am thread " << omp_get_thread_num() << std::endl;
    std::cout << msg.str();
  }
  std::cout << "All done, bye." << std::endl;
  return 0;
}
