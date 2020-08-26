/* ==============================================================
 * Copyright (c) 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ==============================================================
 */

#include <iostream>

using namespace std;

int main() {
  cout << "Hello, Internet of Things World!" << endl;
#ifdef __INTEL_COMPILER
  cout << "The Intel(R) C++ Compiler Classic was used for compiling this sample."
       << endl;
#else
  cout << "The Intel(R) C++ Compiler Classic was not used for compiling this sample."
       << endl;
#endif
  return 0;
}
