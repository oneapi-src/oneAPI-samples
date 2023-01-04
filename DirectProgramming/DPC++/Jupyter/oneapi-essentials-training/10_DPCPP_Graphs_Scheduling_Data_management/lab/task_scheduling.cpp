//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 256;

int main() {
  queue q;
 
  //# 3 vectors initialized to values
  std::vector<int> data1(N, 1);
  std::vector<int> data2(N, 2);
  std::vector<int> data3(N, 3);
 
  //# STEP 1 : Create buffers for data1, data2 and data3

  //# YOUR CODE GOES HERE





  //# STEP 2 : Create a kernel to update data1 += data3, set accessor permissions

  //# YOUR CODE GOES HERE





  //# STEP 3 : Create a kernel to update data2 *= 2, set accessor permissions

  //# YOUR CODE GOES HERE





  //# STEP 4 : Create a kernel to update data3 = data1 + data2, set accessor permissions

  //# YOUR CODE GOES HERE





  //# STEP 5 : Create a host accessor to copy back data3

  //# YOUR CODE GOES HERE




  std::cout << "Output = " << data3[0] << "\n";
  return 0;
}
