//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    
   // Create a buffer of ints from an input iterator
  std::vector<int> myVec;
  buffer b1{myVec};
  buffer b2{myVec.begin(), myVec.end()};
  
  // Create a buffer of ints from std::array
  std::array<int,42> my_data;  
  buffer b3{my_data};
  
  
  // Create a buffer of 4 doubles and initialize it from a host pointer
  double myDoubles[4] = {1.1, 2.2, 3.3, 4.4};
  buffer b4{myDoubles, range{4}}; 

  return 0;
}
