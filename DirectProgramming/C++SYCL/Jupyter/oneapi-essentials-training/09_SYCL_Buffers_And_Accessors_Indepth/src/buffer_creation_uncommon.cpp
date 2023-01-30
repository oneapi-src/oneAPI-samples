//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  
  // Create a buffer of 2x5 ints using the default allocator and CTAD for dimensions
  buffer<int, 2> b1{range{2, 5}};
    
  //Dimensions defaults to 1

  // Create a buffer of 20 floats using a default-constructed std::allocator
  buffer<float> b2{range{20}};
  
  // Create a buffer from a shared pointer to int
  auto sharedPtr = std::make_shared<int>(42);
  buffer b3{sharedPtr, range{1}};
  
  // Create a buffer of 2x5 ints and 2 non-overlapping sub-buffers of 5 ints.
  buffer<int, 2> b4{range{2, 5}};
  buffer b5{b4, id{0, 0}, range{1, 5}};
  buffer b6{b4, id{1, 0}, range{1, 5}};
    
  // Create a buffer of 5 doubles and initialize it from a host pointer to
  // const double
  const double myConstDbls[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  buffer b7{myConstDbls, range{5}};   

  return 0;
}
