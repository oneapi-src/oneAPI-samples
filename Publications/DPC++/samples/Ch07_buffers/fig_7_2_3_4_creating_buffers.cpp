// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Create a buffer of 2x5 ints using the default allocator
  buffer<int, 2, buffer_allocator<int>> b1{range<2>{2, 5}};

  // Create a buffer of 2x5 ints using the default allocator and CTAD for range
  buffer<int, 2> b2{range{2, 5}};

  // Create a buffer of 20 floats using a default-constructed std::allocator
  buffer<float, 1, std::allocator<float>> b3{range{20}};

  // Create a buffer of 20 floats using a passed-in allocator
  std::allocator<float> myFloatAlloc;
  buffer<float, 1, std::allocator<float>> b4{range(20), myFloatAlloc};

  // Create a buffer of 4 doubles and initialize it from a host pointer
  double myDoubles[4] = {1.1, 2.2, 3.3, 4.4};
  buffer b5{myDoubles, range{4}};

  // Create a buffer of 5 doubles and initialize it from a host pointer to
  // const double
  const double myConstDbls[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  buffer b6{myConstDbls, range{5}};

  // Create a buffer from a shared pointer to int
  auto sharedPtr = std::make_shared<int>(42);
  buffer b7{sharedPtr, range{1}};

  // Create a buffer of ints from an input iterator
  std::vector<int> myVec;
  buffer b8{myVec.begin(), myVec.end()};
  buffer b9{myVec};

  // Create a buffer of 2x5 ints and 2 non-overlapping sub-buffers of 5 ints.
  buffer<int, 2> b10{range{2, 5}};
  buffer b11{b10, id{0, 0}, range{1, 5}};
  buffer b12{b10, id{1, 0}, range{1, 5}};

  return 0;
}
