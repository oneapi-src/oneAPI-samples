// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  buffer<int> B{ range{16} };

  // ERROR: Create sub-buffer larger than size of parent buffer
  // An exception is thrown from within the buffer constructor
  buffer<int> B2(B, id{8}, range{16});

  return 0;
}

