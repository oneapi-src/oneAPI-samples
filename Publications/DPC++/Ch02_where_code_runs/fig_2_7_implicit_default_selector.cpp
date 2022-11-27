// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  // Create queue on whatever default device that the implementation
  // chooses. Implicit use of the default_selector. 
  queue Q;

  std::cout << "Selected device: " <<
  Q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}

