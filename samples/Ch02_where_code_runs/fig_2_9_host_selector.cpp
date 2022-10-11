// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// TODO: Following example is already CPU selector.  Decide what to do with this one
#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  // Create queue to use the host device explicitly
  queue Q{ cpu_selector_v };

  std::cout << "Selected device: " <<
    Q.get_device().get_info<info::device::name>() << "\n";
  std::cout << " -> Device vendor: " <<
    Q.get_device().get_info<info::device::vendor>() << "\n";

  return 0;
}

