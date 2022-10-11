// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

// START CODE SNIP

int my_selector(const device &dev) {
  if (dev.get_info<info::device::name>().find("Arria") != std::string::npos &&
      dev.get_info<info::device::vendor>().find("Intel") != std::string::npos) {
    return 1;
  }
  return -1;
}

// END CODE SNIP


int main() {
  queue Q( my_selector );

  std::cout << "Selected device is: " <<
    Q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}

