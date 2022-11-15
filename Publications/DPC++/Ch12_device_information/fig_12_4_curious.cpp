// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {

  // Loop through available platforms
  for (auto const& this_platform : platform::get_platforms() ) {
    std::cout << "Found platform: "
      << this_platform.get_info<info::platform::name>() << "\n";

    // Loop through available devices in this platform
    for (auto const& this_device : this_platform.get_devices() ) {
      std::cout << " Device: "
        << this_device.get_info<info::device::name>() << "\n";
    }
    std::cout << "\n";
  }

  return 0;
}

