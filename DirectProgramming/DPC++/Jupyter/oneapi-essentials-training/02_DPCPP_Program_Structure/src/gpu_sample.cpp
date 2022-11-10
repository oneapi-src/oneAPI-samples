//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  //# Create a device queue with device selector
  
  gpu_selector selector;
  //cpu_selector selector;
  //default_selector selector;
  //host_selector selector;
  
  queue q(selector);

  //# Print the device name
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}
