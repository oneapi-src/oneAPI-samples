//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  //# Create a device queue with device selector
  
  queue q(gpu_selector_v);
  //queue q(cpu_selector_v);
  //queue q(accelerator_selector_v);
  //queue q(default_selector_v);
  //queue q;

  //# Print the device name
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}
