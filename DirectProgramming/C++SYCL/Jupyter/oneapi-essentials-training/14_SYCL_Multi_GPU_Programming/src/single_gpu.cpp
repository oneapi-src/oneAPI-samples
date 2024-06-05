//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  // Create a device queue with device selector
  sycl::queue q(sycl::gpu_selector_v);

  // Print the device name
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  return 0;
}
