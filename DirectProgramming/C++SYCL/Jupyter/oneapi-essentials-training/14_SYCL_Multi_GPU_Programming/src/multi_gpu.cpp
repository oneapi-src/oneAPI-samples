//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  // get all GPUs devices into a vector
  auto gpus = sycl::platform(sycl::gpu_selector_v).get_devices();

  // Print the device names
  for(auto gpu : gpus)
    std::cout << "Device: " << gpu.get_info<sycl::info::device::name>() << "\n";

  return 0;
}
