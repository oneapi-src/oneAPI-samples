// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {

// BEGIN CODE SNIP
  auto GPU_is_available = false;

  try {
    device testForGPU(gpu_selector_v);
    GPU_is_available = true;
  } catch (exception const& ex) {
    std::cout << "Caught this SYCL exception: " << ex.what() << std::endl;
  }

  auto Q = GPU_is_available ? queue(gpu_selector_v) : queue(default_selector_v);

  std::cout << "After checking for a GPU, we are running on:\n "
    << Q.get_device().get_info<info::device::name>() << "\n";

  // sample output using a system with a GPU:
  // After checking for a GPU, we are running on:
  //  Intel(R) Gen9 HD Graphics NEO.
  // 
  // sample output using a system with an FPGA accelerator, but no GPU:
  // Caught this SYCL exception: No device of requested type available.
  // ...(PI_ERROR_DEVICE_NOT_FOUND)
  // After checking for a GPU, we are running on:
  //  SYCL host device.
  //

// END CODE SNIP
  return 0;
}

