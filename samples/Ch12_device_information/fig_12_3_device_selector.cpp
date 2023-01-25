// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int my_selector(const device &dev) {
  int score = -1;

  // We prefer non-Martian GPUs, especially ACME GPUs
  if (dev.is_gpu()) {
    if (dev.get_info<info::device::vendor>().find("ACME") != std::string::npos)
      score += 25;

    if (dev.get_info<info::device::vendor>().find("Martian") ==
        std::string::npos)
      score += 800;
  }

  // If there is no GPU on the system all devices will be given a negative score
  // and the selector will not select a device. This will cause an exception.
  return score;
}

int main() {
  try {
    auto Q = queue{ my_selector };
    std::cout << "After checking for a GPU, we are running on:\n "
              << Q.get_device().get_info<info::device::name>() << "\n";
  } catch (exception const& ex) {
    std::cout << "Custom device selector did not select a device.\n";
    std::cout << "Caught this SYCL exception: " << ex.what() << std::endl;
  }

  // Sample output using a system with a GPU:
  // After checking for a GPU, we are running on:
  //  Intel(R) Gen9 HD Graphics NEO.
  // 
  // Sample output using a system with an FPGA accelerator, but no GPU:
  // After checking for a GPU, we are running on:
  //  Custom device selector did not select a device.
  //  Caught this SYCL exception: No device of requested type available.
  //  ...(PI_ERROR_DEVICE_NOT_FOUND)

  return 0;
}

