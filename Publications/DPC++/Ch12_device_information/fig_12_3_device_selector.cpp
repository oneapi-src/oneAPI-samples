// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

class my_selector : public device_selector {
  public:
    int operator()(const device &dev) const {
      int score = -1;

      // We prefer non-Martian GPUs, especially ACME GPUs
      if (dev.is_gpu()) {
        if (dev.get_info<info::device::vendor>().find("ACME")
            != std::string::npos) score += 25;

        if (dev.get_info<info::device::vendor>().find("Martian")
            == std::string::npos) score += 800;
      }

      // Give host device points so it is used if no GPU is available.
      // Without these next two lines, systems with no GPU would select
      // nothing, since we initialize the score to a negative number above.
      if (dev.is_host()) score += 100;

      return score;
    }
};

int main() {
  auto Q = queue{ my_selector{} };

  std::cout << "After checking for a GPU, we are running on:\n "
    << Q.get_device().get_info<info::device::name>() << "\n";

  // Sample output using a system with a GPU:
  // After checking for a GPU, we are running on:
  //  Intel(R) Gen9 HD Graphics NEO.
  // 
  // Sample output using a system with an FPGA accelerator, but no GPU:
  // After checking for a GPU, we are running on:
  //  SYCL host device.

  return 0;
}

