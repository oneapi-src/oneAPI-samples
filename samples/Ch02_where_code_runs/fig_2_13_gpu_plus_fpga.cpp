// Copyright (C) 2020-2021 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // For fpga_selector
#include <iostream>
using namespace sycl;

int main() {
  queue my_gpu_queue( gpu_selector_v );
  queue my_fpga_queue( ext::intel::fpga_selector{} );

  std::cout << "Selected device 1: " <<
    my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

  std::cout << "Selected device 2: " <<
    my_fpga_queue.get_device().get_info<info::device::name>() << "\n";

  return 0;
}

