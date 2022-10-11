// Copyright (C) 2020-2021 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // For fpga_selector
#include <iostream>
#include <string>
using namespace sycl;

void output_dev_info( const device& dev, const std::string& selector_name) {
  std::cout << selector_name << ": Selected device: " <<
    dev.get_info<info::device::name>() << "\n";
  std::cout << "                  -> Device vendor: " <<
    dev.get_info<info::device::vendor>() << "\n";
}

int main() {
  output_dev_info( device{ default_selector_v}, "default_selector" );
  output_dev_info( device{ cpu_selector_v}, "cpu_selector" );
  output_dev_info( device{ gpu_selector_v}, "gpu_selector" );
  output_dev_info( device{ accelerator_selector_v}, "accelerator_selector" );
  output_dev_info( device{ ext::intel::fpga_selector{}}, "fpga_selector" );

  return 0;
}

