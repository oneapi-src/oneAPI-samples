// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  queue Q;
  device dev = Q.get_device();

  std::cout << "We are running on:\n"
    << dev.get_info<info::device::name>() << "\n";

  // Query results like the following can be used to calculate how
  // large your kernel invocations should be.
  auto maxWG = dev.get_info<info::device::max_work_group_size>();
  auto maxGmem = dev.get_info<info::device::global_mem_size>();
  auto maxLmem = dev.get_info<info::device::local_mem_size>();

  std::cout << "Max WG size is " << maxWG
    << "\nMax Global memory size is " << maxGmem
    << "\nMax Local memory size is " << maxLmem << "\n";

  return 0;
}

