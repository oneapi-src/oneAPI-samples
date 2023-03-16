// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  // In the aspect_selector form taking a comma seperated group of
  // aspects, all aspects must be present for a device to be selected.
  queue Q1{ aspect_selector(aspect::fp16, aspect::gpu) };

  // In the aspect_selector form that takes two vectors, the first
  // vector contains aspects that a device must exhibit, and the
  // second contains aspects that must NOT be exhibited.
  queue Q2{ aspect_selector(
		  std::vector{aspect::fp64, aspect::fp16},
		  std::vector{aspect::gpu, aspect::fpga} };

  std::cout << "First selected device is: " <<
    Q1.get_device().get_info<info::device::name>() << "\n";

  std::cout << "Second selected device is: " <<
    Q2.get_device().get_info<info::device::name>() << "\n";

  return 0;
}

