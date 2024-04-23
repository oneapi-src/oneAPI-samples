//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>

int main(void) {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  // Snippet begin
  std::cout << "Sub-group Sizes: ";
  for (const auto &s :
       q.get_device().get_info<sycl::info::device::sub_group_sizes>()) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
  // Snippet end

  return 0;
}
