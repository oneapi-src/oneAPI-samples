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
  std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;
  // Snippet end

  return 0;
}
