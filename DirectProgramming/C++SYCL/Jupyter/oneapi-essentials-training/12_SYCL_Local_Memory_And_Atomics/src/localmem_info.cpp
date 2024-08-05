//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue q;

  //# Print the device info
  std::cout << "device name   : " << q.get_device().get_info<info::device::name>() << "\n";
  std::cout << "local_mem_size: " << q.get_device().get_info<info::device::local_mem_size>() << "\n";

  auto local_mem_type = q.get_device().get_info<info::device::local_mem_type>();
  if(local_mem_type == info::local_mem_type::local) 
    std::cout << "local_mem_type: info::local_mem_type::local" << "\n";
  else if(local_mem_type == info::local_mem_type::global) 
    std::cout << "local_mem_type: info::local_mem_type::global" << "\n";
  else if(local_mem_type == info::local_mem_type::none) 
    std::cout << "local_mem_type: info::local_mem_type::none" << "\n";
 
  return 0;
}
