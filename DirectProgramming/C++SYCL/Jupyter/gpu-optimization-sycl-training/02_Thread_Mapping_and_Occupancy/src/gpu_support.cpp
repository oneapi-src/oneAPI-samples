//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
   
  //# Query for device information
  auto device_name = q.get_device().get_info<sycl::info::device::name>();
  auto wg_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto slm_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
    
  std::cout << "Device : " << device_name << "\n";

  std::cout << "Max Work-Group Size : " << wg_size << "\n";

  std::cout << "Supported Sub-Group Sizes : ";
  for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << "\n";

  std::cout << "Local Memory Size : " << slm_size << "\n";
   
  q.submit([&](sycl::handler &h){
    h.parallel_for(sycl::nd_range<3>(sycl::range<3>(112, 120, 128), sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item)[[intel::reqd_sub_group_size(32)]] {
     // Kernel Code
    });
  }).wait();
}
