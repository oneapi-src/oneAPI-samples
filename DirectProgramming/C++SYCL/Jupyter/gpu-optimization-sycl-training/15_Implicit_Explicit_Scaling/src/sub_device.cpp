//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main(){
  sycl::queue q;
  sycl::device RootDevice =  q.get_device();
  std::cout << "Device: " << RootDevice.get_info<sycl::info::device::name>() << "\n";
  std::cout << "-EUs  : " << RootDevice.get_info<sycl::info::device::max_compute_units>() << "\n\n";

  //# Check if GPU can be partitioned (Stack)
  auto partitions = RootDevice.get_info<sycl::info::device::partition_max_sub_devices>();
  if(partitions > 0){
    std::cout << "-partition_max_sub_devices: " << partitions << "\n\n";
    std::vector<sycl::device> SubDevices = RootDevice.create_sub_devices<
                  sycl::info::partition_property::partition_by_affinity_domain>(
                                                  sycl::info::partition_affinity_domain::numa);
    for (auto &SubDevice : SubDevices) {
      std::cout << "Sub-Device: " << SubDevice.get_info<sycl::info::device::name>() << "\n";
      std::cout << "-EUs      : " << SubDevice.get_info<sycl::info::device::max_compute_units>() << "\n";
    }  
  }
}
