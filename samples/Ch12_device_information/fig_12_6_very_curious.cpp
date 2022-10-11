// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

template <typename queryT, typename T>
void do_query( const T& obj_to_query, const std::string& name, int indent=4) {
  std::cout << std::string(indent, ' ') << name << " is '"
    << obj_to_query.template get_info<queryT>() << "'\n";
}


int main() {
  // Loop through the available platforms
  for (auto const& this_platform : platform::get_platforms() ) {
    std::cout << "Found Platform:\n";
    do_query<info::platform::name>(this_platform, "info::platform::name");
    do_query<info::platform::vendor>(this_platform, "info::platform::vendor");
    do_query<info::platform::version>(this_platform, "info::platform::version");
    do_query<info::platform::profile>(this_platform, "info::platform::profile");

    // Loop through the devices available in this plaform
    for (auto &dev : this_platform.get_devices() ) {
      std::cout << "  Device: " << dev.get_info<info::device::name>() << "\n";
      std::cout << "    is_cpu(): " << (dev.is_cpu() ? "Yes" : "No") << "\n";
      std::cout << "    is_gpu(): " << (dev.is_gpu() ? "Yes" : "No") << "\n";
      std::cout << "    is_accelerator(): "
                          << (dev.is_accelerator() ? "Yes" : "No") << "\n";

      do_query<info::device::vendor>(dev, "info::device::vendor");
      do_query<info::device::driver_version>(dev,
                  "info::device::driver_version");
      do_query<info::device::max_work_item_dimensions>(dev,
                  "info::device::max_work_item_dimensions");
      do_query<info::device::max_work_group_size>(dev,
                  "info::device::max_work_group_size");
      do_query<info::device::mem_base_addr_align>(dev,
                  "info::device::mem_base_addr_align");
      do_query<info::device::partition_max_sub_devices>(dev,
                  "info::device::partition_max_sub_devices");

      std::cout << "    Many more queries are available than shown here!\n";
    }
    std::cout << "\n";
  }
  return 0;
}

