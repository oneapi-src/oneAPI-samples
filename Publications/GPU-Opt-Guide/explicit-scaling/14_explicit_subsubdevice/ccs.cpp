//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;
int main() {
  sycl::device d(sycl::gpu_selector_v);
  std::vector<sycl::device> *subdevices = new std::vector<sycl::device>();
  std::vector<sycl::device> *CCS = new std::vector<sycl::device>();
  auto part_prop = d.get_info<sycl::info::device::partition_properties>();
  size_t num_of_tiles;
  size_t num_of_ccs;
  if (part_prop.empty()) {
    num_of_tiles = 1;
  } else {
    for (int i = 0; i < part_prop.size(); i++) {
      if (part_prop[i] ==
          sycl::info::partition_property::partition_by_affinity_domain) {
        auto sub_devices = d.create_sub_devices<
            sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);
        num_of_tiles = sub_devices.size();
        for (int j = 0; j < num_of_tiles; j++)
          subdevices->push_back(sub_devices[j]);
        break;
      } else {
        num_of_tiles = 1;
      }
    }
  }
  std::cout << "List of Tiles:\n";
  for (int i = 0; i < num_of_tiles; i++) {
    std::cout << i << ") Device name: "
              << (*subdevices)[i].get_info<sycl::info::device::name>() << "\n";
    std::cout
        << "  Max Compute Units: "
        << (*subdevices)[i].get_info<sycl::info::device::max_compute_units>()
        << "\n";
  }
  for (int j = 0; j < num_of_tiles; j++) {
    auto part_prop1 =
        (*subdevices)[j].get_info<sycl::info::device::partition_properties>();
    if (part_prop1.empty()) {
      std::cout << "No partition support\n";
    } else {
      for (int i = 0; i < part_prop1.size(); i++) {
        if (part_prop1[i] ==
            sycl::info::partition_property::partition_by_affinity_domain) {
          auto ccses =
              (*subdevices)[j]
                  .create_sub_devices<sycl::info::partition_property::
                                          partition_by_affinity_domain>(
                      sycl::info::partition_affinity_domain::numa);
          num_of_ccs = ccses.size();
          for (int k = 0; k < num_of_ccs; k++)
            CCS->push_back(ccses[k]);
          break;
        } else {
          num_of_ccs = 1;
        }
      }
    }
  }
  std::cout << "List of Compute Command Streamers:\n";
  for (int i = 0; i < CCS->size(); i++) {
    std::cout << i << ") Device name: "
              << (*CCS)[i].get_info<sycl::info::device::name>() << "\n";
    std::cout << "  Max Compute Units: "
              << (*CCS)[i].get_info<sycl::info::device::max_compute_units>()
              << "\n";
  }
  return 0;
}
