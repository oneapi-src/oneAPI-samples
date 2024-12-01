//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  // Find all GPU devices
  auto devices = sycl::platform(sycl::gpu_selector_v).get_devices();
  for (size_t n = 0; n < devices.size(); n++) {
    std::cout << "\nGPU" << n << ": "
              << devices[n].get_info<sycl::info::device::name>() << " ("
              << devices[n].get_info<sycl::info::device::max_compute_units>()
              << ")\n";
    std::vector<sycl::device> subdevices;
    std::vector<sycl::device> subsubdevices;
    auto part_prop =
        devices[n].get_info<sycl::info::device::partition_properties>();
    if (part_prop.empty()) {
      std::cout << "No partition_properties\n";
    } else {
      for (size_t i = 0; i < part_prop.size(); i++) {
        // Check if device can be partitioned into Tiles
        if (part_prop[i] ==
            sycl::info::partition_property::partition_by_affinity_domain) {
          auto sub_devices =
              devices[n]
                  .create_sub_devices<sycl::info::partition_property::
                                          partition_by_affinity_domain>(
                      sycl::info::partition_affinity_domain::numa);
          for (size_t j = 0; j < sub_devices.size(); j++) {
            subdevices.push_back(sub_devices[j]);
            std::cout << "\ntile" << j << ": "
                      << subdevices[j].get_info<sycl::info::device::name>()
                      << " ("
                      << subdevices[j]
                             .get_info<sycl::info::device::max_compute_units>()
                      << ")\n";
            auto part_prop1 =
                subdevices[j]
                    .get_info<sycl::info::device::partition_properties>();
            if (part_prop1.empty()) {
              std::cout << "No partition_properties\n";
            } else {
              for (size_t i = 0; i < part_prop1.size(); i++) {
                // Check if Tile can be partitioned into Slices (CCS)
                if (part_prop1[i] == sycl::info::partition_property::
                                         ext_intel_partition_by_cslice) {
                  auto sub_devices =
                      subdevices[j]
                          .create_sub_devices<
                              sycl::info::partition_property::
                                  ext_intel_partition_by_cslice>();
                  for (size_t k = 0; k < sub_devices.size(); k++) {
                    subsubdevices.push_back(sub_devices[k]);
                    std::cout
                        << "slice" << k << ": "
                        << subsubdevices[k].get_info<sycl::info::device::name>()
                        << " ("
                        << subsubdevices[k]
                               .get_info<
                                   sycl::info::device::max_compute_units>()
                        << ")\n";
                  }
                  break;
                } else {
                  std::cout << "No ext_intel_partition_by_cslice\n";
                }
              }
            }
          }
          break;
          // Check if device can be partitioned into Slices (CCS)
        } else if (part_prop[i] == sycl::info::partition_property::
                                       ext_intel_partition_by_cslice) {
          auto sub_devices =
              devices[n]
                  .create_sub_devices<sycl::info::partition_property::
                                          ext_intel_partition_by_cslice>();
          for (size_t k = 0; k < sub_devices.size(); k++) {
            subsubdevices.push_back(sub_devices[k]);
            std::cout << "slice" << k << ": "
                      << subsubdevices[k].get_info<sycl::info::device::name>()
                      << " ("
                      << subsubdevices[k]
                             .get_info<sycl::info::device::max_compute_units>()
                      << ")\n";
          }
          break;
        } else {
          std::cout << "No ext_intel_partition_by_cslice or "
                       "partition_by_affinity_domain\n";
        }
      }
    }
  }
  return 0;
}
