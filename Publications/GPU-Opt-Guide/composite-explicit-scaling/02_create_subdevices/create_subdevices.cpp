//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
try {
  vector<device> SubDevices = RootDevice.create_sub_devices<
      cl::sycl::info::partition_property::partition_by_affinity_domain>(
      cl::sycl::info::partition_affinity_domain::numa);
}
// Snippet end
