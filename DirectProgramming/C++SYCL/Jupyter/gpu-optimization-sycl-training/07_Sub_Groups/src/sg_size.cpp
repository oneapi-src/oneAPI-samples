//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()<< "\n";

  q.submit([&](auto &h) {
    sycl::stream out(65536, 256, h);
    h.parallel_for(sycl::nd_range<1>(32,32), [=](sycl::nd_item<1> it) {
         int groupId = it.get_group(0);
         int globalId = it.get_global_linear_id();
         auto sg = it.get_sub_group();
         int sgSize = sg.get_local_range()[0];
         int sgGroupId = sg.get_group_id()[0];
         int sgId = sg.get_local_id()[0];

         out << "globalId = " << sycl::setw(2) << globalId
             << " groupId = " << groupId
             << " sgGroupId = " << sgGroupId << " sgId = " << sgId
             << " sgSize = " << sycl::setw(2) << sgSize
             << sycl::endl;
    });
  });

  q.wait();
  return 0;
}
