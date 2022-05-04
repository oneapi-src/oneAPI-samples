//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 64; // global size
static constexpr size_t B = 64; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  q.submit([&](handler &h) {
    //# setup sycl stream class to print standard output from device code
    auto out = stream(1024, 768, h);

    //# nd-range kernel
    h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
      //# get sub_group handle
      auto sg = item.get_sub_group();

      //# query sub_group and print sub_group info once per sub_group
      if (sg.get_local_id()[0] == 0) {
        out << "sub_group id: " << sg.get_group_id()[0] << " of "
            << sg.get_group_range()[0] << ", size=" << sg.get_local_range()[0]
            << "\n";
      }
    });
  }).wait();
}
