//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "offload.h"

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

void do_work(std::vector<int> &ans) {
  // # define queue which has default device associated for offload
  sycl::queue q;
  std::cout << "Using device: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // # Unified Shared Memory Allocation enables data access on host and device
  int *data = sycl::malloc_shared<int>(ans.size(), q);
  assert(data);

  // # Initialization
  for (int i = 0; i < ans.size(); i++) {
    data[i] = i;
  }

  // # Offload parallel computation to device
  size_t n_items = ans.size();
  q.parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> i) {
     data[i] *= 2;
   }).wait();

  for (int i = 0; i < ans.size(); ++i) {
    ans[i] = data[i];
  }
  sycl::free(data, q);
}