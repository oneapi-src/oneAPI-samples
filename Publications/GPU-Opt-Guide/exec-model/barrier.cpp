//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
auto command_group =
    [&](auto &cgh) {
      cgh.parallel_for(nd_range(sycl::range(64, 64, 128), // global range
                                sycl::range(1, R, 128)    // local range
                                ),
                       [=](sycl::nd_item<3> item) {
                         // (kernel code)
                         // Internal synchronization
                         item.barrier(access::fence_space::global_space);
                         // (kernel code)
                       })
    }
// Snippet end
