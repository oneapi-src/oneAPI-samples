//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
auto command_group =
    [&](auto &cgh) {
      // local memory variables shared among work items
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          myLocal(sycl::range(R), cgh);
      cgh.parallel_for(nd_range(sycl::range<3>(64, 64, 128), // global range
                                sycl::range<3>(1, R, 128)    // local range
                                ),
                       [=](ngroup<3> myGroup) {
                         // (work group code)
                         myLocal[myGroup.get_local_id()[1]] = ...
                       })
    }
// Snippet end
