//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
auto command_group =
    [&](auto &cgh) {
      cgh.parallel_for(sycl::range<3>(64, 64, 64), // global range
                       [=](item<3> it) {
                         // (kernel code)
                       })
    }
// Snippet end
