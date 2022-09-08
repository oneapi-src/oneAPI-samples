//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
range<3> global{nz, ny, nx};
range<3> local{1, 1, 16};

cgh.parallel_for(nd_range<3>(global, local), [=](nd_item<3> item) {
    //
});
// Snippet end
