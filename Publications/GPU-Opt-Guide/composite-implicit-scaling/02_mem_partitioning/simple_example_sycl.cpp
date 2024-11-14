//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
int *a = sycl::malloc_device<int>(N, q);

q.parallel_for(N, [=](auto i) {
    a[i] = i;
});
// Snippet end
