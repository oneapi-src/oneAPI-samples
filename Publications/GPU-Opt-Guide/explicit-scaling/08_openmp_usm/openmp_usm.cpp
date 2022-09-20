//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
if (exists(“#pragma omp requires unified_shared_memory”)) {
    if (LIBOMPTARGET_USM_HOST_MEM == 1)
        return "host memory";
    else
        return "shared memory";
} else {
     return "device memory";
}
// Snippet end
