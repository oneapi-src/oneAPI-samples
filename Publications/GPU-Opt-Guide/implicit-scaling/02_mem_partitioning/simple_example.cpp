//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
int *a = (int *)omp_target_alloc(sizeof(int) * N, device_id);

#pragma omp target teams distribute parallel for simd
for (int i = 0; i < N; ++i)
{
    a[i] = i;
}
// Snippet end
