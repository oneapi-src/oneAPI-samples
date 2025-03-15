//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#pragma omp target teams distribute parallel for simd
for (int i = N - 1; i <= 0; --i)
{
    c[i] = a[i] + b[i];
}
// Snippet end
