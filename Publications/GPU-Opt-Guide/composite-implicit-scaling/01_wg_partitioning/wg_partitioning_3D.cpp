//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#pragma omp target teams distribute parallel for simd collapse(3)
for (int z = 0; z < nz; ++z)
{
    for (int y = 0; y < ny; ++y)
    {
        for (int x = 0; x < nx; ++x)
        {
            //
        }
    }
}
// Snippet end
