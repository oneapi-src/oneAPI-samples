//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
int root_id = omp_get_default_device();

#pragma omp target teams distribute parallel for device(root_id) map(…)
for (int i = 0, i < N; i++) {
  …
}
// Snippet end
