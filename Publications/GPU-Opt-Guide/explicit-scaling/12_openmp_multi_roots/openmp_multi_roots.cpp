//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
Int num_devices = omp_get_num_devices();

#pragma omp parallel for
for (int root_id = 0; root_id < num_devices; root_id++) {
#pragma omp target teams distribute parallel for device(root_id) map(…)
  for (int i = lb(root_id); I < ub(root_id); i++) {
    …
  }
}
// Snippet end
