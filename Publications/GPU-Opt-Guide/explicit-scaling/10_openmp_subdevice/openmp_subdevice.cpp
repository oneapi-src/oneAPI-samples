//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#define DEVKIND 0 // TILE

int root_id = omp_get_default_device();

#pragma omp parallel for
for (int id = 0; id < NUM_SUBDEVICES; ++id) {
#pragma omp target teams distribute parallel for device(root_id)               \
    subdevice(DEVKIND, id) map(…)
  for (int i = lb(id), i < ub(id); i++) {
    …
  }
}
// Snippet end
