//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
try {
  auto P = platform(gpu_selector{});
  auto RootDevices = P.get_devices();
  auto C = context(RootDevices);
  for (auto &D : RootDevices) {
    // Context has multiple root-devices, data can be shared across
    // multi - card(requires explict copying)
    auto Q = queue(C, D);
    Q.submit([&](handler &cgh) { ... });
  }
}
// Snippet end
