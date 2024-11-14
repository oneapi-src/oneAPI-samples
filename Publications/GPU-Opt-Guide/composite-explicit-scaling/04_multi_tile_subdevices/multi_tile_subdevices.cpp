//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
try {
  vector<device> SubDevices = ...;
  auto C = context(SubDevices);
  for (auto &D : SubDevices) {
    // All queues share the same context, data can be shared across
    // queues.
    auto Q = queue(C, D);
    Q.submit([&](handler &cgh) { ... });
  }
}
// Snippet end
