//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
try {
  vector<device> SubDevices = ...;
  for (auto &D : SubDevices) {
    // Each queue is in its own context, no data sharing across them.
    auto Q = queue(D);
    Q.submit([&](handler &cgh) { ... });
  }
}
// Snippet end
