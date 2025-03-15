//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
try {
  // The queue is attached to the root-device, driver distributes to
  // sub - devices, if any.
  auto D = device(gpu_selector{});
  auto Q = queue(D);
  Q.submit([&](handler &cgh) { ... });
}
// Snippet end
