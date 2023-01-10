//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

__kernel void vector_add(__global const float *x, __global const float *y,
                         __global float *restrict z) {
  // get index of the work item
  int index = get_global_id(0);
  // add the vector elements
  z[index] = x[index] + y[index];
}
