//==============================================================
// Copyright � 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

constexpr size_t kHalfLength = 8;
constexpr float dxyz = 50.0f;
constexpr float dt = 0.002f;

#define STENCIL_LOOKUP(ir)                                          \
  (coeff[ir] * ((ptr_prev[ix + ir] + ptr_prev[ix - ir]) +           \
                (ptr_prev[ix + ir * n1] + ptr_prev[ix - ir * n1]) + \
                (ptr_prev[ix + ir * dimn1n2] + ptr_prev[ix - ir * dimn1n2])))


#define KERNEL_STENCIL_LOOKUP(x)                                          \
  coeff[x] * (tab[l_idx + x] + tab[l_idx - x] + front[x] + back[x - 1]    \
           + tab[l_idx + l_n3 * x] + tab[l_idx - l_n3 * x]) 