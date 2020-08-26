//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
extern "C" unsigned RtlByteswap(unsigned x) { return x << 16 | x >> 16; }
