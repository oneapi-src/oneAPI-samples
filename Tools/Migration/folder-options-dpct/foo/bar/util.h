//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _UTIL_H
#define _UTIL_H
#include <stdio.h>

typedef int myint;

__device__ int mymax(myint a, myint b) {
    return (a>b)? a : b;
}

#endif
