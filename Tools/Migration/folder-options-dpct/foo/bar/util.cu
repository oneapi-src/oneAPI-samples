//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cuda_runtime.h>
#include "util.h"

__global__ void kernel_util(myint a, myint b) {
    myint c= mymax(a,b);
    printf("kernel_util,%d\n", c);
}

void run_util(){
    kernel_util<<<1,1>>>(1,2);
}
