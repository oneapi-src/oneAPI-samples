//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <stdio.h>
#include <cuda_runtime.h>

extern void run_util();

__global__ void kernel_main(int n) {
    printf("kernel_main!\n");
}

int main(){

    kernel_main<<<1, 1>>>(1);
    cudaDeviceSynchronize();

    run_util();
    cudaDeviceSynchronize();

    return 0;
}
