//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cuda.h>
#include <iostream>
#include <vector>
#define N 16

//# kernel code to perform VectorAdd on GPU
__global__ void VectorAddKernel(float* A, float* B, float* C)
{
        C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main()
{
        //# Initialize vectors on host
        float A[N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float B[N] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        float C[N] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        //# Allocate memory on device
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N*sizeof(float));
        cudaMalloc(&d_B, N*sizeof(float));
        cudaMalloc(&d_C, N*sizeof(float));

        //# copy vector data from host to device
        cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);

        //# sumbit task to compute VectorAdd on device
        VectorAddKernel<<<1, N>>>(d_A, d_B, d_C);

        //# copy result of vector data from device to host
        cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

        //# print result on host
        for (int i = 0; i < N; i++) std::cout<< C[i] << " ";
        std::cout << "\n";

        //# free allocation on device
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 0;
}