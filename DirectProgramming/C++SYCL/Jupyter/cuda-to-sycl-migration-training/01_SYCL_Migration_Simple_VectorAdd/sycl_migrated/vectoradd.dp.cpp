//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#define N 16

//# kernel code to perform VectorAdd on GPU
void VectorAddKernel(float* A, float* B, float* C,
                     const sycl::nd_item<3> &item_ct1)
{
        C[item_ct1.get_local_id(2)] =
            A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
}

int main()
{
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        //# Print device name
        dpct::device_info dev;
        dpct::get_device(0).get_device_info(dev);
        std::cout << "Device: " << dev.get_name() << "\n";

        //# Initialize vectors on host
        float A[N] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        float B[N] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        float C[N] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        //# Allocate memory on device
        float *d_A, *d_B, *d_C;
        d_A = sycl::malloc_device<float>(N, q_ct1);
        d_B = sycl::malloc_device<float>(N, q_ct1);
        d_C = sycl::malloc_device<float>(N, q_ct1);

        //# copy vector data from host to device
        q_ct1.memcpy(d_A, A, N * sizeof(float));
        q_ct1.memcpy(d_B, B, N * sizeof(float));

        //# sumbit task to compute VectorAdd on device
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N), sycl::range<3>(1, 1, N)),
            [=](sycl::nd_item<3> item_ct1) {
                    VectorAddKernel(d_A, d_B, d_C, item_ct1);
            });

        //# copy result of vector data from device to host
        q_ct1.memcpy(C, d_C, N * sizeof(float)).wait();

        //# print result on host
        for (int i = 0; i < N; i++) std::cout<< C[i] << " ";
        std::cout << "\n";

        //# free allocation on device
        dpct::dpct_free(d_A, q_ct1);
        dpct::dpct_free(d_B, q_ct1);
        dpct::dpct_free(d_C, q_ct1);
        return 0;
}