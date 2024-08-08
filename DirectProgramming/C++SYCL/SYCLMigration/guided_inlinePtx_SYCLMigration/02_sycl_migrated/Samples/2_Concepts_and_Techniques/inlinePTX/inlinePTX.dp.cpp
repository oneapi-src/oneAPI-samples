/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Demonstration of inline PTX (assembly language) usage in CUDA kernels
 */

// System includes
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <assert.h>

// CUDA runtime

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

void sequence_gpu(int *d_ptr, int length, const sycl::nd_item<3> &item_ct1)
{
    int elemID = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

    if (elemID < length)
    {
        unsigned int laneid;
        //This command gets the lane ID within the current warp
        laneid = item_ct1.get_sub_group().get_local_linear_id();
        d_ptr[elemID] = laneid;
    }
}


void sequence_cpu(int *h_ptr, int length)
{
    for (int elemID=0; elemID<length; elemID++)
    {
        h_ptr[elemID] = elemID % 32;
    }
}

int main(int argc, char **argv)
{
    printf("SYCL inline PTX assembler sample\n");

    const int N = 1000;

    int dev = 0;
 
    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    int *d_ptr;
    DPCT_CHECK_ERROR(
        d_ptr = sycl::malloc_device<int>(N, dpct::get_in_order_queue()));

    int *h_ptr;
    DPCT_CHECK_ERROR(
        h_ptr = sycl::malloc_host<int>(N, dpct::get_in_order_queue()));

    sycl::range<3> cudaBlockSize(1, 1, 256);
    sycl::range<3> cudaGridSize(1, 1,
                                (N + cudaBlockSize[2] - 1) / cudaBlockSize[2]);

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cudaGridSize * cudaBlockSize, cudaBlockSize),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            sequence_gpu(d_ptr, N, item_ct1);
        });

    DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());

    sequence_cpu(h_ptr, N);

    int *h_d_ptr;
    DPCT_CHECK_ERROR(
        h_d_ptr = sycl::malloc_host<int>(N, dpct::get_in_order_queue()));
    
    DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(h_d_ptr, d_ptr, N * sizeof(int))
                             .wait());

    bool bValid = true;

    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i] != h_d_ptr[i])
        {
            bValid = false;
        }
    }

    printf("Test %s.\n", bValid ? "Successful" : "Failed");

    
    DPCT_CHECK_ERROR(sycl::free(d_ptr, dpct::get_in_order_queue()));
    DPCT_CHECK_ERROR(sycl::free(h_ptr, dpct::get_in_order_queue()));
    DPCT_CHECK_ERROR(sycl::free(h_d_ptr, dpct::get_in_order_queue()));

    return bValid ? EXIT_SUCCESS: EXIT_FAILURE;
}
