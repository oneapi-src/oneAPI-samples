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

#ifndef FWT_KERNEL_CUH
#define FWT_KERNEL_CUH
#ifndef fwt_kernel_cuh
#define fwt_kernel_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 10

void fwtBatch1Kernel(float *d_Output, float *d_Input, int log2N,
                     const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  const int N = 1 << log2N;
  const int base = item_ct1.get_group(2) << log2N;

  //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
  auto s_data = (float *)dpct_local;
  float *d_Src = d_Input + base;
  float *d_Dst = d_Output + base;

  for (int pos = item_ct1.get_local_id(2); pos < N;
       pos += item_ct1.get_local_range(2)) {
    s_data[pos] = d_Src[pos];
  }

  // Main radix-4 stages
  const int pos = item_ct1.get_local_id(2);

  for (int stride = N >> 2; stride > 0; stride >>= 2) {
    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;
    item_ct1.barrier();
    float D0 = s_data[i0];
    float D1 = s_data[i1];
    float D2 = s_data[i2];
    float D3 = s_data[i3];

    float T;
    T = D0;
    D0 = D0 + D2;
    D2 = T - D2;
    T = D1;
    D1 = D1 + D3;
    D3 = T - D3;
    T = D0;
    s_data[i0] = D0 + D1;
    s_data[i1] = T - D1;
    T = D2;
    s_data[i2] = D2 + D3;
    s_data[i3] = T - D3;
  }

  // Do single radix-2 stage for odd power of two
  if (log2N & 1) {
   
    item_ct1.barrier();

    for (int pos = item_ct1.get_local_id(2); pos < N / 2;
         pos += item_ct1.get_local_range(2)) {
      int i0 = pos << 1;
      int i1 = i0 + 1;

      float D0 = s_data[i0];
      float D1 = s_data[i1];
      s_data[i0] = D0 + D1;
      s_data[i1] = D0 - D1;
    }
  }

  item_ct1.barrier();

  for (int pos = item_ct1.get_local_id(2); pos < N;
       pos += item_ct1.get_local_range(2)) {
    d_Dst[pos] = s_data[pos];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
void fwtBatch2Kernel(float *d_Output, float *d_Input, int stride,
                     const sycl::nd_item<3> &item_ct1) {
  const int pos = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
  const int N = item_ct1.get_local_range(2) * item_ct1.get_group_range(2) * 4;

  float *d_Src = d_Input + item_ct1.get_group(1) * N;
  float *d_Dst = d_Output + item_ct1.get_group(1) * N;

  int lo = pos & (stride - 1);
  int i0 = ((pos - lo) << 2) + lo;
  int i1 = i0 + stride;
  int i2 = i1 + stride;
  int i3 = i2 + stride;

  float D0 = d_Src[i0];
  float D1 = d_Src[i1];
  float D2 = d_Src[i2];
  float D3 = d_Src[i3];

  float T;
  T = D0;
  D0 = D0 + D2;
  D2 = T - D2;
  T = D1;
  D1 = D1 + D3;
  D3 = T - D3;
  T = D0;
  d_Dst[i0] = D0 + D1;
  d_Dst[i1] = T - D1;
  T = D2;
  d_Dst[i2] = D2 + D3;
  d_Dst[i3] = T - D3;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(float *d_Data, int M, int log2N) {
  const int THREAD_N = 256;

  int N = 1 << log2N;
  sycl::range<3> grid(1, M, (1 << log2N) / (4 * THREAD_N));

  for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2) {
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      int N_ct2 = N / 4;

      cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, THREAD_N),
                                         sycl::range<3>(1, 1, THREAD_N)),
                       [=](sycl::nd_item<3> item_ct1) {
                         fwtBatch2Kernel(d_Data, d_Data, N_ct2, item_ct1);
                       });
    });
  }

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
   
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(N * sizeof(float)), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, M) * sycl::range<3>(1, 1, N / 4),
                          sycl::range<3>(1, 1, N / 4)),
        [=](sycl::nd_item<3> item_ct1) {
          fwtBatch1Kernel(
              d_Data, d_Data, log2N, item_ct1,
              dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });

}

////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
void modulateKernel(float *d_A, float *d_B, int N,
                    const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int numThreads = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
  float rcpN = 1.0f / (float)N;

  for (int pos = tid; pos < N; pos += numThreads) {
    d_A[pos] *= d_B[pos] * rcpN;
  }
}

// Interface to modulateKernel()
void modulateGPU(float *d_A, float *d_B, int N) {
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 128) * sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
      [=](sycl::nd_item<3> item_ct1) {
        modulateKernel(d_A, d_B, N, item_ct1);
      });
}

#endif
#endif
