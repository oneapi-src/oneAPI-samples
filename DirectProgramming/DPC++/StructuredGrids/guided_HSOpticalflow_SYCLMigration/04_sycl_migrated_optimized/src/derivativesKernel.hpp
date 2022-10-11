//=========================================================
// Modifications Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
//=========================================================

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

#include <CL/sycl.hpp>

#include "common.h"

using namespace sycl;

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// SYCL kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////

void ComputeDerivativesKernel(
    int width, int height, int stride, float *Ix, float *Iy, float *Iz,
    accessor<cl::sycl::float4, 2, cl::sycl::access::mode::read,
             sycl::access::target::image>
        texSource,
    accessor<cl::sycl::float4, 2, cl::sycl::access::mode::read,
             sycl::access::target::image>
        texTarget,
    cl::sycl::sampler texDesc, sycl::nd_item<3> item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float t0, t1;

  auto x_inputCoords1 = float2(ix - 2.0f, iy);
  auto x_inputCoords2 = float2(ix - 1.0f, iy);
  auto x_inputCoords3 = float2(ix + 1.0f, iy);
  auto x_inputCoords4 = float2(ix + 2.0f, iy);

  t0 = texSource.read(x_inputCoords1, texDesc)[0];
  t0 -= texSource.read(x_inputCoords2, texDesc)[0] * 8.0f;
  t0 += texSource.read(x_inputCoords3, texDesc)[0] * 8.0f;
  t0 -= texSource.read(x_inputCoords4, texDesc)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(x_inputCoords1, texDesc)[0];
  t1 -= texTarget.read(x_inputCoords2, texDesc)[0] * 8.0f;
  t1 += texTarget.read(x_inputCoords3, texDesc)[0] * 8.0f;
  t1 -= texTarget.read(x_inputCoords4, texDesc)[0];
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;

  // t derivative
  auto inputCoord = float2(ix, iy);
  Iz[pos] = texTarget.read(inputCoord, texDesc)[0] -
            texSource.read(inputCoord, texDesc)[0];

  // y derivative
  auto y_inputCoords1 = float2(ix, iy - 2.0f);
  auto y_inputCoords2 = float2(ix, iy - 1.0f);
  auto y_inputCoords3 = float2(ix, iy + 1.0f);
  auto y_inputCoords4 = float2(ix, iy + 2.0f);

  t0 = texSource.read(y_inputCoords1, texDesc)[0];
  t0 -= texSource.read(y_inputCoords2, texDesc)[0] * 8.0f;
  t0 += texSource.read(y_inputCoords3, texDesc)[0] * 8.0f;
  t0 -= texSource.read(y_inputCoords4, texDesc)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(y_inputCoords1, texDesc)[0];
  t1 -= texTarget.read(y_inputCoords2, texDesc)[0] * 8.0f;
  t1 += texTarget.read(y_inputCoords3, texDesc)[0] * 8.0f;
  t1 -= texTarget.read(y_inputCoords4, texDesc)[0];
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1) * 0.5f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0, const float *I1, float *pI0_h,
                               float *pI1_h, float *I0_h, float *I1_h,
                               float *src_d0, float *src_d1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz,
                               queue q) {
  sycl::range<3> threads(1, 6, 32);
  auto max_wg_size =
      q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
  if (max_wg_size < 6 * 32) {
    threads[0] = 1;
    threads[2] = 32;
    threads[1] = max_wg_size / threads[2];
  }
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  int dataSize = s * h * sizeof(float);

  q.memcpy(I0_h, I0, dataSize);
  q.memcpy(I1_h, I1, dataSize);

  q.wait();

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int index = i * s + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;
    }
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int index = i * s + j;
      pI1_h[index * 4 + 0] = I1_h[index];
      pI1_h[index * 4 + 1] = pI1_h[index * 4 + 2] = pI1_h[index * 4 + 3] = 0.f;
    }
  }
  q.memcpy(src_d0, pI0_h, s * h * sizeof(sycl::float4));
  q.memcpy(src_d1, pI1_h, s * h * sizeof(sycl::float4));

  q.wait();

  auto texDescr = cl::sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest);

  auto texSource =
      cl::sycl::image<2>(src_d0, cl::sycl::image_channel_order::rgba,
                         cl::sycl::image_channel_type::fp32, range<2>(w, h),
                         range<1>(s * sizeof(sycl::float4)));

  auto texTarget =
      cl::sycl::image<2>(src_d1, cl::sycl::image_channel_order::rgba,
                         cl::sycl::image_channel_type::fp32, range<2>(w, h),
                         range<1>(s * sizeof(sycl::float4)));

  q.submit([&](sycl::handler &cgh) {
     auto texSource_acc =
         texSource.template get_access<cl::sycl::float4,
                                       cl::sycl::access::mode::read>(cgh);
     auto texTarget_acc =
         texTarget.template get_access<cl::sycl::float4,
                                       cl::sycl::access::mode::read>(cgh);

     cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                      [=](sycl::nd_item<3> item_ct1) {
                        ComputeDerivativesKernel(w, h, s, Ix, Iy, Iz,
                                                 texSource_acc, texTarget_acc,
                                                 texDescr, item_ct1);
                      });
   }).wait();
}
