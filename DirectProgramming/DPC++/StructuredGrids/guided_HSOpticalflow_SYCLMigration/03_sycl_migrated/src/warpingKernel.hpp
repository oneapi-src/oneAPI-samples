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
/// \brief warp image with a given displacement field, SYCL kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void WarpingKernel(int width, int height, int stride, const float *u,
                   const float *v, float *out,
                   accessor<cl::sycl::float4, 2, cl::sycl::access::mode::read,
                            sycl::access::target::image>
                       texToWarp,
                   cl::sycl::sampler texDesc, sycl::nd_item<3> item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float x = ((float)ix + u[pos]);
  float y = ((float)iy + v[pos]);

  auto inputCoord = float2(x, y);

  out[pos] = texToWarp.read(inputCoord, texDesc)[0];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field, SYCL kernel wrapper.
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  s   stride
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static void WarpImage(const float *src, int w, int h, int s, const float *u,
                      const float *v, float *out, queue q) {
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
  float *src_h = (float *)malloc(dataSize);
  q.memcpy(src_h, src, dataSize).wait();

  float *src_p = (float *)sycl::malloc_shared(h * s * sizeof(sycl::float4), q);
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int index = i * s + j;
      src_p[index * 4 + 0] = src_h[index];
      src_p[index * 4 + 1] = src_p[index * 4 + 2] = src_p[index * 4 + 3] = 0.f;
    }
  }
  auto texDescr = cl::sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::linear);

  auto texToWarp =
      cl::sycl::image<2>(src_p, cl::sycl::image_channel_order::rgba,
                         cl::sycl::image_channel_type::fp32, range<2>(w, h),
                         range<1>(s * sizeof(sycl::float4)));

  q.submit([&](sycl::handler &cgh) {
     auto texToWarp_acc =
         texToWarp.template get_access<cl::sycl::float4,
                                       cl::sycl::access::mode::read>(cgh);

     cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                      [=](sycl::nd_item<3> item_ct1) {
                        WarpingKernel(w, h, s, u, v, out, texToWarp_acc,
                                      texDescr, item_ct1);
                      });
   }).wait();
}
