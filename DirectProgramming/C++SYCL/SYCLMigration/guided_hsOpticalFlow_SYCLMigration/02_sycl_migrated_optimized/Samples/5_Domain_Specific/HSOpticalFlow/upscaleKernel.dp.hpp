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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, CUDA kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void UpscaleKernel(int width, int height, int stride, float scale, float *out,
                  sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                            sycl::access::target::image>
                       texCoarse_acc,
                   sycl::sampler texDesc,
                   const sycl::nd_item<3> &item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  if (ix >= width || iy >= height) return;

  float x = ((float)ix - 0.5f) * 0.5f;
  float y = ((float)iy - 0.5f) * 0.5f;

  auto inputCoord = sycl::float2(x, y);

  // exploit hardware interpolation
  // and scale interpolated vector to match next pyramid level resolution
  out[ix + iy * stride] = texCoarse_acc.read(inputCoord, texDesc)[0] * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static void Upscale(const float *src, float *pI0_h, float *I0_h, float *src_p, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride, float scale,
                    float *out, sycl::queue q) {
  sycl::range<3> threads(1, 8, 32);
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));

  int dataSize = stride * height * sizeof(float);
  q.memcpy(I0_h, src, dataSize).wait();

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * stride + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;
    }
  }
  q.memcpy(src_p, pI0_h, height * stride * sizeof(sycl::float4)).wait();

  auto texDescr = sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::linear);

  auto texCoarse = sycl::image<2>(
      src_p, sycl::image_channel_order::rgba,
      sycl::image_channel_type::fp32, sycl::range<2>(width, height),
      sycl::range<1>(stride * sizeof(sycl::float4)));
  
  q.submit([&](sycl::handler &cgh) {
    auto texCoarse_acc =
         texCoarse.template get_access<sycl::float4,
                                       sycl::access::mode::read>(cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       UpscaleKernel(newWidth, newHeight, newStride, scale, out,
                                     texCoarse_acc, texDescr, item_ct1);
                     });
  });
}