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
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void DownscaleKernel(int width, int height, int stride, float *out,
                     sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                              sycl::access::target::image>
                         tex_acc,
                     sycl::sampler texDesc,
                     const sycl::nd_item<3> &item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  if (ix >= width || iy >= height) {
    return;
  }

  int srcx = ix * 2;
  int srcy = iy * 2;

  auto inputCoords1 = sycl::float2(srcx + 0, srcy + 0);
  auto inputCoords2 = sycl::float2(srcx + 0, srcy + 1);
  auto inputCoords3 = sycl::float2(srcx + 1, srcy + 0);
  auto inputCoords4 = sycl::float2(srcx + 1, srcy + 1);

  out[ix + iy * stride] = 0.25f * (tex_acc.read(inputCoords1, texDesc)[0] +
                                   tex_acc.read(inputCoords2, texDesc)[0] +
                                   tex_acc.read(inputCoords3, texDesc)[0] +
                                   tex_acc.read(inputCoords4, texDesc)[0]);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static void Downscale(const float *src, float *pI0_h, float *I0_h, float *src_p, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out, sycl::queue q) {
  sycl::range<3> threads(1, 8, 32);
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));

  int dataSize = height * stride * sizeof(float);

  q.memcpy(I0_h, src, dataSize).wait();

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * stride + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;
    }
  }

  q.memcpy(src_p, pI0_h, height * width * sizeof(sycl::float4)).wait();
  
  auto texDescr = sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest);

  auto texFine = sycl::image<2>(src_p, sycl::image_channel_order::rgba,
                                    sycl::image_channel_type::fp32,
                                    sycl::range<2>(width, height),
                                    sycl::range<1>(stride * sizeof(sycl::float4)));
  
  q.submit([&](sycl::handler &cgh) {
    auto tex_acc =
         texFine.template get_access<sycl::float4,
                                     sycl::access::mode::read>(cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       DownscaleKernel(newWidth, newHeight, newStride, out,
                                       tex_acc, texDescr, item_ct1);
                     });
  });
}