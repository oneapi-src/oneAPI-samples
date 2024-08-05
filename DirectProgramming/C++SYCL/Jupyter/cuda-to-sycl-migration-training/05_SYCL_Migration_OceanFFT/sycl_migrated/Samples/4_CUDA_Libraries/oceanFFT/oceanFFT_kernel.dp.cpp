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

///////////////////////////////////////////////////////////////////////////////
#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>
#include <sycl/sycl.hpp>

// Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b) { return (a + (b - 1)) / b; }

// complex math functions
sycl::float2 conjugate(sycl::float2 arg) {
  return sycl::float2(arg.x(), -arg.y());
}

sycl::float2 complex_exp(float arg) {
  return sycl::float2(sycl::cos(arg), sycl::sin(arg));
}

sycl::float2 complex_add(sycl::float2 a, sycl::float2 b) {
  return sycl::float2(a.x() + b.x(), a.y() + b.y());
}

sycl::float2 complex_mult(sycl::float2 ab, sycl::float2 cd) {
  return sycl::float2(ab.x() * cd.x() - ab.y() * cd.y(),
                      ab.x() * cd.y() + ab.y() * cd.x());
}

// generate wave heightfield at time t based on initial heightfield and
// dispersion relationship
void generateSpectrumKernel(sycl::float2 *h0, sycl::float2 *ht,
                            unsigned int in_width, unsigned int out_width,
                            unsigned int out_height, float t, float patchSize,
                            const sycl::nd_item<3> &item_ct1) {
  unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
  unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                   item_ct1.get_local_id(1);
  unsigned int in_index = y * in_width + x;
  unsigned int in_mindex =
      (out_height - y) * in_width + (out_width - x);  // mirrored
  unsigned int out_index = y * out_width + x;

  // calculate wave vector
  sycl::float2 k;
  k.x() = (-(int)out_width / 2.0f + x) * (2.0f * 3.141592654F / patchSize);
  k.y() = (-(int)out_width / 2.0f + y) * (2.0f * 3.141592654F / patchSize);

  // calculate dispersion w(k)
  float k_len = sycl::sqrt(k.x() * k.x() + k.y() * k.y());
  float w = sycl::sqrt(9.81f * k_len);

  if ((x < out_width) && (y < out_height)) {
    sycl::float2 h0_k = h0[in_index];
    sycl::float2 h0_mk = h0[in_mindex];

    // output frequency-space complex values
    ht[out_index] =
        complex_add(complex_mult(h0_k, complex_exp(w * t)),
                    complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
    // ht[out_index] = h0_k;
  }
}

// update height map values based on output of FFT
void updateHeightmapKernel(float *heightMap, sycl::float2 *ht,
                           unsigned int width,
                           const sycl::nd_item<3> &item_ct1) {
  unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
  unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                   item_ct1.get_local_id(1);
  unsigned int i = y * width + x;

  // cos(pi * (m1 + m2))
  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].x() * sign_correction;
}

// update height map values based on output of FFT
void updateHeightmapKernel_y(float *heightMap, sycl::float2 *ht,
                             unsigned int width,
                             const sycl::nd_item<3> &item_ct1) {
  unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
  unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                   item_ct1.get_local_id(1);
  unsigned int i = y * width + x;

  // cos(pi * (m1 + m2))
  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].y() * sign_correction;
}

// generate slope by partial differences in spatial domain
void calculateSlopeKernel(float *h, sycl::float2 *slopeOut, unsigned int width,
                          unsigned int height,
                          const sycl::nd_item<3> &item_ct1) {
  unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
  unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                   item_ct1.get_local_id(1);
  unsigned int i = y * width + x;

  sycl::float2 slope = sycl::float2(0.0f, 0.0f);

  if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1)) {
    slope.x() = h[i + 1] - h[i - 1];
    slope.y() = h[i + width] - h[i - width];
  }

  slopeOut[i] = slope;
}

// wrapper functions
extern "C" void cudaGenerateSpectrumKernel(sycl::float2 *d_h0,
                                           sycl::float2 *d_ht,
                                           unsigned int in_width,
                                           unsigned int out_width,
                                           unsigned int out_height,
                                           float animTime, float patchSize) {
  sycl::range<3> block(1, 8, 8);
  sycl::range<3> grid(1, cuda_iDivUp(out_height, block[1]),
                      cuda_iDivUp(out_width, block[2]));

  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
        generateSpectrumKernel(d_h0, d_ht, in_width, out_width, out_height,
                               animTime, patchSize, item_ct1);
      });
}

extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap,
                                          sycl::float2 *d_ht,
                                          unsigned int width,
                                          unsigned int height, bool autoTest) {
  sycl::range<3> block(1, 8, 8);
  sycl::range<3> grid(1, cuda_iDivUp(height, block[1]),
                      cuda_iDivUp(width, block[2]));
  if (autoTest) {
    dpct::get_default_queue().parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          updateHeightmapKernel_y(d_heightMap, d_ht, width, item_ct1);
        });
  } else {
    dpct::get_default_queue().parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          updateHeightmapKernel(d_heightMap, d_ht, width, item_ct1);
        });
  }
}

extern "C" void cudaCalculateSlopeKernel(float *hptr, sycl::float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height) {
  sycl::range<3> block(1, 8, 8);
  sycl::range<3> grid2(1, cuda_iDivUp(height, block[1]),
                       cuda_iDivUp(width, block[2]));

  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(grid2 * block, block), [=](sycl::nd_item<3> item_ct1) {
        calculateSlopeKernel(hptr, slopeOut, width, height, item_ct1);
      });
}
