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

#include "common.h"

using namespace sycl;

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, SYCL kernel.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// Template parameters are describe CTA size
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
template <int bx, int by>
void JacobiIteration(const float *du0, const float *dv0, const float *Ix,
                     const float *Iy, const float *Iz, int w, int h, int s,
                     float alpha, float *du1, float *dv1,
                     sycl::nd_item<3> item_ct1, volatile float *du,
                     volatile float *dv) {
  // Handle to thread block group

  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

  // position within global memory array
  const int pos = sycl::min(ix, (int)(w - 1)) + sycl::min(iy, (int)(h - 1)) * s;

  // position within shared memory array
  const int shMemPos =
      item_ct1.get_local_id(2) + 1 + (item_ct1.get_local_id(1) + 1) * (bx + 2);

  // Load data to shared memory.
  // load tile being processed
  du[shMemPos] = du0[pos];
  dv[shMemPos] = dv0[pos];

  // load necessary neighbouring elements
  // We clamp out-of-range coordinates.
  // It is equivalent to mirroring
  // because we access data only one step away from borders.
  if (item_ct1.get_local_id(1) == 0) {
    // beginning of the tile
    const int bsx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    const int bsy = item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
    // element position within matrix
    int x, y;
    // element position within linear array
    // gm - global memory
    // sm - shared memory
    int gmPos, smPos;

    x = sycl::min((unsigned int)(bsx + item_ct1.get_local_id(2)),
                  (unsigned int)(w - 1));
    // row just below the tile
    y = sycl::max((int)(bsy - 1), 0);
    gmPos = y * s + x;
    smPos = item_ct1.get_local_id(2) + 1;
    du[smPos] = du0[gmPos];
    dv[smPos] = dv0[gmPos];

    // row above the tile
    y = sycl::min((int)(bsy + by), (int)(h - 1));
    smPos += (by + 1) * (bx + 2);
    gmPos = y * s + x;
    du[smPos] = du0[gmPos];
    dv[smPos] = dv0[gmPos];
  } else if (item_ct1.get_local_id(1) == 1) {
    // beginning of the tile
    const int bsx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    const int bsy = item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
    // element position within matrix
    int x, y;
    // element position within linear array
    // gm - global memory
    // sm - shared memory
    int gmPos, smPos;

    y = sycl::min((unsigned int)(bsy + item_ct1.get_local_id(2)),
                  (unsigned int)(h - 1));
    // column to the left
    x = sycl::max((int)(bsx - 1), 0);
    smPos = bx + 2 + item_ct1.get_local_id(2) * (bx + 2);
    gmPos = x + y * s;

    // check if we are within tile
    if (item_ct1.get_local_id(2) < by) {
      du[smPos] = du0[gmPos];
      dv[smPos] = dv0[gmPos];
      // column to the right
      x = sycl::min((int)(bsx + bx), (int)(w - 1));
      gmPos = y * s + x;
      smPos += bx + 1;
      du[smPos] = du0[gmPos];
      dv[smPos] = dv0[gmPos];
    }
  }

  group_barrier(item_ct1.get_group());

  if (ix >= w || iy >= h) return;

  // now all necessary data are loaded to shared memory
  int left, right, up, down;
  left = shMemPos - 1;
  right = shMemPos + 1;
  up = shMemPos + bx + 2;
  down = shMemPos - bx - 2;

  float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
  float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

  float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) /
               (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

  du1[pos] = sumU - Ix[pos] * frac;
  dv1[pos] = sumV - Iy[pos] * frac;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, SYCL kernel wrapper.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
static void SolveForUpdate(const float *du0, const float *dv0, const float *Ix,
                           const float *Iy, const float *Iz, int w, int h,
                           int s, float alpha, float *du1, float *dv1,
                           queue q) {
  // CTA size
  sycl::range<3> threads(1, 6, 32);
  auto max_wg_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (max_wg_size < 6 * 32) {
    threads[0] = 1;
    threads[2] = 32;
    threads[1] = max_wg_size / threads[2];
  }

  // grid size
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  q.submit([&](sycl::handler &cgh) {
     sycl::accessor<float, 1, sycl::access_mode::read_write,
                    sycl::access::target::local>
         du_acc_ct1(sycl::range<1>((32 + 2) * (6 + 2)), cgh);
     sycl::accessor<float, 1, sycl::access_mode::read_write,
                    sycl::access::target::local>
         dv_acc_ct1(sycl::range<1>((32 + 2) * (6 + 2)), cgh);

     cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                      [=](sycl::nd_item<3> item_ct1) {
                        JacobiIteration<32, 6>(du0, dv0, Ix, Iy, Iz, w, h, s,
                                               alpha, du1, dv1, item_ct1,
                                               du_acc_ct1.get_pointer(),
                                               dv_acc_ct1.get_pointer());
                      });
   }).wait();
}
