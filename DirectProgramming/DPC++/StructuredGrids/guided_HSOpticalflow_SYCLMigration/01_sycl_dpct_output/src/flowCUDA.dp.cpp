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
#include <dpct/dpct.hpp>

#include "common.h"

// include kernels
#include "addKernel.dp.hpp"
#include "derivativesKernel.dp.hpp"
#include "downscaleKernel.dp.hpp"
#include "solverKernel.dp.hpp"
#include "upscaleKernel.dp.hpp"
#include "warpingKernel.dp.hpp"

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowCUDA(const float *I0, const float *I1, int width, int height,
                     int stride, float alpha, int nLevels, int nWarpIters,
                     int nSolverIters, float *u, float *v) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  printf("Computing optical flow on GPU...\n");

  // pI0 and pI1 will hold device pointers
  const float **pI0 = new const float *[nLevels];
  const float **pI1 = new const float *[nLevels];

  int *pW = new int[nLevels];
  int *pH = new int[nLevels];
  int *pS = new int[nLevels];

  // device memory pointers
  float *d_tmp;
  float *d_du0;
  float *d_dv0;
  float *d_du1;
  float *d_dv1;

  float *d_Ix;
  float *d_Iy;
  float *d_Iz;

  float *d_u;
  float *d_v;
  float *d_nu;
  float *d_nv;

  const int dataSize = stride * height * sizeof(float);

  /*
  DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_tmp = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_du0 = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_dv0 = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_du1 = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_dv1 = (float *)sycl::malloc_device(dataSize, q_ct1), 0));

  checkCudaErrors((d_Ix = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_Iy = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_Iz = (float *)sycl::malloc_device(dataSize, q_ct1), 0));

  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_u = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  checkCudaErrors((d_v = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_nu = (float *)sycl::malloc_device(dataSize, q_ct1), 0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_nv = (float *)sycl::malloc_device(dataSize, q_ct1), 0));

  // prepare pyramid

  int currentLevel = nLevels - 1;
  // allocate GPU memory for input images
  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((*(pI0 + currentLevel) =
                       (const float *)sycl::malloc_device(dataSize, q_ct1),
                   0));
  /*
  DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((*(pI1 + currentLevel) =
                       (const float *)sycl::malloc_device(dataSize, q_ct1),
                   0));

  /*
  DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (q_ct1.memcpy((void *)pI0[currentLevel], I0, dataSize).wait(), 0));
  /*
  DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (q_ct1.memcpy((void *)pI1[currentLevel], I1, dataSize).wait(), 0));

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;

  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);

    /*
    DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (*(pI0 + currentLevel - 1) =
             (const float *)sycl::malloc_device(ns * nh * sizeof(float), q_ct1),
         0));
    /*
    DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (*(pI1 + currentLevel - 1) =
             (const float *)sycl::malloc_device(ns * nh * sizeof(float), q_ct1),
         0));

    Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1]);

    Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1]);

    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }

  /*
  DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (q_ct1.memset(d_u, 0, stride * height * sizeof(float)).wait(), 0));
  /*
  DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (q_ct1.memset(d_v, 0, stride * height * sizeof(float)).wait(), 0));

  // compute flow
  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      /*
      DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((q_ct1.memset(d_du0, 0, dataSize).wait(), 0));
      /*
      DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((q_ct1.memset(d_dv0, 0, dataSize).wait(), 0));

      /*
      DPCT1003:41: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((q_ct1.memset(d_du1, 0, dataSize).wait(), 0));
      /*
      DPCT1003:42: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((q_ct1.memset(d_dv1, 0, dataSize).wait(), 0));

      // on current level we compute optical flow
      // between frame 0 and warped frame 1
      WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                pS[currentLevel], d_u, d_v, d_tmp);

      ComputeDerivatives(pI0[currentLevel], d_tmp, pW[currentLevel],
                         pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz);

      for (int iter = 0; iter < nSolverIters; ++iter) {
        SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz, pW[currentLevel],
                       pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1);

        Swap(d_du0, d_du1);
        Swap(d_dv0, d_dv1);
      }

      // update u, v
      Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u);
      Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v);
    }

    if (currentLevel != nLevels - 1) {
      // prolongate solution
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];

      Upscale(d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, d_nu);

      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

      Upscale(d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, d_nv);

      Swap(d_u, d_nu);
      Swap(d_v, d_nv);
    }
  }

  /*
  DPCT1003:43: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((q_ct1.memcpy(u, d_u, dataSize).wait(), 0));
  /*
  DPCT1003:44: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((q_ct1.memcpy(v, d_v, dataSize).wait(), 0));

  // cleanup
  for (int i = 0; i < nLevels; ++i) {
    /*
    DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free((void *)pI0[i], q_ct1), 0));
    /*
    DPCT1003:46: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free((void *)pI1[i], q_ct1), 0));
  }

  delete[] pI0;
  delete[] pI1;
  delete[] pW;
  delete[] pH;
  delete[] pS;

  /*
  DPCT1003:47: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_tmp, q_ct1), 0));
  /*
  DPCT1003:48: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_du0, q_ct1), 0));
  /*
  DPCT1003:49: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_dv0, q_ct1), 0));
  /*
  DPCT1003:50: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_du1, q_ct1), 0));
  /*
  DPCT1003:51: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_dv1, q_ct1), 0));
  /*
  DPCT1003:52: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_Ix, q_ct1), 0));
  /*
  DPCT1003:53: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_Iy, q_ct1), 0));
  /*
  DPCT1003:54: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_Iz, q_ct1), 0));
  /*
  DPCT1003:55: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_nu, q_ct1), 0));
  /*
  DPCT1003:56: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_nv, q_ct1), 0));
  /*
  DPCT1003:57: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_u, q_ct1), 0));
  /*
  DPCT1003:58: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_v, q_ct1), 0));
}
