/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 * Copyright (c) 2019-2021 Intel Corporation (oneAPI modifications)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PointPillars/operations/anchorgrid.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>
#include "PointPillars/operations/common.hpp"
#include "PointPillars/operations/maskanchors.hpp"
#include "PointPillars/operations/scan.hpp"

namespace dnn {

AnchorGrid::AnchorGrid(AnchorGridConfig &config)
    : mConfig{config},
      mDevAnchorsPx{nullptr},
      mDevAnchorsPy{nullptr},
      mDevAnchorsPz{nullptr},
      mDevAnchorsDx{nullptr},
      mDevAnchorsDy{nullptr},
      mDevAnchorsDz{nullptr},
      mDevAnchorsRo{nullptr},
      mDevAnchorsRad{nullptr},
      mHostAnchorsPx{nullptr},
      mHostAnchorsPy{nullptr},
      mHostAnchorsPz{nullptr},
      mHostAnchorsDx{nullptr},
      mHostAnchorsDy{nullptr},
      mHostAnchorsDz{nullptr},
      mHostAnchorsRo{nullptr},
      mHostAnchorsRad{nullptr} {
  mC = mConfig.anchors.size();
  mR = mConfig.rotations.size();

  // In the anchor map the Width is along y and the Height along x
  mH = static_cast<std::size_t>((mConfig.maxXRange - mConfig.minXRange) / mConfig.xStride);
  mW = static_cast<std::size_t>((mConfig.maxYRange - mConfig.minYRange) / mConfig.yStride);
  mNumAnchors = mW * mH * mC * mR;
}

AnchorGrid::~AnchorGrid() {
  clearDeviceMemory();

  clearHostMemory();
}

void AnchorGrid::generateAnchors() {
  allocateHostMemory(mNumAnchors);

  float xOffset = mConfig.minXRange + 0.5f * mConfig.xStride;
  float yOffset = mConfig.minYRange + 0.5f * mConfig.yStride;

  // In the anchor map the Width is along y and the Height along x
  for (size_t y = 0; y < mW; y++) {
    for (size_t x = 0; x < mH; x++) {
      for (size_t c = 0; c < mC; c++) {
        for (size_t r = 0; r < mR; r++) {
          std::size_t index = y * mH * mC * mR + x * mC * mR + c * mR + r;

          mHostAnchorsPx[index] = static_cast<float>(x) * mConfig.xStride + xOffset;
          mHostAnchorsPy[index] = static_cast<float>(y) * mConfig.yStride + yOffset;
          mHostAnchorsPz[index] = mConfig.anchors[c].dz;

          mHostAnchorsRo[index] = mConfig.rotations[r];

          mHostAnchorsDx[index] = mConfig.anchors[c].x;
          mHostAnchorsDy[index] = mConfig.anchors[c].y;
          mHostAnchorsDz[index] = mConfig.anchors[c].z;
        }
      }
    }
  }

  for (std::size_t c = 0; c < mC; c++) {
    mHostAnchorsRad[c] = std::min(mConfig.anchors[c].x, mConfig.anchors[c].y);
  }
}

void AnchorGrid::allocateHostMemory(const std::size_t &numAnchors) {
  mHostAnchorsPx = new float[numAnchors];
  mHostAnchorsPy = new float[numAnchors];
  mHostAnchorsPz = new float[numAnchors];
  mHostAnchorsDx = new float[numAnchors];
  mHostAnchorsDy = new float[numAnchors];
  mHostAnchorsDz = new float[numAnchors];
  mHostAnchorsRo = new float[numAnchors];

  for (std::size_t i = 0; i < numAnchors; i++) {
    mHostAnchorsPx[i] = 0.f;
    mHostAnchorsPy[i] = 0.f;
    mHostAnchorsPz[i] = 0.f;
    mHostAnchorsDx[i] = 0.f;
    mHostAnchorsDy[i] = 0.f;
    mHostAnchorsDz[i] = 0.f;
    mHostAnchorsRo[i] = 0.f;
  }

  mHostAnchorsRad = new float[mC];

  for (std::size_t i = 0; i < mC; i++) {
    mHostAnchorsRad[i] = 0.f;
  }
}

void AnchorGrid::clearHostMemory() {
  delete[] mHostAnchorsPx;
  delete[] mHostAnchorsPy;
  delete[] mHostAnchorsPz;
  delete[] mHostAnchorsDx;
  delete[] mHostAnchorsDy;
  delete[] mHostAnchorsDz;
  delete[] mHostAnchorsRo;

  delete[] mHostAnchorsRad;

  mHostAnchorsPx = nullptr;
  mHostAnchorsPy = nullptr;
  mHostAnchorsPz = nullptr;
  mHostAnchorsDx = nullptr;
  mHostAnchorsDy = nullptr;
  mHostAnchorsDz = nullptr;
  mHostAnchorsRo = nullptr;
  mHostAnchorsRad = nullptr;
}

void AnchorGrid::clearDeviceMemory() {
  sycl::free(mDevAnchorsPx, dpct::get_default_context());
  sycl::free(mDevAnchorsPy, dpct::get_default_context());
  sycl::free(mDevAnchorsPz, dpct::get_default_context());
  sycl::free(mDevAnchorsDx, dpct::get_default_context());
  sycl::free(mDevAnchorsDy, dpct::get_default_context());
  sycl::free(mDevAnchorsDz, dpct::get_default_context());
  sycl::free(mDevAnchorsRo, dpct::get_default_context());
  sycl::free(mDevAnchorsRad, dpct::get_default_context());

  mDevAnchorsPx = nullptr;
  mDevAnchorsPy = nullptr;
  mDevAnchorsPz = nullptr;
  mDevAnchorsDx = nullptr;
  mDevAnchorsDy = nullptr;
  mDevAnchorsDz = nullptr;
  mDevAnchorsRo = nullptr;
  mDevAnchorsRad = nullptr;
}

void AnchorGrid::allocateDeviceMemory(const std::size_t &numAnchors) {
  mDevAnchorsPx =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsPy =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsPz =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsDx =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsDy =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsDz =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsRo =
      (float *)sycl::malloc_device(numAnchors * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  mDevAnchorsRad =
      (float *)sycl::malloc_device(mC * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
}

void AnchorGrid::moveAnchorsToDevice() {
  allocateDeviceMemory(mNumAnchors);

  sycl::queue &q_ct0 = dpct::get_default_queue();
  q_ct0.wait();
  q_ct0.memcpy(mDevAnchorsPx, mHostAnchorsPx, mNumAnchors * sizeof(float)).wait();
  q_ct0.memcpy(mDevAnchorsPy, mHostAnchorsPy, mNumAnchors * sizeof(float)).wait();
  q_ct0.memcpy(mDevAnchorsPz, mHostAnchorsPz, mNumAnchors * sizeof(float)).wait();
  q_ct0.memcpy(mDevAnchorsDx, mHostAnchorsDx, mNumAnchors * sizeof(float)).wait();
  q_ct0.memcpy(mDevAnchorsDy, mHostAnchorsDy, mNumAnchors * sizeof(float)).wait();
  q_ct0.memcpy(mDevAnchorsDz, mHostAnchorsDz, mNumAnchors * sizeof(float)).wait();
  q_ct0.memcpy(mDevAnchorsRo, mHostAnchorsRo, mNumAnchors * sizeof(float)).wait();

  q_ct0.memcpy(mDevAnchorsRad, mHostAnchorsRad, mC * sizeof(float)).wait();

  clearHostMemory();
}

void AnchorGrid::createAnchorMask(int *devPillarMap, const int pillarMapW, const int pillarMapH,
                                  const float pillarSizeX, const float pillarSizeY, int *devAnchorMask,
                                  int *devPillarWorkspace) {
  std::size_t N = nextPower(static_cast<std::size_t>(std::max(pillarMapH, pillarMapW)));
  scanX(devPillarWorkspace, devPillarMap, pillarMapH, pillarMapW, N);
  scanY(devPillarMap, devPillarWorkspace, pillarMapH, pillarMapW, N);

  maskAnchors(mDevAnchorsPx, mDevAnchorsPy, devPillarMap, devAnchorMask, mDevAnchorsRad, mConfig.minXRange,
              mConfig.minYRange, pillarSizeX, pillarSizeY, pillarMapH, pillarMapW, mC, mR, mH, mW);
}

}  // namespace dnn
