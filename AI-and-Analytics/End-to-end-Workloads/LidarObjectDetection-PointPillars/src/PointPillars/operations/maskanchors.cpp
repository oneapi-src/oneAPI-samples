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

#include "PointPillars/operations/maskanchors.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>

namespace dnn {

void mask_anchors(const float *anchorsPx, const float *anchorsPy, const int *pillarMap, int *anchorMask,
                  const float *anchorsRad, const float minXRange, const float minYRange, const float pillarXSize,
                  const float pillarYSize, const int gridXSize, const int gridYSize, sycl::nd_item<3> item_ct1) {
  const int H = item_ct1.get_local_range().get(2);
  const int R = item_ct1.get_local_range().get(1);
  const int C = item_ct1.get_group_range(1);

  const int x = item_ct1.get_local_id(2);
  const int r = item_ct1.get_local_id(1);
  const int y = item_ct1.get_group(2);
  const int c = item_ct1.get_group(1);

  int index = y * H * C * R + x * C * R + c * R + r;

  float rad = anchorsRad[c];

  float xAnchor = anchorsPx[index];
  float yAnchor = anchorsPy[index];

  int anchorCoordinatesMinX = (xAnchor - rad - minXRange) / pillarXSize;
  int anchorCoordinatesMinY = (yAnchor - rad - minYRange) / pillarYSize;
  int anchorCoordinatesMaxX = (xAnchor + rad - minXRange) / pillarXSize;
  int anchorCoordinatesMaxY = (yAnchor + rad - minYRange) / pillarYSize;

  // @todo: anchor mask is set to zero at the grid borders
  anchorCoordinatesMinX = sycl::max(anchorCoordinatesMinX, 0);
  anchorCoordinatesMinY = sycl::max(anchorCoordinatesMinY, 0);
  anchorCoordinatesMaxX = sycl::min(anchorCoordinatesMaxX, (int)(gridXSize - 1));
  anchorCoordinatesMaxY = sycl::min(anchorCoordinatesMaxY, (int)(gridYSize - 1));

  // cumulative sum difference
  int bottomLeft = pillarMap[anchorCoordinatesMaxY * gridXSize + anchorCoordinatesMinX];
  int topLeft = pillarMap[anchorCoordinatesMinY * gridXSize + anchorCoordinatesMinX];

  int bottomRight = pillarMap[anchorCoordinatesMaxY * gridXSize + anchorCoordinatesMaxX];
  int topRight = pillarMap[anchorCoordinatesMinY * gridXSize + anchorCoordinatesMaxX];

  // Area calculation
  int area = bottomRight - topRight - bottomLeft + topLeft;

  if (area >= 1) {
    anchorMask[index] = 1;
  } else {
    anchorMask[index] = 0;
  }
}

void mask_anchors_simple(const float *anchorsPx, const float *anchorsPy, const int *pillarMap, int *anchorMask,
                         const float *anchorsRad, const float minXRange, const float minYRange, const float pillarXSize,
                         const float pillarYSize, const int gridXSize, const int gridYSize, const int index,
                         const int c) {
  float rad = anchorsRad[c];

  float xAnchor = anchorsPx[index];
  float yAnchor = anchorsPy[index];

  int anchorCoordinatesMinX = (xAnchor - rad - minXRange) / pillarXSize;
  int anchorCoordinatesMinY = (yAnchor - rad - minYRange) / pillarYSize;
  int anchorCoordinatesMaxX = (xAnchor + rad - minXRange) / pillarXSize;
  int anchorCoordinatesMaxY = (yAnchor + rad - minYRange) / pillarYSize;

  // @todo: anchor mask is set to zero at the grid borders
  anchorCoordinatesMinX = sycl::max(anchorCoordinatesMinX, 0);
  anchorCoordinatesMinY = sycl::max(anchorCoordinatesMinY, 0);
  anchorCoordinatesMaxX = sycl::min(anchorCoordinatesMaxX, (int)(gridXSize - 1));
  anchorCoordinatesMaxY = sycl::min(anchorCoordinatesMaxY, (int)(gridYSize - 1));

  // cumulative sum difference
  int bottomLeft = pillarMap[anchorCoordinatesMaxY * gridXSize + anchorCoordinatesMinX];
  int topLeft = pillarMap[anchorCoordinatesMinY * gridXSize + anchorCoordinatesMinX];

  int bottomRight = pillarMap[anchorCoordinatesMaxY * gridXSize + anchorCoordinatesMaxX];
  int topRight = pillarMap[anchorCoordinatesMinY * gridXSize + anchorCoordinatesMaxX];

  // Area calculation
  int area = bottomRight - topRight - bottomLeft + topLeft;

  if (area >= 1) {
    anchorMask[index] = 1;
  } else {
    anchorMask[index] = 0;
  }
}

void maskAnchors(const float *devAnchorsPx, const float *devAnchorsPy, const int *devPillarMap, int *devAnchorMask,
                 const float *devAnchorsRad, const float minXRange, const float minYRange, const float pillarXSize,
                 const float pillarYSize, const int gridXSize, const int gridYSize, const int C, const int R,
                 const int H, const int W) {
  if (!dpct::get_current_device().is_gpu()) {
    sycl::range<3> block(H, R, 1);
    sycl::range<3> grid(W, C, 1);

    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto dpct_global_range = grid * block;

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                           sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                       [=](sycl::nd_item<3> item_ct1) {
                         mask_anchors(devAnchorsPx, devAnchorsPy, devPillarMap, devAnchorMask, devAnchorsRad, minXRange,
                                      minYRange, pillarXSize, pillarYSize, gridXSize, gridYSize, item_ct1);
                       });
    });
  } else {
    const unsigned int length = H * W * C * R;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {

      cgh.parallel_for(sycl::range<1>{length}, [=](sycl::id<1> it) {
        const int index = it[0];

        const int y = index / (H * C * R);
        const int x = (index - y * H * C * R) / (C * R);
        const int c = (index - y * H * C * R - x * C * R) / R;
        mask_anchors_simple(devAnchorsPx, devAnchorsPy, devPillarMap, devAnchorMask, devAnchorsRad, minXRange,
                            minYRange, pillarXSize, pillarYSize, gridXSize, gridYSize, index, c);
      });
    });
  }
}

}  // namespace dnn
