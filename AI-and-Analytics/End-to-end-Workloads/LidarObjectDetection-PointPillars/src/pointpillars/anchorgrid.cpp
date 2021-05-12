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

#include "pointpillars/anchorgrid.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include "devicemanager/devicemanager.hpp"
#include "pointpillars/common.hpp"
#include "pointpillars/scan.hpp"

namespace pointpillars {

AnchorGrid::AnchorGrid(AnchorGridConfig &config)
    : config_{config},
      dev_anchors_px_{nullptr},
      dev_anchors_py_{nullptr},
      dev_anchors_pz_{nullptr},
      dev_anchors_dx_{nullptr},
      dev_anchors_dy_{nullptr},
      dev_anchors_dz_{nullptr},
      dev_anchors_ro_{nullptr},
      dev_anchors_rad_{nullptr},
      host_anchors_px_{nullptr},
      host_anchors_py_{nullptr},
      host_anchors_pz_{nullptr},
      host_anchors_dx_{nullptr},
      host_anchors_dy_{nullptr},
      host_anchors_dz_{nullptr},
      host_anchors_ro_{nullptr},
      host_anchors_rad_{nullptr} {
  mc_ = config_.anchors.size();
  mr_ = config_.rotations.size();

  // In the anchor map the Width is along y and the Height along x
  mh_ = static_cast<std::size_t>((config_.max_x_range - config_.min_x_range) / config_.x_stride);
  mw_ = static_cast<std::size_t>((config_.max_y_range - config_.min_y_range) / config_.y_stride);
  num_anchors_ = mw_ * mh_ * mc_ * mr_;
}

AnchorGrid::~AnchorGrid() {
  ClearDeviceMemory();
  ClearHostMemory();
}

void AnchorGrid::GenerateAnchors() {
  AllocateHostMemory();

  // Minimum (x, y) anchor grid coordinates + pillar center offset
  float x_offset = config_.min_x_range + 0.5f * config_.x_stride;
  float y_offset = config_.min_y_range + 0.5f * config_.y_stride;

  // In the anchor map the Width is along y and the Height along x, c is the class, r is the rotation
  for (size_t y = 0; y < mw_; y++) {
    for (size_t x = 0; x < mh_; x++) {
      for (size_t c = 0; c < mc_; c++) {
        for (size_t r = 0; r < mr_; r++) {
          std::size_t index = y * mh_ * mc_ * mr_ + x * mc_ * mr_ + c * mr_ + r;

          // Set anchor grid locations at the center of the pillar
          host_anchors_px_[index] = static_cast<float>(x) * config_.x_stride + x_offset;
          host_anchors_py_[index] = static_cast<float>(y) * config_.y_stride + y_offset;

          // Assign z as dz
          host_anchors_pz_[index] = config_.anchors[c].dz;

          // Assign current anchor rotation r
          host_anchors_ro_[index] = config_.rotations[r];

          // Assign anchors sizes for the given class c
          host_anchors_dx_[index] = config_.anchors[c].x;
          host_anchors_dy_[index] = config_.anchors[c].y;
          host_anchors_dz_[index] = config_.anchors[c].z;
        }
      }
    }
  }

  // host_anchors_rad_ is used to optimize the decoding by precalculating an effective radius around the anchor
  for (std::size_t c = 0; c < mc_; c++) {
    host_anchors_rad_[c] = std::min(config_.anchors[c].x, config_.anchors[c].y);
  }

  MoveAnchorsToDevice();
}

void AnchorGrid::AllocateHostMemory() {
  host_anchors_px_ = new float[num_anchors_];
  host_anchors_py_ = new float[num_anchors_];
  host_anchors_pz_ = new float[num_anchors_];
  host_anchors_dx_ = new float[num_anchors_];
  host_anchors_dy_ = new float[num_anchors_];
  host_anchors_dz_ = new float[num_anchors_];
  host_anchors_ro_ = new float[num_anchors_];

  for (std::size_t i = 0; i < num_anchors_; i++) {
    host_anchors_px_[i] = 0.f;
    host_anchors_py_[i] = 0.f;
    host_anchors_pz_[i] = 0.f;
    host_anchors_dx_[i] = 0.f;
    host_anchors_dy_[i] = 0.f;
    host_anchors_dz_[i] = 0.f;
    host_anchors_ro_[i] = 0.f;
  }

  host_anchors_rad_ = new float[mc_];

  for (std::size_t i = 0; i < mc_; i++) {
    host_anchors_rad_[i] = 0.f;
  }
}

void AnchorGrid::ClearHostMemory() {
  delete[] host_anchors_px_;
  delete[] host_anchors_py_;
  delete[] host_anchors_pz_;
  delete[] host_anchors_dx_;
  delete[] host_anchors_dy_;
  delete[] host_anchors_dz_;
  delete[] host_anchors_ro_;

  delete[] host_anchors_rad_;

  host_anchors_px_ = nullptr;
  host_anchors_py_ = nullptr;
  host_anchors_pz_ = nullptr;
  host_anchors_dx_ = nullptr;
  host_anchors_dy_ = nullptr;
  host_anchors_dz_ = nullptr;
  host_anchors_ro_ = nullptr;
  host_anchors_rad_ = nullptr;
}

void AnchorGrid::ClearDeviceMemory() {
  sycl::queue queue = devicemanager::GetCurrentQueue();
  sycl::free(dev_anchors_px_, queue);
  sycl::free(dev_anchors_py_, queue);
  sycl::free(dev_anchors_pz_, queue);
  sycl::free(dev_anchors_dx_, queue);
  sycl::free(dev_anchors_dy_, queue);
  sycl::free(dev_anchors_dz_, queue);
  sycl::free(dev_anchors_ro_, queue);
  sycl::free(dev_anchors_rad_, queue);

  dev_anchors_px_ = nullptr;
  dev_anchors_py_ = nullptr;
  dev_anchors_pz_ = nullptr;
  dev_anchors_dx_ = nullptr;
  dev_anchors_dy_ = nullptr;
  dev_anchors_dz_ = nullptr;
  dev_anchors_ro_ = nullptr;
  dev_anchors_rad_ = nullptr;
}

void AnchorGrid::AllocateDeviceMemory() {
  sycl::queue queue = devicemanager::GetCurrentQueue();
  dev_anchors_px_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_py_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_pz_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_dx_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_dy_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_dz_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_ro_ = sycl::malloc_device<float>(num_anchors_, queue);
  dev_anchors_rad_ = sycl::malloc_device<float>(mc_, queue);
}

void AnchorGrid::MoveAnchorsToDevice() {
  AllocateDeviceMemory();

  sycl::queue queue = devicemanager::GetCurrentQueue();
  queue.memcpy(dev_anchors_px_, host_anchors_px_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_py_, host_anchors_py_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_pz_, host_anchors_pz_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_dx_, host_anchors_dx_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_dy_, host_anchors_dy_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_dz_, host_anchors_dz_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_ro_, host_anchors_ro_, num_anchors_ * sizeof(float));
  queue.memcpy(dev_anchors_rad_, host_anchors_rad_, mc_ * sizeof(float));
  queue.wait();

  ClearHostMemory();
}

void AnchorGrid::CreateAnchorMask(int *dev_pillar_map, const int pillar_map_w, const int pillar_map_h,
                                  const float pillar_size_x, const float pillar_size_y, int *dev_anchor_mask,
                                  int *dev_pillar_workspace) {
  // Calculate the cumulative sum over the 2D grid dev_pillar_map in both X and Y

  // Calculate an N greater than the current pillar map size enough to hold the cummulative sum matrix
  const std::size_t n = NextPower(static_cast<std::size_t>(std::max(pillar_map_h, pillar_map_w)));

  // Calculate the cumulative sum
  ScanX(dev_pillar_workspace, dev_pillar_map, pillar_map_h, pillar_map_w, n);
  ScanY(dev_pillar_map, dev_pillar_workspace, pillar_map_h, pillar_map_w, n);

  // Mask anchors only where input data is found
  MaskAnchors(dev_anchors_px_, dev_anchors_py_, dev_pillar_map, dev_anchor_mask, dev_anchors_rad_, config_.min_x_range,
              config_.min_y_range, pillar_size_x, pillar_size_y, pillar_map_h, pillar_map_w, mc_, mr_, mh_, mw_);
}

void MaskAnchorsKernel(const float *anchors_px, const float *anchors_py, const int *pillar_map, int *anchor_mask,
                       const float *anchors_rad, const float min_x_range, const float min_y_range,
                       const float pillar_x_size, const float pillar_y_size, const int grid_x_size,
                       const int grid_y_size, sycl::nd_item<3> item_ct1) {
  const int H = item_ct1.get_local_range().get(2);
  const int R = item_ct1.get_local_range().get(1);
  const int C = item_ct1.get_group_range(1);

  const int x = item_ct1.get_local_id(2);
  const int r = item_ct1.get_local_id(1);
  const int y = item_ct1.get_group(2);
  const int c = item_ct1.get_group(1);

  int index = y * H * C * R + x * C * R + c * R + r;

  float rad = anchors_rad[c];

  float x_anchor = anchors_px[index];
  float y_anchor = anchors_py[index];

  int anchor_coordinates_min_x = (x_anchor - rad - min_x_range) / pillar_x_size;
  int anchor_coordinates_min_y = (y_anchor - rad - min_y_range) / pillar_y_size;
  int anchor_coordinates_max_x = (x_anchor + rad - min_x_range) / pillar_x_size;
  int anchor_coordinates_max_y = (y_anchor + rad - min_y_range) / pillar_y_size;

  anchor_coordinates_min_x = sycl::max(anchor_coordinates_min_x, 0);
  anchor_coordinates_min_y = sycl::max(anchor_coordinates_min_y, 0);
  anchor_coordinates_max_x = sycl::min(anchor_coordinates_max_x, (int)(grid_x_size - 1));
  anchor_coordinates_max_y = sycl::min(anchor_coordinates_max_y, (int)(grid_y_size - 1));

  // cumulative sum difference
  int bottom_left = pillar_map[anchor_coordinates_max_y * grid_x_size + anchor_coordinates_min_x];
  int top_left = pillar_map[anchor_coordinates_min_y * grid_x_size + anchor_coordinates_min_x];

  int bottom_right = pillar_map[anchor_coordinates_max_y * grid_x_size + anchor_coordinates_max_x];
  int top_right = pillar_map[anchor_coordinates_min_y * grid_x_size + anchor_coordinates_max_x];

  // Area calculation
  int area = bottom_right - top_right - bottom_left + top_left;

  if (area >= 1) {
    anchor_mask[index] = 1;
  } else {
    anchor_mask[index] = 0;
  }
}

void MaskAnchorsSimpleKernel(const float *anchors_px, const float *anchors_py, const int *pillar_map, int *anchor_mask,
                             const float *anchors_rad, const float min_x_range, const float min_y_range,
                             const float pillar_x_size, const float pillar_y_size, const int grid_x_size,
                             const int grid_y_size, const int index, const int c) {
  float rad = anchors_rad[c];

  float x_anchor = anchors_px[index];
  float y_anchor = anchors_py[index];

  int anchor_coordinates_min_x = (x_anchor - rad - min_x_range) / pillar_x_size;
  int anchor_coordinates_min_y = (y_anchor - rad - min_y_range) / pillar_y_size;
  int anchor_coordinates_max_x = (x_anchor + rad - min_x_range) / pillar_x_size;
  int anchor_coordinates_max_y = (y_anchor + rad - min_y_range) / pillar_y_size;

  anchor_coordinates_min_x = sycl::max(anchor_coordinates_min_x, 0);
  anchor_coordinates_min_y = sycl::max(anchor_coordinates_min_y, 0);
  anchor_coordinates_max_x = sycl::min(anchor_coordinates_max_x, (int)(grid_x_size - 1));
  anchor_coordinates_max_y = sycl::min(anchor_coordinates_max_y, (int)(grid_y_size - 1));

  // cumulative sum difference
  int bottom_left = pillar_map[anchor_coordinates_max_y * grid_x_size + anchor_coordinates_min_x];
  int top_left = pillar_map[anchor_coordinates_min_y * grid_x_size + anchor_coordinates_min_x];

  int bottom_right = pillar_map[anchor_coordinates_max_y * grid_x_size + anchor_coordinates_max_x];
  int top_right = pillar_map[anchor_coordinates_min_y * grid_x_size + anchor_coordinates_max_x];

  // Area calculation
  int area = bottom_right - top_right - bottom_left + top_left;

  if (area >= 1) {
    anchor_mask[index] = 1;
  } else {
    anchor_mask[index] = 0;
  }
}

void AnchorGrid::MaskAnchors(const float *dev_anchors_px, const float *dev_anchors_py, const int *dev_pillar_map,
                             int *dev_anchor_mask, const float *dev_anchors_rad, const float min_x_range,
                             const float min_y_range, const float pillar_x_size, const float pillar_y_size,
                             const int grid_x_size, const int grid_y_size, const int C, const int R, const int H,
                             const int W) {
  sycl::queue queue = devicemanager::GetCurrentQueue();

  if (!devicemanager::GetCurrentDevice().is_gpu()) {
    sycl::range<3> block(H, R, 1);
    sycl::range<3> grid(W, C, 1);

    queue.submit([&](auto &h) {
      auto range = grid * block;

      h.parallel_for(sycl::nd_range<3>(sycl::range<3>(range.get(2), range.get(1), range.get(0)),
                                       sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                     [=](sycl::nd_item<3> item_ct1) {
                       MaskAnchorsKernel(dev_anchors_px, dev_anchors_py, dev_pillar_map, dev_anchor_mask,
                                         dev_anchors_rad, min_x_range, min_y_range, pillar_x_size, pillar_y_size,
                                         grid_x_size, grid_y_size, item_ct1);
                     });
    });
  } else {
    const unsigned int length = H * W * C * R;
    queue.submit([&](auto &h) {
      h.parallel_for(sycl::range<1>{length}, [=](sycl::id<1> it) {
        const int index = it[0];

        const int y = index / (H * C * R);
        const int x = (index - y * H * C * R) / (C * R);
        const int c = (index - y * H * C * R - x * C * R) / R;
        MaskAnchorsSimpleKernel(dev_anchors_px, dev_anchors_py, dev_pillar_map, dev_anchor_mask, dev_anchors_rad,
                                min_x_range, min_y_range, pillar_x_size, pillar_y_size, grid_x_size, grid_y_size, index,
                                c);
      });
    });
  }

  queue.wait();
}

}  // namespace pointpillars
