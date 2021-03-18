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

#include "pointpillars/preprocess.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include "devicemanager/devicemanager.hpp"
#include "pointpillars/common.hpp"

namespace pointpillars {

// This kernel is called on each point of the Point Cloud.  It calculates the coordinates of the point in the 2D pillar
// map and adds it to the corresponding pillar.
void MakePillarHistoKernel(const float *dev_points, float *dev_pillar_x_in_coors, float *dev_pillar_y_in_coors,
                           float *dev_pillar_z_in_coors, float *dev_pillar_i_in_coors,
                           int *pillar_count_histo,  // holds the point count on the cell
                           const int num_points, const int max_points_per_pillar, const int grid_x_size,
                           const int grid_y_size, const int grid_z_size, const float min_x_range,
                           const float min_y_range, const float min_z_range, const float pillar_x_size,
                           const float pillar_y_size, const float pillar_z_size, sycl::nd_item<3> item_ct1) {
  int point_index = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  if (point_index >= num_points) {
    return;
  }

  // Indexes in the pillar map
  int xIndex = sycl::floor((float)((dev_points[point_index * 4 + 0] - min_x_range) / pillar_x_size));
  int yIndex = sycl::floor((float)((dev_points[point_index * 4 + 1] - min_y_range) / pillar_y_size));
  int zIndex = sycl::floor((float)((dev_points[point_index * 4 + 2] - min_z_range) / pillar_z_size));

  // Check if it is within grid range
  if (xIndex >= 0 && xIndex < grid_x_size && yIndex >= 0 && yIndex < grid_y_size && zIndex >= 0 &&
      zIndex < grid_z_size) {
    // increase the point count
    int count = AtomicFetchAdd(&pillar_count_histo[yIndex * grid_x_size + xIndex], 1);
    if (count < max_points_per_pillar) {
      // pillar index
      int pillarIndex = yIndex * grid_x_size * max_points_per_pillar + xIndex * max_points_per_pillar;

      // pointPillarIndex is the m-point in the n-pillar
      int pointPillarIndex = pillarIndex + count;

      // add point to pillar data
      dev_pillar_x_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 0];
      dev_pillar_y_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 1];
      dev_pillar_z_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 2];
      dev_pillar_i_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 3];
    }
  }
}

// This kernel is executed on a specific location in the pillar map.
// It will test if the corresponding pillar has points.
// In such case it will mark the pillar for use as input to the PillarFeatureExtraction
// A pillar mask is also generated and can be used to optimize the decoding.
void MakePillarIndexKernel(int *dev_pillar_count_histo, int *dev_counter, int *dev_pillar_count, int *dev_x_coors,
                           int *dev_y_coors, float *dev_x_coors_for_sub, float *dev_y_coors_for_sub,
                           float *dev_num_points_per_pillar, int *dev_sparse_pillar_map, const int max_pillars,
                           const int max_points_per_pillar, const int grid_x_size, const float min_x_range,
                           const float min_y_range, const float pillar_x_size, const float pillar_y_size, const int x,
                           const int y) {
  int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = AtomicFetchAdd(dev_counter, 1);
  if (count < max_pillars) {
    AtomicFetchAdd(dev_pillar_count, 1);
    if (num_points_at_this_pillar >= max_points_per_pillar) {
      dev_num_points_per_pillar[count] = max_points_per_pillar;
    } else {
      dev_num_points_per_pillar[count] = num_points_at_this_pillar;
    }

    // grid coordinates of this pillar
    dev_x_coors[count] = x;
    dev_y_coors[count] = y;

    // metric position of this pillar
    dev_x_coors_for_sub[count] = x * pillar_x_size + 0.5f * pillar_x_size + min_x_range;
    dev_y_coors_for_sub[count] = y * pillar_y_size + 0.5f * pillar_y_size + min_y_range;

    // map of pillars with at least one point
    dev_sparse_pillar_map[y * grid_x_size + x] = 1;
  }
}

// This kernel generates the input feature map to the PillarFeatureExtraction network.
// It takes the pillars that were marked for use and stores the first 4 features (x,y,z,i) in the input feature map.
void MakePillarFeatureKernel(float *dev_pillar_x_in_coors, float *dev_pillar_y_in_coors, float *dev_pillar_z_in_coors,
                             float *dev_pillar_i_in_coors, float *dev_pillar_x, float *dev_pillar_y,
                             float *dev_pillar_z, float *dev_pillar_i, int *dev_x_coors, int *dev_y_coors,
                             float *dev_num_points_per_pillar, const int max_points, const int grid_x_size,
                             sycl::nd_item<3> item_ct1) {
  int ith_pillar = item_ct1.get_group(2);
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = item_ct1.get_local_id(2);
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }
  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  int pillar_ind = ith_pillar * max_points + ith_point;
  int coors_ind = y_ind * grid_x_size * max_points + x_ind * max_points + ith_point;
  dev_pillar_x[pillar_ind] = dev_pillar_x_in_coors[coors_ind];
  dev_pillar_y[pillar_ind] = dev_pillar_y_in_coors[coors_ind];
  dev_pillar_z[pillar_ind] = dev_pillar_z_in_coors[coors_ind];
  dev_pillar_i[pillar_ind] = dev_pillar_i_in_coors[coors_ind];
}

// This kernel takes the pillars that were marked for use and stores the features: (pillar_center_x, pillar_center_y,
// pillar_mask) in the input feature map.
void MakeExtraNetworkInputKernel(float *dev_x_coors_for_sub, float *dev_y_coors_for_sub,
                                 float *dev_num_points_per_pillar, float *dev_x_coors_for_sub_shaped,
                                 float *dev_y_coors_for_sub_shaped, float *dev_pillar_feature_mask,
                                 const int max_num_points_per_pillar, sycl::nd_item<3> item_ct1) {
  int ith_pillar = item_ct1.get_group(2);
  int ith_point = item_ct1.get_local_id(2);
  float x = dev_x_coors_for_sub[ith_pillar];
  float y = dev_y_coors_for_sub[ith_pillar];
  int num_points_for_a_pillar = dev_num_points_per_pillar[ith_pillar];
  int ind = ith_pillar * max_num_points_per_pillar + ith_point;
  dev_x_coors_for_sub_shaped[ind] = x;
  dev_y_coors_for_sub_shaped[ind] = y;

  if (ith_point < num_points_for_a_pillar) {
    dev_pillar_feature_mask[ind] = 1.0f;
  } else {
    dev_pillar_feature_mask[ind] = 0.0f;
  }
}

PreProcess::PreProcess(const int max_num_pillars, const int max_points_per_pillar, const int grid_x_size,
                       const int grid_y_size, const int grid_z_size, const float pillar_x_size,
                       const float pillar_y_size, const float pillar_z_size, const float min_x_range,
                       const float min_y_range, const float min_z_range)
    : max_num_pillars_(max_num_pillars),
      max_num_points_per_pillar_(max_points_per_pillar),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range) {
  sycl::queue queue = devicemanager::GetCurrentQueue();

  // allocate memory
  dev_pillar_x_in_coors_ = sycl::malloc_device<float>(grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_, queue);
  dev_pillar_y_in_coors_ = sycl::malloc_device<float>(grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_, queue);
  dev_pillar_z_in_coors_ = sycl::malloc_device<float>(grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_, queue);
  dev_pillar_i_in_coors_ = sycl::malloc_device<float>(grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_, queue);
  dev_pillar_count_histo_ = sycl::malloc_device<int>(grid_y_size_ * grid_x_size_, queue);
  dev_counter_ = sycl::malloc_device<int>(1, queue);
  dev_pillar_count_ = sycl::malloc_device<int>(1, queue);
  dev_x_coors_for_sub_ = sycl::malloc_device<float>(max_num_pillars_, queue);
  dev_y_coors_for_sub_ = sycl::malloc_device<float>(max_num_pillars_, queue);
}

PreProcess::~PreProcess() {
  sycl::queue queue = devicemanager::GetCurrentQueue();
  sycl::free(dev_pillar_x_in_coors_, queue);
  sycl::free(dev_pillar_y_in_coors_, queue);
  sycl::free(dev_pillar_z_in_coors_, queue);
  sycl::free(dev_pillar_i_in_coors_, queue);
  sycl::free(dev_pillar_count_histo_, queue);
  sycl::free(dev_counter_, queue);
  sycl::free(dev_pillar_count_, queue);
  sycl::free(dev_x_coors_for_sub_, queue);
  sycl::free(dev_y_coors_for_sub_, queue);
}

void PreProcess::DoPreProcess(const float *dev_points, const int in_num_points, int *dev_x_coors, int *dev_y_coors,
                              float *dev_num_points_per_pillar, float *dev_pillar_x, float *dev_pillar_y,
                              float *dev_pillar_z, float *dev_pillar_i, float *dev_x_coors_for_sub_shaped,
                              float *dev_y_coors_for_sub_shaped, float *dev_pillar_feature_mask,
                              int *dev_sparse_pillar_map, int *host_pillar_count) {
  // Set Pillar input features to 0
  sycl::queue queue = devicemanager::GetCurrentQueue();
  queue.memset(dev_pillar_x_in_coors_, 0, grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_y_in_coors_, 0, grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_z_in_coors_, 0, grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_i_in_coors_, 0, grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_count_histo_, 0, grid_y_size_ * grid_x_size_ * sizeof(int));
  queue.memset(dev_counter_, 0, sizeof(int));
  queue.memset(dev_pillar_count_, 0, sizeof(int));
  queue.wait();

  // Use the point cloud data to generate the pillars
  // This will create create assign the point to the corresponding pillar in the grid. A maximum number of points can be
  // assigned to a single pillar.
  int num_block = DIVUP(in_num_points, 256);
  queue.submit([&](auto &h) {
    auto dev_pillar_x_in_coors_ct1 = dev_pillar_x_in_coors_;
    auto dev_pillar_y_in_coors_ct2 = dev_pillar_y_in_coors_;
    auto dev_pillar_z_in_coors_ct3 = dev_pillar_z_in_coors_;
    auto dev_pillar_i_in_coors_ct4 = dev_pillar_i_in_coors_;
    auto dev_pillar_count_histo_ct5 = dev_pillar_count_histo_;
    auto max_num_points_per_pillar_ct7 = max_num_points_per_pillar_;
    auto grid_x_size_ct8 = grid_x_size_;
    auto grid_y_size_ct9 = grid_y_size_;
    auto grid_z_size_ct10 = grid_z_size_;
    auto min_x_range_ct11 = min_x_range_;
    auto min_y_range_ct12 = min_y_range_;
    auto min_z_range_ct13 = min_z_range_;
    auto pillar_x_size_ct14 = pillar_x_size_;
    auto pillar_y_size_ct15 = pillar_y_size_;
    auto pillar_z_size_ct16 = pillar_z_size_;

    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_block) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
          MakePillarHistoKernel(dev_points, dev_pillar_x_in_coors_ct1, dev_pillar_y_in_coors_ct2,
                                dev_pillar_z_in_coors_ct3, dev_pillar_i_in_coors_ct4, dev_pillar_count_histo_ct5,
                                in_num_points, max_num_points_per_pillar_ct7, grid_x_size_ct8, grid_y_size_ct9,
                                grid_z_size_ct10, min_x_range_ct11, min_y_range_ct12, min_z_range_ct13,
                                pillar_x_size_ct14, pillar_y_size_ct15, pillar_z_size_ct16, item_ct1);
        });
  });
  queue.wait();

  // Check which pillars contain points and mark them for use during feature extraction.
  queue.submit([&](auto &h) {
    auto dev_pillar_count_histo_ct0 = dev_pillar_count_histo_;
    auto dev_counter_ct1 = dev_counter_;
    auto dev_pillar_count_ct2 = dev_pillar_count_;
    auto dev_x_coors_for_sub_ct5 = dev_x_coors_for_sub_;
    auto dev_y_coors_for_sub_ct6 = dev_y_coors_for_sub_;
    auto max_num_pillars_ct9 = max_num_pillars_;
    auto max_num_points_per_pillar_ct10 = max_num_points_per_pillar_;
    auto grid_x_size_ct11 = grid_x_size_;
    auto min_x_range_ct12 = min_x_range_;
    auto min_y_range_ct13 = min_y_range_;
    auto pillar_x_size_ct14 = pillar_x_size_;
    auto pillar_y_size_ct15 = pillar_y_size_;

    h.parallel_for(sycl::range<2>{static_cast<unsigned long>(grid_x_size_), static_cast<unsigned long>(grid_y_size_)},
                   [=](sycl::id<2> it) {
                     const int x = it[0];
                     const int y = it[1];
                     MakePillarIndexKernel(dev_pillar_count_histo_ct0, dev_counter_ct1, dev_pillar_count_ct2,
                                           dev_x_coors, dev_y_coors, dev_x_coors_for_sub_ct5, dev_y_coors_for_sub_ct6,
                                           dev_num_points_per_pillar, dev_sparse_pillar_map, max_num_pillars_ct9,
                                           max_num_points_per_pillar_ct10, grid_x_size_ct11, min_x_range_ct12,
                                           min_y_range_ct13, pillar_x_size_ct14, pillar_y_size_ct15, x, y);
                   });
  });
  queue.wait();

  queue.memcpy(host_pillar_count, dev_pillar_count_, sizeof(int)).wait();

  // Generate the first 4 pillar features in the input feature map.
  // This is a list of points up to max_num_points_per_pillar
  queue.submit([&](auto &h) {
    auto dev_pillar_x_in_coors_ct0 = dev_pillar_x_in_coors_;
    auto dev_pillar_y_in_coors_ct1 = dev_pillar_y_in_coors_;
    auto dev_pillar_z_in_coors_ct2 = dev_pillar_z_in_coors_;
    auto dev_pillar_i_in_coors_ct3 = dev_pillar_i_in_coors_;
    auto max_num_points_per_pillar_ct11 = max_num_points_per_pillar_;
    auto grid_x_size_ct12 = grid_x_size_;

    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, host_pillar_count[0]) * sycl::range<3>(1, 1, max_num_points_per_pillar_),
                          sycl::range<3>(1, 1, max_num_points_per_pillar_)),
        [=](sycl::nd_item<3> item_ct1) {
          MakePillarFeatureKernel(dev_pillar_x_in_coors_ct0, dev_pillar_y_in_coors_ct1, dev_pillar_z_in_coors_ct2,
                                  dev_pillar_i_in_coors_ct3, dev_pillar_x, dev_pillar_y, dev_pillar_z, dev_pillar_i,
                                  dev_x_coors, dev_y_coors, dev_num_points_per_pillar, max_num_points_per_pillar_ct11,
                                  grid_x_size_ct12, item_ct1);
        });
  });
  queue.wait();

  // Generate the next features in the pillar input feature map: (pillar_center_x, pillar_center_y, pillar_mask)
  queue.submit([&](auto &h) {
    auto dev_x_coors_for_sub_ct0 = dev_x_coors_for_sub_;
    auto dev_y_coors_for_sub_ct1 = dev_y_coors_for_sub_;
    auto max_num_points_per_pillar_ct6 = max_num_points_per_pillar_;

    h.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, max_num_pillars_) * sycl::range<3>(1, 1, max_num_points_per_pillar_),
                          sycl::range<3>(1, 1, max_num_points_per_pillar_)),
        [=](sycl::nd_item<3> item_ct1) {
          MakeExtraNetworkInputKernel(dev_x_coors_for_sub_ct0, dev_y_coors_for_sub_ct1, dev_num_points_per_pillar,
                                      dev_x_coors_for_sub_shaped, dev_y_coors_for_sub_shaped, dev_pillar_feature_mask,
                                      max_num_points_per_pillar_ct6, item_ct1);
        });
  });
  queue.wait();
}
}  // namespace pointpillars
