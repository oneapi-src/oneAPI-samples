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

#include "PointPillars/operations/preprocess.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>
#include <iostream>
#include "PointPillars/operations/common.hpp"

namespace dnn {
void make_pillar_histo_kernel(const float *dev_points, float *dev_pillar_x_in_coors, float *dev_pillar_y_in_coors,
                              float *dev_pillar_z_in_coors, float *dev_pillar_i_in_coors,
                              int *pillar_count_histo,  // holds the point count on the cell
                              const int num_points, const int max_points_per_pillar, const int GRID_X_SIZE,
                              const int GRID_Y_SIZE, const int GRID_Z_SIZE, const float MIN_X_RANGE,
                              const float MIN_Y_RANGE, const float MIN_Z_RANGE, const float PILLAR_X_SIZE,
                              const float PILLAR_Y_SIZE, const float PILLAR_Z_SIZE, sycl::nd_item<3> item_ct1) {
  int pointIndex = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  if (pointIndex >= num_points) {
    return;
  }

  // Indexes in the pillar map
  int xIndex = sycl::floor((float)((dev_points[pointIndex * 4 + 0] - MIN_X_RANGE) / PILLAR_X_SIZE));
  int yIndex = sycl::floor((float)((dev_points[pointIndex * 4 + 1] - MIN_Y_RANGE) / PILLAR_Y_SIZE));
  int zIndex = sycl::floor((float)((dev_points[pointIndex * 4 + 2] - MIN_Z_RANGE) / PILLAR_Z_SIZE));

  // Check for withing grid ranges
  if (xIndex >= 0 && xIndex < GRID_X_SIZE && yIndex >= 0 && yIndex < GRID_Y_SIZE && zIndex >= 0 &&
      zIndex < GRID_Z_SIZE) {
    // increase the point count
    int count = dpct::atomic_fetch_add(&pillar_count_histo[yIndex * GRID_X_SIZE + xIndex], 1);
    if (count < max_points_per_pillar) {
      // pillar index
      int pillarIndex = yIndex * GRID_X_SIZE * max_points_per_pillar + xIndex * max_points_per_pillar;

      // pointPillarIndex is the m-point in the n-pillar
      int pointPillarIndex = pillarIndex + count;

      // add point to pillar data
      dev_pillar_x_in_coors[pointPillarIndex] = dev_points[pointIndex * 4 + 0];
      dev_pillar_y_in_coors[pointPillarIndex] = dev_points[pointIndex * 4 + 1];
      dev_pillar_z_in_coors[pointPillarIndex] = dev_points[pointIndex * 4 + 2];
      dev_pillar_i_in_coors[pointPillarIndex] = dev_points[pointIndex * 4 + 3];
    }
  }
}

void make_pillar_index_kernel(int *dev_pillar_count_histo, int *dev_counter, int *dev_pillar_count, int *dev_x_coors,
                              int *dev_y_coors, float *dev_x_coors_for_sub, float *dev_y_coors_for_sub,
                              float *dev_num_points_per_pillar, int *dev_sparse_pillar_map, const int max_pillars,
                              const int max_points_per_pillar, const int GRID_X_SIZE, const float MIN_X_RANGE,
                              const float MIN_Y_RANGE, const float PILLAR_X_SIZE, const float PILLAR_Y_SIZE,
                              const int x, const int y) {
  int num_points_at_this_pillar = dev_pillar_count_histo[y * GRID_X_SIZE + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = dpct::atomic_fetch_add(dev_counter, 1);
  if (count < max_pillars) {
    dpct::atomic_fetch_add(dev_pillar_count, 1);
    if (num_points_at_this_pillar >= max_points_per_pillar) {
      dev_num_points_per_pillar[count] = max_points_per_pillar;
    } else {
      dev_num_points_per_pillar[count] = num_points_at_this_pillar;
    }

    // grid coordinates of this pillar
    dev_x_coors[count] = x;
    dev_y_coors[count] = y;

    // metric position of this pillar
    dev_x_coors_for_sub[count] = x * PILLAR_X_SIZE + 0.5f * PILLAR_X_SIZE + MIN_X_RANGE;
    dev_y_coors_for_sub[count] = y * PILLAR_Y_SIZE + 0.5f * PILLAR_Y_SIZE + MIN_Y_RANGE;

    // map of pillars with at least one point
    dev_sparse_pillar_map[y * GRID_X_SIZE + x] = 1;
  }
}

void make_pillar_feature_kernel(float *dev_pillar_x_in_coors, float *dev_pillar_y_in_coors,
                                float *dev_pillar_z_in_coors, float *dev_pillar_i_in_coors, float *dev_pillar_x,
                                float *dev_pillar_y, float *dev_pillar_z, float *dev_pillar_i, int *dev_x_coors,
                                int *dev_y_coors, float *dev_num_points_per_pillar, const int max_points,
                                const int GRID_X_SIZE, sycl::nd_item<3> item_ct1) {
  int ith_pillar = item_ct1.get_group(2);
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = item_ct1.get_local_id(2);
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }
  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  int pillar_ind = ith_pillar * max_points + ith_point;
  int coors_ind = y_ind * GRID_X_SIZE * max_points + x_ind * max_points + ith_point;
  dev_pillar_x[pillar_ind] = dev_pillar_x_in_coors[coors_ind];
  dev_pillar_y[pillar_ind] = dev_pillar_y_in_coors[coors_ind];
  dev_pillar_z[pillar_ind] = dev_pillar_z_in_coors[coors_ind];
  dev_pillar_i[pillar_ind] = dev_pillar_i_in_coors[coors_ind];
}

void make_extra_network_input_kernel(float *dev_x_coors_for_sub, float *dev_y_coors_for_sub,
                                     float *dev_num_points_per_pillar, float *dev_x_coors_for_sub_shaped,
                                     float *dev_y_coors_for_sub_shaped, float *dev_pillar_feature_mask,
                                     const int MAX_NUM_POINTS_PER_PILLAR, sycl::nd_item<3> item_ct1) {
  int ith_pillar = item_ct1.get_group(2);
  int ith_point = item_ct1.get_local_id(2);
  float x = dev_x_coors_for_sub[ith_pillar];
  float y = dev_y_coors_for_sub[ith_pillar];
  int num_points_for_a_pillar = dev_num_points_per_pillar[ith_pillar];
  int ind = ith_pillar * MAX_NUM_POINTS_PER_PILLAR + ith_point;
  dev_x_coors_for_sub_shaped[ind] = x;
  dev_y_coors_for_sub_shaped[ind] = y;

  if (ith_point < num_points_for_a_pillar) {
    dev_pillar_feature_mask[ind] = 1.0f;
  } else {
    dev_pillar_feature_mask[ind] = 0.0f;
  }
}

PreProcess::PreProcess(const int MAX_NUM_PILLARS, const int MAX_POINTS_PER_PILLAR, const int GRID_X_SIZE,
                       const int GRID_Y_SIZE, const int GRID_Z_SIZE, const float PILLAR_X_SIZE,
                       const float PILLAR_Y_SIZE, const float PILLAR_Z_SIZE, const float MIN_X_RANGE,
                       const float MIN_Y_RANGE, const float MIN_Z_RANGE)
    : MAX_NUM_PILLARS_(MAX_NUM_PILLARS),
      MAX_NUM_POINTS_PER_PILLAR_(MAX_POINTS_PER_PILLAR),
      GRID_X_SIZE_(GRID_X_SIZE),
      GRID_Y_SIZE_(GRID_Y_SIZE),
      GRID_Z_SIZE_(GRID_Z_SIZE),
      PILLAR_X_SIZE_(PILLAR_X_SIZE),
      PILLAR_Y_SIZE_(PILLAR_Y_SIZE),
      PILLAR_Z_SIZE_(PILLAR_Z_SIZE),
      MIN_X_RANGE_(MIN_X_RANGE),
      MIN_Y_RANGE_(MIN_Y_RANGE),
      MIN_Z_RANGE_(MIN_Z_RANGE) {
  // allocate memory
  dev_pillar_x_in_coors_ =
      (float *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                   dpct::get_current_device(), dpct::get_default_context());
  dev_pillar_y_in_coors_ =
      (float *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                   dpct::get_current_device(), dpct::get_default_context());
  dev_pillar_z_in_coors_ =
      (float *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                   dpct::get_current_device(), dpct::get_default_context());
  dev_pillar_i_in_coors_ =
      (float *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                   dpct::get_current_device(), dpct::get_default_context());
  dev_pillar_count_histo_ = (int *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int),
                                                       dpct::get_current_device(), dpct::get_default_context());
  dev_counter_ = (int *)sycl::malloc_device(sizeof(int), dpct::get_current_device(), dpct::get_default_context());
  dev_pillar_count_ = (int *)sycl::malloc_device(sizeof(int), dpct::get_current_device(), dpct::get_default_context());
  dev_x_coors_for_sub_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * sizeof(float), dpct::get_current_device(),
                                                      dpct::get_default_context());
  dev_y_coors_for_sub_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * sizeof(float), dpct::get_current_device(),
                                                      dpct::get_default_context());
}

PreProcess::~PreProcess() {
  sycl::free(dev_pillar_x_in_coors_, dpct::get_default_context());
  sycl::free(dev_pillar_y_in_coors_, dpct::get_default_context());
  sycl::free(dev_pillar_z_in_coors_, dpct::get_default_context());
  sycl::free(dev_pillar_i_in_coors_, dpct::get_default_context());
  sycl::free(dev_pillar_count_histo_, dpct::get_default_context());
  sycl::free(dev_counter_, dpct::get_default_context());
  sycl::free(dev_pillar_count_, dpct::get_default_context());
  sycl::free(dev_x_coors_for_sub_, dpct::get_default_context());
  sycl::free(dev_y_coors_for_sub_, dpct::get_default_context());
}

void PreProcess::doPreProcess(const float *dev_points, const int in_num_points, int *dev_x_coors, int *dev_y_coors,
                              float *dev_num_points_per_pillar, float *dev_pillar_x, float *dev_pillar_y,
                              float *dev_pillar_z, float *dev_pillar_i, float *dev_x_coors_for_sub_shaped,
                              float *dev_y_coors_for_sub_shaped, float *dev_pillar_feature_mask,
                              int *dev_sparse_pillar_map, int *host_pillar_count) {
  dpct::get_default_queue()
      .memset(dev_pillar_x_in_coors_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_y_in_coors_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_z_in_coors_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_i_in_coors_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue().memset(dev_pillar_count_histo_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int)).wait();
  dpct::get_default_queue().memset(dev_counter_, 0, sizeof(int)).wait();
  dpct::get_default_queue().memset(dev_pillar_count_, 0, sizeof(int)).wait();
  dpct::get_default_queue().memset(dev_x_coors_for_sub_, 0, MAX_NUM_PILLARS_ * sizeof(float)).wait();
  dpct::get_default_queue().memset(dev_y_coors_for_sub_, 0, MAX_NUM_PILLARS_ * sizeof(float)).wait();

  int num_block = DIVUP(in_num_points, 256);
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto dev_pillar_x_in_coors__ct1 = dev_pillar_x_in_coors_;
    auto dev_pillar_y_in_coors__ct2 = dev_pillar_y_in_coors_;
    auto dev_pillar_z_in_coors__ct3 = dev_pillar_z_in_coors_;
    auto dev_pillar_i_in_coors__ct4 = dev_pillar_i_in_coors_;
    auto dev_pillar_count_histo__ct5 = dev_pillar_count_histo_;
    auto MAX_NUM_POINTS_PER_PILLAR__ct7 = MAX_NUM_POINTS_PER_PILLAR_;
    auto GRID_X_SIZE__ct8 = GRID_X_SIZE_;
    auto GRID_Y_SIZE__ct9 = GRID_Y_SIZE_;
    auto GRID_Z_SIZE__ct10 = GRID_Z_SIZE_;
    auto MIN_X_RANGE__ct11 = MIN_X_RANGE_;
    auto MIN_Y_RANGE__ct12 = MIN_Y_RANGE_;
    auto MIN_Z_RANGE__ct13 = MIN_Z_RANGE_;
    auto PILLAR_X_SIZE__ct14 = PILLAR_X_SIZE_;
    auto PILLAR_Y_SIZE__ct15 = PILLAR_Y_SIZE_;
    auto PILLAR_Z_SIZE__ct16 = PILLAR_Z_SIZE_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_block) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
          make_pillar_histo_kernel(dev_points, dev_pillar_x_in_coors__ct1, dev_pillar_y_in_coors__ct2,
                                   dev_pillar_z_in_coors__ct3, dev_pillar_i_in_coors__ct4, dev_pillar_count_histo__ct5,
                                   in_num_points, MAX_NUM_POINTS_PER_PILLAR__ct7, GRID_X_SIZE__ct8, GRID_Y_SIZE__ct9,
                                   GRID_Z_SIZE__ct10, MIN_X_RANGE__ct11, MIN_Y_RANGE__ct12, MIN_Z_RANGE__ct13,
                                   PILLAR_X_SIZE__ct14, PILLAR_Y_SIZE__ct15, PILLAR_Z_SIZE__ct16, item_ct1);
        });
  });

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto dev_pillar_count_histo__ct0 = dev_pillar_count_histo_;
    auto dev_counter__ct1 = dev_counter_;
    auto dev_pillar_count__ct2 = dev_pillar_count_;
    auto dev_x_coors_for_sub__ct5 = dev_x_coors_for_sub_;
    auto dev_y_coors_for_sub__ct6 = dev_y_coors_for_sub_;
    auto MAX_NUM_PILLARS__ct9 = MAX_NUM_PILLARS_;
    auto MAX_NUM_POINTS_PER_PILLAR__ct10 = MAX_NUM_POINTS_PER_PILLAR_;
    auto GRID_X_SIZE__ct11 = GRID_X_SIZE_;
    auto MIN_X_RANGE__ct12 = MIN_X_RANGE_;
    auto MIN_Y_RANGE__ct13 = MIN_Y_RANGE_;
    auto PILLAR_X_SIZE__ct14 = PILLAR_X_SIZE_;
    auto PILLAR_Y_SIZE__ct15 = PILLAR_Y_SIZE_;

    cgh.parallel_for(sycl::range<2>{static_cast<unsigned long>(GRID_X_SIZE_), static_cast<unsigned long>(GRID_Y_SIZE_)},
                     [=](sycl::id<2> it) {
                       const int x = it[0];
                       const int y = it[1];
                       make_pillar_index_kernel(dev_pillar_count_histo__ct0, dev_counter__ct1, dev_pillar_count__ct2,
                                                dev_x_coors, dev_y_coors, dev_x_coors_for_sub__ct5,
                                                dev_y_coors_for_sub__ct6, dev_num_points_per_pillar,
                                                dev_sparse_pillar_map, MAX_NUM_PILLARS__ct9,
                                                MAX_NUM_POINTS_PER_PILLAR__ct10, GRID_X_SIZE__ct11, MIN_X_RANGE__ct12,
                                                MIN_Y_RANGE__ct13, PILLAR_X_SIZE__ct14, PILLAR_Y_SIZE__ct15, x, y);
                     });
  });

  dpct::get_default_queue().memcpy(host_pillar_count, dev_pillar_count_, sizeof(int)).wait();
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto dev_pillar_x_in_coors__ct0 = dev_pillar_x_in_coors_;
    auto dev_pillar_y_in_coors__ct1 = dev_pillar_y_in_coors_;
    auto dev_pillar_z_in_coors__ct2 = dev_pillar_z_in_coors_;
    auto dev_pillar_i_in_coors__ct3 = dev_pillar_i_in_coors_;
    auto MAX_NUM_POINTS_PER_PILLAR__ct11 = MAX_NUM_POINTS_PER_PILLAR_;
    auto GRID_X_SIZE__ct12 = GRID_X_SIZE_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, host_pillar_count[0]) * sycl::range<3>(1, 1, MAX_NUM_POINTS_PER_PILLAR_),
                          sycl::range<3>(1, 1, MAX_NUM_POINTS_PER_PILLAR_)),
        [=](sycl::nd_item<3> item_ct1) {
          make_pillar_feature_kernel(dev_pillar_x_in_coors__ct0, dev_pillar_y_in_coors__ct1, dev_pillar_z_in_coors__ct2,
                                     dev_pillar_i_in_coors__ct3, dev_pillar_x, dev_pillar_y, dev_pillar_z, dev_pillar_i,
                                     dev_x_coors, dev_y_coors, dev_num_points_per_pillar,
                                     MAX_NUM_POINTS_PER_PILLAR__ct11, GRID_X_SIZE__ct12, item_ct1);
        });
  });

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto dev_x_coors_for_sub__ct0 = dev_x_coors_for_sub_;
    auto dev_y_coors_for_sub__ct1 = dev_y_coors_for_sub_;
    auto MAX_NUM_POINTS_PER_PILLAR__ct6 = MAX_NUM_POINTS_PER_PILLAR_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, MAX_NUM_PILLARS_) * sycl::range<3>(1, 1, MAX_NUM_POINTS_PER_PILLAR_),
                          sycl::range<3>(1, 1, MAX_NUM_POINTS_PER_PILLAR_)),
        [=](sycl::nd_item<3> item_ct1) {
          make_extra_network_input_kernel(dev_x_coors_for_sub__ct0, dev_y_coors_for_sub__ct1, dev_num_points_per_pillar,
                                          dev_x_coors_for_sub_shaped, dev_y_coors_for_sub_shaped,
                                          dev_pillar_feature_mask, MAX_NUM_POINTS_PER_PILLAR__ct6, item_ct1);
        });
  });
}
}
