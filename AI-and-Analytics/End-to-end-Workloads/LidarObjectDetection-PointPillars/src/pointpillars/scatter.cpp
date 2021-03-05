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

#include "pointpillars/scatter.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include "devicemanager/devicemanager.hpp"

namespace pointpillars {
void ScatterKernel(int *x_coors, int *y_coors, float *pfe_output, float *scattered_feature, const int max_num_pillars_,
                   const int grid_x_size, const int grid_y_size, sycl::nd_item<3> item_ct1) {
  // Get pillar index and feature index from current group and local id
  int i_pillar = item_ct1.get_group(2);
  int i_feature = item_ct1.get_local_id(2);  // is accessing the feature by
                                             // thread id, which forces feature
                                             // = thread nums
  // Read (x,y) indices in the sparse feature map of the corresponding pillar
  int x_ind = x_coors[i_pillar];
  int y_ind = y_coors[i_pillar];
  float feature = pfe_output[i_feature * max_num_pillars_ + i_pillar];

  // Copy the i feature from pillar to the sparse feature map
  scattered_feature[i_feature * grid_y_size * grid_x_size + y_ind * grid_x_size + x_ind] = feature;
}

Scatter::Scatter(const int num_features, const int max_num_pillars, const int grid_x_size, const int grid_y_size)
    : num_features_(num_features),
      max_num_pillars_(max_num_pillars),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size) {}

void Scatter::DoScatter(const int pillar_count, int *x_coors, int *y_coors, float *pfe_output,
                        float *scattered_feature) {
  // Launch the scatter kernel on each (n-pillar , m-feature)
  sycl::queue queue = devicemanager::GetCurrentQueue();
  queue.submit([&](auto &h) {
    auto max_num_pillars_ct4 = max_num_pillars_;
    auto grid_x_size_ct5 = grid_x_size_;
    auto grid_y_size_ct6 = grid_y_size_;

    h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, pillar_count) * sycl::range<3>(1, 1, num_features_),
                                     sycl::range<3>(1, 1, num_features_)),
                   [=](sycl::nd_item<3> item_ct1) {
                     ScatterKernel(x_coors, y_coors, pfe_output, scattered_feature, max_num_pillars_ct4,
                                   grid_x_size_ct5, grid_y_size_ct6, item_ct1);
                   });
  });

  queue.wait();
}
}  // namespace pointpillars
