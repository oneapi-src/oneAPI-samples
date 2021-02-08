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

#include "PointPillars/operations/scatter.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>

namespace dnn {
void scatter_kernel(int *x_coors, int *y_coors, float *pfe_output, float *scattered_feature, const int MAX_NUM_PILLARS_,
                    const int GRID_X_SIZE, const int GRID_Y_SIZE, sycl::nd_item<3> item_ct1) {
  int i_pillar = item_ct1.get_group(2);
  int i_feature = item_ct1.get_local_id(2);  // is accessing the feature by
                                             // thread id, which forces feature
                                             // = thread nums
  int x_ind = x_coors[i_pillar];
  int y_ind = y_coors[i_pillar];
  float feature = pfe_output[i_feature * MAX_NUM_PILLARS_ + i_pillar];
  scattered_feature[i_feature * GRID_Y_SIZE * GRID_X_SIZE + y_ind * GRID_X_SIZE + x_ind] = feature;
}

Scatter::Scatter(const int NUM_THREADS, const int MAX_NUM_PILLARS, const int GRID_X_SIZE, const int GRID_Y_SIZE)
    : NUM_THREADS_(NUM_THREADS),
      MAX_NUM_PILLARS_(MAX_NUM_PILLARS),
      GRID_X_SIZE_(GRID_X_SIZE),
      GRID_Y_SIZE_(GRID_Y_SIZE) {}

void Scatter::doScatter(const int pillar_count, int *x_coors, int *y_coors, float *pfe_output,
                        float *scattered_feature) {
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto MAX_NUM_PILLARS__ct4 = MAX_NUM_PILLARS_;
    auto GRID_X_SIZE__ct5 = GRID_X_SIZE_;
    auto GRID_Y_SIZE__ct6 = GRID_Y_SIZE_;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, pillar_count) * sycl::range<3>(1, 1, NUM_THREADS_),
                                       sycl::range<3>(1, 1, NUM_THREADS_)),
                     [=](sycl::nd_item<3> item_ct1) {
                       scatter_kernel(x_coors, y_coors, pfe_output, scattered_feature, MAX_NUM_PILLARS__ct4,
                                      GRID_X_SIZE__ct5, GRID_Y_SIZE__ct6, item_ct1);
                     });
  });
}
}
