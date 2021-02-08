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

#pragma once

#include "PointPillars/operations/common.hpp"

namespace dnn {

class NMS {
 private:
  const int NUM_THREADS_;
  const int NUM_BOX_CORNERS_;
  const float nms_overlap_threshold_;

 public:
  /**
  * @brief Constructor
  * @param[in] NUM_THREADS Number of threads when launching kernel
  * @param[in] NUM_BOX_CORNERS Number of corners for 2D box
  * @param[in] nms_overlap_threshold IOU threshold for NMS
  * @details Captital variables never change after the compile, Non-captital
  * variables could be chaned through rosparam
  */
  NMS(const int NUM_THREADS, const int NUM_BOX_CORNERS, const float nms_overlap_threshold);

  /**
  * @brief Non-Maximum Suppresion for network output
  * @param[in] host_filter_count Number of filtered output
  * @param[in] dev_sorted_box_for_nms Bounding box output sorted by score
  * @param[out] out_keep_inds Indexes of selected bounding box
  * @param[out] out_num_to_keep Number of kept bounding boxes
  */
  void doNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
             size_t &out_num_to_keep);

 private:
  /**
   * @brief Parallel Non-Maximum Suppresion for network output using SYCL or GPU
   * @details Parallel NMS and postprocessing for selecting box
   */
  void parallelNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                   size_t &out_num_to_keep);

  /**
   * @brief Sequential Non-Maximum Suppresion for network output in CPU
   */
  void sequentialNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                     size_t &out_num_to_keep);
};
}
