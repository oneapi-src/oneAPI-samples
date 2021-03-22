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

#include <CL/sycl.hpp>
#include "pointpillars/common.hpp"

namespace pointpillars {

/**
 * Non-Maximum-Suppression
 *
 * Non-maximum suppression (NMS) is a way to eliminate points that do not lie in the important edges
 * of detected data. Here NMS is used to filter out overlapping object detections. Therefore, an
 * intersection-over-union (IOU) approach is used to caculate the overlap of two objects. At the end,
 * only the most relevant objects are kept.
 */
class NMS {
 private:
  const int num_threads_;              // Number of threads used to execute the NMS kernel
  const int num_box_corners_;          // Number of corners of a 2D box
  const float nms_overlap_threshold_;  // Threshold below which objects are discarded

 public:
  /**
  * @brief Constructor
  * @param[in] num_threads Number of threads when launching kernel
  * @param[in] num_box_corners Number of corners for 2D box
  * @param[in] nms_overlap_threshold IOU threshold for NMS
  */
  NMS(const int num_threads, const int num_box_corners, const float nms_overlap_threshold);

  /**
  * @brief Execute Non-Maximum Suppresion for network output
  * @param[in] host_filter_count Number of filtered output
  * @param[in] dev_sorted_box_for_nms Bounding box output sorted by score
  * @param[out] out_keep_inds Indexes of selected bounding box
  * @param[out] out_num_to_keep Number of kept bounding boxes
  */
  void DoNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
             size_t &out_num_to_keep);

 private:
  /**
   * @brief Parallel Non-Maximum Suppresion for network output using SYCL GPU
   * @details Parallel NMS and postprocessing for selecting box
   */
  void ParallelNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                   size_t &out_num_to_keep);

  /**
   * @brief Sequential Non-Maximum Suppresion for network output in SYCL CPU or Host device
   */
  void SequentialNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                     size_t &out_num_to_keep);
};
}  // namespace pointpillars
