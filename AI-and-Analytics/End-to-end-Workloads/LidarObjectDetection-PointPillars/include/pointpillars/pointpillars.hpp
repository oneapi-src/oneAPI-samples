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

#include <inference_engine.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "pointpillars/anchorgrid.hpp"
#include "pointpillars/pointpillars_config.hpp"
#include "pointpillars/pointpillars_util.hpp"
#include "pointpillars/postprocess.hpp"
#include "pointpillars/preprocess.hpp"
#include "pointpillars/scatter.hpp"

namespace pointpillars {

/**
 * PointPillar's Main Class
 *
 * This class encapsulates the complete end-to-end
 * implementation of PointPillars.
 *
 * Users only need to create an object and call 'Detect'.
 */
class PointPillars {
 protected:
  PointPillarsConfig config_;
  const float score_threshold_;
  const float nms_overlap_threshold_;
  const std::string pfe_model_file_;
  const std::string rpn_model_file_;
  const int max_num_pillars_;
  const int max_num_points_per_pillar_;
  const int pfe_output_size_;
  const int grid_x_size_;
  const int grid_y_size_;
  const int grid_z_size_;
  const int rpn_input_size_;
  const int num_cls_;
  const int num_anchor_x_inds_;
  const int num_anchor_y_inds_;
  const int num_anchor_r_inds_;
  const int num_anchor_;
  const int rpn_box_output_size_;
  const int rpn_cls_output_size_;
  const int rpn_dir_output_size_;
  const float pillar_x_size_;
  const float pillar_y_size_;
  const float pillar_z_size_;
  const float min_x_range_;
  const float min_y_range_;
  const float min_z_range_;
  const float max_x_range_;
  const float max_y_range_;
  const float max_z_range_;
  const int batch_size_;
  const int num_features_;
  const int num_threads_;
  const int num_box_corners_;
  const int num_output_box_feature_;

  int host_pillar_count_[1];

  int *dev_x_coors_;                  // Array that holds the coordinates of corresponding pillar in x
  int *dev_y_coors_;                  // Array that holds the coordinates of corresponding pillar in y
  float *dev_num_points_per_pillar_;  // Array that stores the number of points in the corresponding pillar
  int *dev_sparse_pillar_map_;  // Mask with values 0 or 1 that specifies if the corresponding pillar has points or not
  int *dev_cumsum_workspace_;  // Device variable used as temporary storage of the cumulative sum during the anchor mask
                               // creation

  // variables to store the pillar's points
  float *dev_pillar_x_;
  float *dev_pillar_y_;
  float *dev_pillar_z_;
  float *dev_pillar_i_;

  // variables to store the pillar coordinates in the pillar grid
  float *dev_x_coors_for_sub_shaped_;
  float *dev_y_coors_for_sub_shaped_;

  // Pillar mask used to ignore the features generated with empty pillars
  float *dev_pillar_feature_mask_;

  // Mask used to filter the anchors in regions with input points
  int *dev_anchor_mask_;

  // Device memory used to store the RPN input feature map after Scatter
  float *dev_scattered_feature_;

  // Device memory locations to store the object detections
  float *dev_filtered_box_;
  float *dev_filtered_score_;
  float *dev_multiclass_score_;
  int *dev_filtered_dir_;
  int *dev_filtered_class_id_;
  float *dev_box_for_nms_;
  int *dev_filter_count_;

  std::unique_ptr<PreProcess> preprocess_points_ptr_;
  std::unique_ptr<Scatter> scatter_ptr_;
  std::unique_ptr<PostProcess> postprocess_ptr_;
  std::unique_ptr<AnchorGrid> anchor_grid_ptr_;

 public:
  PointPillars() = delete;

  /**
  * @brief Constructor
  * @param[in] score_threshold Score threshold for filtering output
  * @param[in] nms_overlap_threshold IOU threshold for NMS
  * @param[in] config PointPillars net configuration file
  */
  PointPillars(const float score_threshold, const float nms_threshold, const PointPillarsConfig &config);

  ~PointPillars();

  /**
  * @brief Call PointPillars to perform the end-to-end object detection chain
  * @param[in] in_points_array Pointcloud array
  * @param[in] in_num_points Number of points
  * @param[in] detections Network output bounding box list
  * @details This is the main public interface to run the algorithm
  */
  void Detect(const float *in_points_array, const int in_num_points, std::vector<ObjectDetection> &detections);

 private:
  InferenceEngine::ExecutableNetwork pfe_exe_network_;
  std::map<std::string, float *> pfe_input_map_;
  InferenceEngine::ExecutableNetwork rpn_exe_network_;

  float *pfe_output_;
  float *rpn_1_output_;
  float *rpn_2_output_;
  float *rpn_3_output_;

  void InitComponents();

  /**
  * @brief Memory allocation for device memory
  * @details Called in the constructor
  */
  void DeviceMemoryMalloc();

  /**
  * @brief Preprocess points
  * @param[in] in_points_array pointcloud array
  * @param[in] in_num_points Number of points
  * @details Call oneAPI preprocess
  */
  void PreProcessing(const float *in_points_array, const int in_num_points);

  /**
  * @brief Setup the PFE executable network
  * @details Setup the PFE network
  */
  void SetupPfeNetwork();

  /**
  * @brief Setup the RPN executable network
  * @param[in] resizeInput If false, the network is not adapted to input size changes
  * @details Setup the RPN network
  */
  void SetupRpnNetwork(bool resize_input);
};
}  // namespace pointpillars
