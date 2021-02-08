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
#include <string>
#include <vector>
#include "PointPillars/PointPillarsConfig.hpp"
#include "PointPillars/PointPillarsUtil.hpp"
#include "PointPillars/operations/anchorgrid.hpp"
#include "PointPillars/operations/postprocess.hpp"
#include "PointPillars/operations/preprocess.hpp"
#include "PointPillars/operations/scatter.hpp"

namespace dnn {

class PointPillars {
 protected:
  PointPillarsConfig mConfig;
  const float score_threshold_;
  const float nms_overlap_threshold_;
  const std::string pfe_model_file_;
  const std::string rpn_model_file_;
  const int MAX_NUM_PILLARS_;
  const int MAX_NUM_POINTS_PER_PILLAR_;
  const int PFE_OUTPUT_SIZE_;
  const int GRID_X_SIZE_;
  const int GRID_Y_SIZE_;
  const int GRID_Z_SIZE_;
  const int RPN_INPUT_SIZE_;
  const int NUM_CLS_;
  const int NUM_ANCHOR_X_INDS_;
  const int NUM_ANCHOR_Y_INDS_;
  const int NUM_ANCHOR_R_INDS_;
  const int NUM_ANCHOR_;
  const int RPN_BOX_OUTPUT_SIZE_;
  const int RPN_CLS_OUTPUT_SIZE_;
  const int RPN_DIR_OUTPUT_SIZE_;
  const float PILLAR_X_SIZE_;
  const float PILLAR_Y_SIZE_;
  const float PILLAR_Z_SIZE_;
  const float MIN_X_RANGE_;
  const float MIN_Y_RANGE_;
  const float MIN_Z_RANGE_;
  const float MAX_X_RANGE_;
  const float MAX_Y_RANGE_;
  const float MAX_Z_RANGE_;
  const int BATCH_SIZE_;
  const int NUM_FEATURES_;
  const int NUM_THREADS_;
  const int NUM_BOX_CORNERS_;
  const int NUM_OUTPUT_BOX_FEATURE_;

  int host_pillar_count_[1];

  int *dev_x_coors_;
  int *dev_y_coors_;
  float *dev_num_points_per_pillar_;
  int *dev_sparse_pillar_map_;
  int *dev_cumsum_workspace_;

  float *dev_pillar_x_;
  float *dev_pillar_y_;
  float *dev_pillar_z_;
  float *dev_pillar_i_;

  float *dev_x_coors_for_sub_shaped_;
  float *dev_y_coors_for_sub_shaped_;
  float *dev_pillar_feature_mask_;

  int *dev_anchor_mask_;

  float *dev_scattered_feature_;

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
  std::unique_ptr<AnchorGrid> mAnchorGridPtr;

 public:
  PointPillars() = delete;

  /**
  * @brief Constructor
  * @param[in] score_threshold Score threshold for filtering output
  * @param[in] nms_overlap_threshold IOU threshold for NMS
  * @param[in] config Point Pillars net configuration file
  * @details Variables could be chaned through rosparam
  */
  PointPillars(const float score_threshold, const float nms_threshold, const PointPillarsConfig &config);

  ~PointPillars();

  /**
  * @brief Call PointPillars to perform the end-to-end object detection chain
  * @param[in] in_points_array Pointcloud array
  * @param[in] in_num_points Number of points
  * @param[in] detections Network output bounding box list
  * @details This is an interface for the algorithm
  */
  void detect(const float *in_points_array, const int in_num_points, std::vector<ObjectDetection> &detections);

 private:
  InferenceEngine::ExecutableNetwork mPfeExeNetwork;
  std::map<std::string, float *> mPfeInputMap;
  InferenceEngine::ExecutableNetwork mRpnExeNetwork;

  float *pfe_output_;
  float *rpn_1_output_;
  float *rpn_2_output_;
  float *rpn_3_output_;

  void initComponents();

  /**
  * @brief Memory allocation for device memory
  * @details Called in the constructor
  */
  void deviceMemoryMalloc();

  /**
  * @brief Preprocess points
  * @param[in] in_points_array pointcloud array
  * @param[in] in_num_points Number of points
  * @details Call oneAPI preprocess
  */
  void preprocess(const float *in_points_array, const int in_num_points);

  /**
  * @brief Setup the PFE executable network
  * @details Setup the PFE network
  */
  void setupPfeNetwork();

  /**
  * @brief Setup the RPN executable network
  * @param[in] resizeInput If false, the network is not adapted to input size changes
  * @details Setup the RPN network
  */
  void setupRpnNetwork(bool resizeInput);
};
}
