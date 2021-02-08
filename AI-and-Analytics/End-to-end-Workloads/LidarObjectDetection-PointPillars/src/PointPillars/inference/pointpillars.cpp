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

#include "PointPillars/inference/pointpillars.hpp"
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
#include "PointPillars/operations/common.hpp"

namespace dnn {
// clang-format off
PointPillars::PointPillars(const float score_threshold, const float nms_threshold, const PointPillarsConfig &config)
  : mConfig(config)
  , score_threshold_(score_threshold)
  , nms_overlap_threshold_(nms_threshold)
  , pfe_model_file_(config.pfeModelFile)
  , rpn_model_file_(config.rpnModelFile)
  , MAX_NUM_PILLARS_(config.maxNumPillars)
  , MAX_NUM_POINTS_PER_PILLAR_(config.maxNumPointsPerPillar)
  , PFE_OUTPUT_SIZE_(config.maxNumPillars * config.pillarFeatures)
  , GRID_X_SIZE_(config.gridXSize)
  , GRID_Y_SIZE_(config.gridYSize)
  , GRID_Z_SIZE_(config.gridZSize)
  , RPN_INPUT_SIZE_(config.pillarFeatures * config.gridXSize * config.gridYSize)
  , NUM_CLS_(config.numClasses)
  , NUM_ANCHOR_X_INDS_(config.gridXSize * config.rpnScale)
  , NUM_ANCHOR_Y_INDS_(config.gridYSize * config.rpnScale)
  , NUM_ANCHOR_R_INDS_(2)
  , NUM_ANCHOR_(NUM_ANCHOR_X_INDS_ * NUM_ANCHOR_Y_INDS_ * NUM_ANCHOR_R_INDS_ * NUM_CLS_)
  , RPN_BOX_OUTPUT_SIZE_(NUM_ANCHOR_ * 7) // feature score
  , RPN_CLS_OUTPUT_SIZE_(NUM_ANCHOR_ * NUM_CLS_) // classification score
  , RPN_DIR_OUTPUT_SIZE_(NUM_ANCHOR_ * 2) // orientation score
  , PILLAR_X_SIZE_(config.pillarXSize)
  , PILLAR_Y_SIZE_(config.pillarYSize)
  , PILLAR_Z_SIZE_(config.pillarZSize)
  , MIN_X_RANGE_(config.minXRange)
  , MIN_Y_RANGE_(config.minYRange)
  , MIN_Z_RANGE_(config.minZRange)
  , MAX_X_RANGE_(config.maxXRange)
  , MAX_Y_RANGE_(config.maxYRange)
  , MAX_Z_RANGE_(config.maxXRange)
  , BATCH_SIZE_(1)
  , NUM_FEATURES_(64) // Number of pillar features
  , NUM_THREADS_(64) // if you change NUM_THREADS_, need to modify NUM_THREADS_MACRO in common.h
  , NUM_BOX_CORNERS_(4)
  , NUM_OUTPUT_BOX_FEATURE_(7)
{
  dnn::AnchorGridConfig anchorGridConfig;

  anchorGridConfig.minXRange = config.minXRange;
  anchorGridConfig.maxXRange = config.maxXRange;
  anchorGridConfig.minYRange = config.minYRange;
  anchorGridConfig.maxYRange = config.maxYRange;
  anchorGridConfig.minZRange = config.minZRange;
  anchorGridConfig.maxZRange = config.maxZRange;

  anchorGridConfig.xStride = config.xStride;
  anchorGridConfig.yStride = config.yStride;

  anchorGridConfig.anchors = config.anchors;
  anchorGridConfig.rotations = {0.f, M_PI_2};

  mAnchorGridPtr = std::make_unique<AnchorGrid>(anchorGridConfig);

  initComponents();
  deviceMemoryMalloc();  
  
  setupPfeNetwork();
  setupRpnNetwork(true);    
}
// clang-format on

PointPillars::~PointPillars() {
  sycl::free(dev_x_coors_, dpct::get_default_context());
  sycl::free(dev_y_coors_, dpct::get_default_context());
  sycl::free(dev_num_points_per_pillar_, dpct::get_default_context());
  sycl::free(dev_sparse_pillar_map_, dpct::get_default_context());
  sycl::free(dev_pillar_x_, dpct::get_default_context());
  sycl::free(dev_pillar_y_, dpct::get_default_context());
  sycl::free(dev_pillar_z_, dpct::get_default_context());
  sycl::free(dev_pillar_i_, dpct::get_default_context());
  sycl::free(dev_x_coors_for_sub_shaped_, dpct::get_default_context());
  sycl::free(dev_y_coors_for_sub_shaped_, dpct::get_default_context());
  sycl::free(dev_pillar_feature_mask_, dpct::get_default_context());
  sycl::free(dev_cumsum_workspace_, dpct::get_default_context());
  sycl::free(dev_anchor_mask_, dpct::get_default_context());

  sycl::free(dev_scattered_feature_, dpct::get_default_context());
  sycl::free(dev_filtered_box_, dpct::get_default_context());
  sycl::free(dev_filtered_score_, dpct::get_default_context());
  sycl::free(dev_filtered_dir_, dpct::get_default_context());
  sycl::free(dev_filtered_class_id_, dpct::get_default_context());
  sycl::free(dev_box_for_nms_, dpct::get_default_context());
  sycl::free(dev_filter_count_, dpct::get_default_context());
}

void PointPillars::initComponents() {
  mAnchorGridPtr->generateAnchors();
  mAnchorGridPtr->moveAnchorsToDevice();

  preprocess_points_ptr_ = std::make_unique<PreProcess>(MAX_NUM_PILLARS_, MAX_NUM_POINTS_PER_PILLAR_, GRID_X_SIZE_,
                                                        GRID_Y_SIZE_, GRID_Z_SIZE_, PILLAR_X_SIZE_, PILLAR_Y_SIZE_,
                                                        PILLAR_Z_SIZE_, MIN_X_RANGE_, MIN_Y_RANGE_, MIN_Z_RANGE_);

  scatter_ptr_ = std::make_unique<Scatter>(NUM_THREADS_, MAX_NUM_PILLARS_, GRID_X_SIZE_, GRID_Y_SIZE_);

  const float FLOAT_MIN = std::numeric_limits<float>::lowest();
  const float FLOAT_MAX = std::numeric_limits<float>::max();

  postprocess_ptr_ = std::make_unique<PostProcess>(FLOAT_MIN, FLOAT_MAX, NUM_ANCHOR_X_INDS_, NUM_ANCHOR_Y_INDS_,
                                                   NUM_ANCHOR_R_INDS_, NUM_CLS_, score_threshold_, NUM_THREADS_,
                                                   nms_overlap_threshold_, NUM_BOX_CORNERS_, NUM_OUTPUT_BOX_FEATURE_);
}

void PointPillars::setupPfeNetwork() {
  std::size_t numThreads = std::thread::hardware_concurrency();
  std::string device = "CPU";
  if (dpct::get_current_device().is_host()) {
    numThreads = 1;
  }

  if (dpct::get_current_device().is_gpu()) {
    device = "GPU";
  }

  InferenceEngine::Core inferenceEngine;
  auto network = inferenceEngine.ReadNetwork(pfe_model_file_ + ".onnx");

  if (device == "CPU") {
    inferenceEngine.SetConfig({{CONFIG_KEY(CPU_THREADS_NUM), std::to_string(numThreads)}}, device);
  }

  for (auto &item : network.getInputsInfo()) {
    std::string inputName = item.first;
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    if (inputName == "num_points_per_pillar") {
      item.second->setLayout(InferenceEngine::Layout::NC);
    } else {
      item.second->setLayout(InferenceEngine::Layout::NCHW);
    }
  }

  for (auto &item : network.getOutputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  mPfeExeNetwork = inferenceEngine.LoadNetwork(network, device);

  // create map of network inputs to memory objects
  mPfeInputMap.insert({"pillar_x", dev_pillar_x_});
  mPfeInputMap.insert({"pillar_y", dev_pillar_y_});
  mPfeInputMap.insert({"pillar_z", dev_pillar_z_});
  mPfeInputMap.insert({"pillar_i", dev_pillar_i_});
  mPfeInputMap.insert({"num_points_per_pillar", dev_num_points_per_pillar_});
  mPfeInputMap.insert({"x_sub_shaped", dev_x_coors_for_sub_shaped_});
  mPfeInputMap.insert({"y_sub_shaped", dev_y_coors_for_sub_shaped_});
  mPfeInputMap.insert({"mask", dev_pillar_feature_mask_});
}

void PointPillars::setupRpnNetwork(bool resizeInput) {
  std::size_t numThreads = std::thread::hardware_concurrency();
  std::string device = "CPU";
  if (dpct::get_current_device().is_host()) {
    numThreads = 1;
  }

  if (dpct::get_current_device().is_gpu()) {
    device = "GPU";
  }

  InferenceEngine::Core inferenceEngine;
  auto network = inferenceEngine.ReadNetwork(rpn_model_file_ + ".onnx");

  if (device == "CPU") {
    inferenceEngine.SetConfig({{CONFIG_KEY(CPU_THREADS_NUM), std::to_string(numThreads)}}, device);
  }

  for (auto &item : network.getInputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  if (resizeInput) {
    auto inputShapes = network.getInputShapes();
    std::string inputName;
    InferenceEngine::SizeVector inputShape;
    std::tie(inputName, inputShape) = *inputShapes.begin();  // only one input
    inputShape[0] = 1;                                       // set batch size to the first input dimension
    inputShape[1] = mConfig.pillarFeatures;                  // set batch size to the first input dimension
    inputShape[2] = mConfig.gridYSize;                       // changes input height to the image one
    inputShape[3] = mConfig.gridXSize;                       // changes input width to the image one

    inputShapes[inputName] = inputShape;
    network.reshape(inputShapes);
  }

  for (auto &item : network.getOutputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  mRpnExeNetwork = inferenceEngine.LoadNetwork(network, device);
}

void PointPillars::deviceMemoryMalloc() {
  dev_x_coors_ = (int *)sycl::malloc_device(MAX_NUM_PILLARS_ * sizeof(int), dpct::get_current_device(),
                                            dpct::get_default_context());
  dev_y_coors_ = (int *)sycl::malloc_device(MAX_NUM_PILLARS_ * sizeof(int), dpct::get_current_device(),
                                            dpct::get_default_context());
  dev_num_points_per_pillar_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * sizeof(float),
                                                            dpct::get_current_device(), dpct::get_default_context());
  dev_sparse_pillar_map_ = (int *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int),
                                                      dpct::get_current_device(), dpct::get_default_context());

  dev_pillar_x_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                               dpct::get_current_device(), dpct::get_default_context());

  dev_pillar_y_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                               dpct::get_current_device(), dpct::get_default_context());

  dev_pillar_z_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                               dpct::get_current_device(), dpct::get_default_context());

  dev_pillar_i_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),
                                               dpct::get_current_device(), dpct::get_default_context());

  // clang-format off
  dev_x_coors_for_sub_shaped_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  dev_y_coors_for_sub_shaped_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  dev_pillar_feature_mask_ = (float *)sycl::malloc_device(MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), dpct::get_current_device(), dpct::get_default_context());
  // clang-format on

  // cumsum kernel
  dev_cumsum_workspace_ = (int *)sycl::malloc_device(GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int),
                                                     dpct::get_current_device(), dpct::get_default_context());

  // for make anchor mask kernel
  dev_anchor_mask_ =
      (int *)sycl::malloc_device(NUM_ANCHOR_ * sizeof(int), dpct::get_current_device(), dpct::get_default_context());

  // for scatter kernel
  dev_scattered_feature_ = (float *)sycl::malloc_device(NUM_FEATURES_ * GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(float),
                                                        dpct::get_current_device(), dpct::get_default_context());

  // for filter
  dev_filtered_box_ = (float *)sycl::malloc_device(NUM_ANCHOR_ * NUM_OUTPUT_BOX_FEATURE_ * sizeof(float),
                                                   dpct::get_current_device(), dpct::get_default_context());

  dev_filtered_score_ = (float *)sycl::malloc_device(NUM_ANCHOR_ * sizeof(float), dpct::get_current_device(),
                                                     dpct::get_default_context());
  dev_multiclass_score_ = (float *)sycl::malloc_device(NUM_ANCHOR_ * NUM_CLS_ * sizeof(float),
                                                       dpct::get_current_device(), dpct::get_default_context());
  dev_filtered_dir_ =
      (int *)sycl::malloc_device(NUM_ANCHOR_ * sizeof(int), dpct::get_current_device(), dpct::get_default_context());
  dev_filtered_class_id_ =
      (int *)sycl::malloc_device(NUM_ANCHOR_ * sizeof(int), dpct::get_current_device(), dpct::get_default_context());
  dev_box_for_nms_ = (float *)sycl::malloc_device(NUM_ANCHOR_ * NUM_BOX_CORNERS_ * sizeof(float),
                                                  dpct::get_current_device(), dpct::get_default_context());
  dev_filter_count_ = (int *)sycl::malloc_device(sizeof(int), dpct::get_current_device(), dpct::get_default_context());

  pfe_output_ = (float *)sycl::malloc_device(PFE_OUTPUT_SIZE_ * sizeof(float), dpct::get_current_device(),
                                             dpct::get_default_context());
  rpn_1_output_ = (float *)sycl::malloc_device(RPN_BOX_OUTPUT_SIZE_ * sizeof(float), dpct::get_current_device(),
                                               dpct::get_default_context());
  rpn_2_output_ = (float *)sycl::malloc_device(RPN_CLS_OUTPUT_SIZE_ * sizeof(float), dpct::get_current_device(),
                                               dpct::get_default_context());
  rpn_3_output_ = (float *)sycl::malloc_device(RPN_DIR_OUTPUT_SIZE_ * sizeof(float), dpct::get_current_device(),
                                               dpct::get_default_context());
}

void PointPillars::preprocess(const float *in_points_array, const int in_num_points) {
  float *dev_points;

  dev_points = (float *)sycl::malloc_device(in_num_points * NUM_BOX_CORNERS_ * sizeof(float),
                                            dpct::get_current_device(), dpct::get_default_context());
  dpct::get_default_queue()
      .memcpy(dev_points, in_points_array, in_num_points * NUM_BOX_CORNERS_ * sizeof(float))
      .wait();
  dpct::get_default_queue().memset(dev_sparse_pillar_map_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int)).wait();
  dpct::get_default_queue()
      .memset(dev_pillar_x_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_y_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_z_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_i_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue().memset(dev_x_coors_, 0, MAX_NUM_PILLARS_ * sizeof(int)).wait();
  dpct::get_default_queue().memset(dev_y_coors_, 0, MAX_NUM_PILLARS_ * sizeof(int)).wait();
  dpct::get_default_queue().memset(dev_num_points_per_pillar_, 0, MAX_NUM_PILLARS_ * sizeof(float)).wait();
  dpct::get_default_queue().memset(dev_anchor_mask_, 0, NUM_ANCHOR_ * sizeof(int)).wait();
  dpct::get_default_queue().memset(dev_cumsum_workspace_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int)).wait();

  dpct::get_default_queue()
      .memset(dev_x_coors_for_sub_shaped_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_y_coors_for_sub_shaped_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memset(dev_pillar_feature_mask_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float))
      .wait();

  preprocess_points_ptr_->doPreProcess(dev_points, in_num_points, dev_x_coors_, dev_y_coors_,
                                       dev_num_points_per_pillar_, dev_pillar_x_, dev_pillar_y_, dev_pillar_z_,
                                       dev_pillar_i_, dev_x_coors_for_sub_shaped_, dev_y_coors_for_sub_shaped_,
                                       dev_pillar_feature_mask_, dev_sparse_pillar_map_, host_pillar_count_);

  sycl::free(dev_points, dpct::get_default_context());
}

void PointPillars::detect(const float *in_points_array, const int in_num_points,
                          std::vector<ObjectDetection> &detections) {
  detections.clear();

  std::cout << "Starting PointPillars\n";
  std::cout << "   PreProcessing";

  std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
  preprocess(in_points_array, in_num_points);
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms\n";

  std::cout << "   AnchorMask";
  mAnchorGridPtr->createAnchorMask(dev_sparse_pillar_map_, GRID_Y_SIZE_, GRID_X_SIZE_, PILLAR_X_SIZE_, PILLAR_Y_SIZE_,
                                   dev_anchor_mask_, dev_cumsum_workspace_);
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";

  std::cout << "   PFE Inference";
  InferenceEngine::InferRequest::Ptr inferRequestPtr = mPfeExeNetwork.CreateInferRequestPtr();
  // Fill input tensor with data
  for (auto &item : mPfeExeNetwork.GetInputsInfo()) {
    InferenceEngine::Blob::Ptr inputBlob = inferRequestPtr->GetBlob(item.first);
    InferenceEngine::SizeVector dims = inputBlob->getTensorDesc().getDims();
    auto data =
        inputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    if (dpct::get_current_device().is_gpu()) {
      // copying data directly with SYCL to the inputBlob buffer does not work for discrete accelerators
      // copy the data first to the host, and then to the inputBlob buffer
      float *temp_buffer_;
      temp_buffer_ = new float[inputBlob->size()];
      dpct::get_default_queue()
          .memcpy(temp_buffer_, mPfeInputMap[item.first], inputBlob->size() * sizeof(float))
          .wait();
      memcpy(data, temp_buffer_, inputBlob->size() * sizeof(float));
    } else {
      dpct::get_default_queue().memcpy(data, mPfeInputMap[item.first], inputBlob->size() * sizeof(float)).wait();
    }
  }

  inferRequestPtr->StartAsync();
  auto inferenceResultStatus = inferRequestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  if (InferenceEngine::OK != inferenceResultStatus) {
    throw std::runtime_error("PFE Inference failed");
  }
  std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms\n";

  // get inference ouput
  auto outputName = mPfeExeNetwork.GetOutputsInfo().begin()->first;
  auto output_blob = inferRequestPtr->GetBlob(outputName)->buffer().as<float *>();

  std::cout << "   Scattering";
  // copy inference data to SYCL device
  dpct::get_default_queue().memcpy(pfe_output_, output_blob, PFE_OUTPUT_SIZE_ * sizeof(float)).wait();

  // perform scatter
  dpct::get_default_queue().memset(dev_scattered_feature_, 0, RPN_INPUT_SIZE_ * sizeof(float)).wait();
  scatter_ptr_->doScatter(host_pillar_count_[0], dev_x_coors_, dev_y_coors_, pfe_output_, dev_scattered_feature_);

  std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms\n";

  std::cout << "   RPN Inference";
  // 2nd inference
  inferRequestPtr = mRpnExeNetwork.CreateInferRequestPtr();
  auto inputBlob = inferRequestPtr->GetBlob(mRpnExeNetwork.GetInputsInfo().begin()->first);
  // Fill input tensor with data
  auto data = inputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
  if (dpct::get_current_device().is_gpu()) {
    // copying data directly with SYCL to the inputBlob buffer does not work for discrete accelerators
    // copy the data first to the host, and then to the inputBlob buffer
    float *temp_buffer_;
    temp_buffer_ = new float[inputBlob->size()];
    dpct::get_default_queue().memcpy(temp_buffer_, dev_scattered_feature_, inputBlob->size() * sizeof(float)).wait();
    memcpy(data, temp_buffer_, inputBlob->size() * sizeof(float));
  } else {
    dpct::get_default_queue().memcpy(data, dev_scattered_feature_, inputBlob->size() * sizeof(float)).wait();
  }

  inferRequestPtr->StartAsync();
  inferenceResultStatus = inferRequestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  if (InferenceEngine::OK != inferenceResultStatus) {
    throw std::runtime_error("RPN Inference failed");
  }

  std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << "ms\n";

  std::vector<InferenceEngine::Blob::Ptr> output_blobs;
  for (auto &output : mRpnExeNetwork.GetOutputsInfo()) {
    output_blobs.push_back(inferRequestPtr->GetBlob(output.first));
  }

  std::cout << "   Postprocessing";
  // postprocessing
  // copy inference data to SYCL device
  dpct::get_default_queue()
      .memcpy(rpn_1_output_, output_blobs[0]
                                 ->buffer()
                                 .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(),
              RPN_BOX_OUTPUT_SIZE_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memcpy(rpn_2_output_, output_blobs[1]
                                 ->buffer()
                                 .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(),
              RPN_CLS_OUTPUT_SIZE_ * sizeof(float))
      .wait();
  dpct::get_default_queue()
      .memcpy(rpn_3_output_, output_blobs[2]
                                 ->buffer()
                                 .as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(),
              RPN_DIR_OUTPUT_SIZE_ * sizeof(float))
      .wait();

  dpct::get_default_queue().memset(dev_filter_count_, 0, sizeof(int)).wait();
  postprocess_ptr_->doPostProcess(
      rpn_1_output_, rpn_2_output_, rpn_3_output_, dev_anchor_mask_, mAnchorGridPtr->mDevAnchorsPx,
      mAnchorGridPtr->mDevAnchorsPy, mAnchorGridPtr->mDevAnchorsPz, mAnchorGridPtr->mDevAnchorsDx,
      mAnchorGridPtr->mDevAnchorsDy, mAnchorGridPtr->mDevAnchorsDz, mAnchorGridPtr->mDevAnchorsRo,
      dev_multiclass_score_, dev_filtered_box_, dev_filtered_score_, dev_filtered_dir_, dev_filtered_class_id_,
      dev_box_for_nms_, dev_filter_count_, detections);

  std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << "ms\n";

  std::cout << "Done\n";
}
}
