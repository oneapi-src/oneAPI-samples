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

#include "pointpillars/pointpillars.hpp"
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
#include "devicemanager/devicemanager.hpp"

namespace pointpillars {
PointPillars::PointPillars(const float score_threshold, const float nms_threshold, const PointPillarsConfig &config)
    : config_(config),
      score_threshold_(score_threshold),
      nms_overlap_threshold_(nms_threshold),
      pfe_model_file_(config.pfe_model_file),
      rpn_model_file_(config.rpn_model_file),
      max_num_pillars_(config.max_num_pillars),
      max_num_points_per_pillar_(config.max_num_points_per_pillar),
      pfe_output_size_(config.max_num_pillars * config.pillar_features),
      grid_x_size_(config.grid_x_size),
      grid_y_size_(config.grid_y_size),
      grid_z_size_(config.grid_z_size),
      rpn_input_size_(config.pillar_features * config.grid_x_size * config.grid_y_size),
      num_cls_(config.num_classes),
      num_anchor_x_inds_(config.grid_x_size * config.rpn_scale),
      num_anchor_y_inds_(config.grid_y_size * config.rpn_scale),
      num_anchor_r_inds_(2),
      num_anchor_(num_anchor_x_inds_ * num_anchor_y_inds_ * num_anchor_r_inds_ * num_cls_),
      rpn_box_output_size_(num_anchor_ * 7),         // feature score
      rpn_cls_output_size_(num_anchor_ * num_cls_),  // classification score
      rpn_dir_output_size_(num_anchor_ * 2),         // orientation score
      pillar_x_size_(config.pillar_x_size),
      pillar_y_size_(config.pillar_y_size),
      pillar_z_size_(config.pillar_z_size),
      min_x_range_(config.min_x_range),
      min_y_range_(config.min_y_range),
      min_z_range_(config.min_z_range),
      max_x_range_(config.max_x_range),
      max_y_range_(config.max_y_range),
      max_z_range_(config.max_x_range),
      batch_size_(1),
      num_features_(64),  // number of pillar features
      num_threads_(64),
      num_box_corners_(4),
      num_output_box_feature_(7) {
  AnchorGridConfig anchor_grid_config;

  anchor_grid_config.min_x_range = config.min_x_range;
  anchor_grid_config.max_x_range = config.max_x_range;
  anchor_grid_config.min_y_range = config.min_y_range;
  anchor_grid_config.max_y_range = config.max_y_range;
  anchor_grid_config.min_z_range = config.min_z_range;
  anchor_grid_config.max_z_range = config.max_z_range;

  anchor_grid_config.x_stride = config.x_stride;
  anchor_grid_config.y_stride = config.y_stride;

  anchor_grid_config.anchors = config.anchors;
  anchor_grid_config.rotations = {0.f, M_PI_2};

  anchor_grid_ptr_ = std::make_unique<AnchorGrid>(anchor_grid_config);

  InitComponents();
  DeviceMemoryMalloc();

  SetupPfeNetwork();
  SetupRpnNetwork(true);
}

PointPillars::~PointPillars() {
  // Upon destruction clear all SYCL memory
  sycl::queue queue = devicemanager::GetCurrentQueue();
  sycl::free(dev_x_coors_, queue);
  sycl::free(dev_y_coors_, queue);
  sycl::free(dev_num_points_per_pillar_, queue);
  sycl::free(dev_sparse_pillar_map_, queue);
  sycl::free(dev_pillar_x_, queue);
  sycl::free(dev_pillar_y_, queue);
  sycl::free(dev_pillar_z_, queue);
  sycl::free(dev_pillar_i_, queue);
  sycl::free(dev_x_coors_for_sub_shaped_, queue);
  sycl::free(dev_y_coors_for_sub_shaped_, queue);
  sycl::free(dev_pillar_feature_mask_, queue);
  sycl::free(dev_cumsum_workspace_, queue);
  sycl::free(dev_anchor_mask_, queue);

  sycl::free(dev_scattered_feature_, queue);
  sycl::free(dev_filtered_box_, queue);
  sycl::free(dev_filtered_score_, queue);
  sycl::free(dev_filtered_dir_, queue);
  sycl::free(dev_filtered_class_id_, queue);
  sycl::free(dev_box_for_nms_, queue);
  sycl::free(dev_filter_count_, queue);
}

void PointPillars::InitComponents() {
  // Setup anchor grid
  anchor_grid_ptr_->GenerateAnchors();

  // Setup preprocessing
  preprocess_points_ptr_ = std::make_unique<PreProcess>(max_num_pillars_, max_num_points_per_pillar_, grid_x_size_,
                                                        grid_y_size_, grid_z_size_, pillar_x_size_, pillar_y_size_,
                                                        pillar_z_size_, min_x_range_, min_y_range_, min_z_range_);

  // Setup scatter
  scatter_ptr_ = std::make_unique<Scatter>(num_features_, max_num_pillars_, grid_x_size_, grid_y_size_);

  const float float_min = std::numeric_limits<float>::lowest();
  const float float_max = std::numeric_limits<float>::max();

  // Setup postprocessing
  postprocess_ptr_ = std::make_unique<PostProcess>(float_min, float_max, num_anchor_x_inds_, num_anchor_y_inds_,
                                                   num_anchor_r_inds_, num_cls_, score_threshold_, num_threads_,
                                                   nms_overlap_threshold_, num_box_corners_, num_output_box_feature_);
}

void PointPillars::SetupPfeNetwork() {
  std::size_t num_threads = 1;
  std::string device = "CPU";
  if (devicemanager::GetCurrentDevice().is_cpu()) {
    // get number of threads supported by the current CPU
    num_threads = std::thread::hardware_concurrency();
  }

  // if the chosen exceution device is GPU, also use this for OpenVINO
  if (devicemanager::GetCurrentDevice().is_gpu()) {
    device = "GPU";
  }

  // Setup InferenceEngine and load the network file
  InferenceEngine::Core inference_engine;
  auto network = inference_engine.ReadNetwork(pfe_model_file_ + ".onnx");

  // Setup device configuration
  if (device == "CPU") {
    inference_engine.SetConfig({{CONFIG_KEY(CPU_THREADS_NUM), std::to_string(num_threads)}}, device);
  }

  // Setup network input configuration
  // The PillarFeatureExtraction (PFE) network has multiple inputs, which all use FP32 precision
  for (auto &item : network.getInputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    if (item.first == "num_points_per_pillar") {
      item.second->setLayout(InferenceEngine::Layout::NC);
    } else {
      item.second->setLayout(InferenceEngine::Layout::NCHW);
    }
  }

  // Setup network output configuration
  // The PillarFeatureExtraction (PFE) network has one output, which uses FP32 precision
  for (auto &item : network.getOutputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  // Finally load the network onto the execution device
  pfe_exe_network_ = inference_engine.LoadNetwork(network, device);

  // create map of network inputs to memory objects
  pfe_input_map_.insert({"pillar_x", dev_pillar_x_});
  pfe_input_map_.insert({"pillar_y", dev_pillar_y_});
  pfe_input_map_.insert({"pillar_z", dev_pillar_z_});
  pfe_input_map_.insert({"pillar_i", dev_pillar_i_});
  pfe_input_map_.insert({"num_points_per_pillar", dev_num_points_per_pillar_});
  pfe_input_map_.insert({"x_sub_shaped", dev_x_coors_for_sub_shaped_});
  pfe_input_map_.insert({"y_sub_shaped", dev_y_coors_for_sub_shaped_});
  pfe_input_map_.insert({"mask", dev_pillar_feature_mask_});
}

void PointPillars::SetupRpnNetwork(bool resize_input) {
  std::size_t num_threads = 1;
  std::string device = "CPU";
  if (devicemanager::GetCurrentDevice().is_cpu()) {
    // get number of threads supported by the current CPU
    num_threads = std::thread::hardware_concurrency();
  }

  // if the chosen exceution device is GPU, also use this for OpenVINO
  if (devicemanager::GetCurrentDevice().is_gpu()) {
    device = "GPU";
  }

  // Setup InferenceEngine and load the network file
  InferenceEngine::Core inference_engine;
  auto network = inference_engine.ReadNetwork(rpn_model_file_ + ".onnx");

  // Setup device configuration
  if (device == "CPU") {
    inference_engine.SetConfig({{CONFIG_KEY(CPU_THREADS_NUM), std::to_string(num_threads)}}, device);
  }

  // Setup network input configuration
  // The RegionProposalNetwork (RPN) network has one input, which uses FP32 precision
  for (auto &item : network.getInputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  // A resizing of the RPN input is possible and required depening on the pillar and LiDAR configuration
  if (resize_input) {
    auto input_shapes = network.getInputShapes();
    std::string input_name;
    InferenceEngine::SizeVector input_shape;
    std::tie(input_name, input_shape) = *input_shapes.begin();  // only one input
    input_shape[0] = 1;                                         // set batch size to the first input dimension
    input_shape[1] = config_.pillar_features;                   // set batch size to the first input dimension
    input_shape[2] = config_.grid_y_size;                       // changes input height to the image one
    input_shape[3] = config_.grid_x_size;                       // changes input width to the image one

    input_shapes[input_name] = input_shape;
    network.reshape(input_shapes);
  }

  // Setup network output configuration
  // The PillarFeatureExtraction (PFE) network has multiple outputs, which all use FP32 precision
  for (auto &item : network.getOutputsInfo()) {
    item.second->setPrecision(InferenceEngine::Precision::FP32);
    item.second->setLayout(InferenceEngine::Layout::NCHW);
  }

  // Finally load the network onto the execution device
  rpn_exe_network_ = inference_engine.LoadNetwork(network, device);
}

void PointPillars::DeviceMemoryMalloc() {
  sycl::queue queue = devicemanager::GetCurrentQueue();

  // Allocate all device memory vector
  dev_x_coors_ = sycl::malloc_device<int>(max_num_pillars_, queue);
  dev_y_coors_ = sycl::malloc_device<int>(max_num_pillars_, queue);
  dev_num_points_per_pillar_ = sycl::malloc_device<float>(max_num_pillars_, queue);
  dev_sparse_pillar_map_ = sycl::malloc_device<int>(grid_y_size_ * grid_x_size_, queue);

  dev_pillar_x_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_y_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_z_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_i_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);

  dev_x_coors_for_sub_shaped_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_y_coors_for_sub_shaped_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_feature_mask_ = sycl::malloc_device<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);

  // cumsum kernel
  dev_cumsum_workspace_ = sycl::malloc_device<int>(grid_y_size_ * grid_x_size_, queue);

  // for make anchor mask kernel
  dev_anchor_mask_ = sycl::malloc_device<int>(num_anchor_, queue);

  // for scatter kernel
  dev_scattered_feature_ = sycl::malloc_device<float>(num_features_ * grid_y_size_ * grid_x_size_, queue);

  // for filter
  dev_filtered_box_ = sycl::malloc_device<float>(num_anchor_ * num_output_box_feature_, queue);
  dev_filtered_score_ = sycl::malloc_device<float>(num_anchor_, queue);
  dev_multiclass_score_ = sycl::malloc_device<float>(num_anchor_ * num_cls_, queue);
  dev_filtered_dir_ = sycl::malloc_device<int>(num_anchor_, queue);
  dev_filtered_class_id_ = sycl::malloc_device<int>(num_anchor_, queue);
  dev_box_for_nms_ = sycl::malloc_device<float>(num_anchor_ * num_box_corners_, queue);
  dev_filter_count_ = sycl::malloc_device<int>(sizeof(int), queue);

  // CNN outputs
  pfe_output_ = sycl::malloc_device<float>(pfe_output_size_, queue);
  rpn_1_output_ = sycl::malloc_device<float>(rpn_box_output_size_, queue);
  rpn_2_output_ = sycl::malloc_device<float>(rpn_cls_output_size_, queue);
  rpn_3_output_ = sycl::malloc_device<float>(rpn_dir_output_size_, queue);
}

void PointPillars::PreProcessing(const float *in_points_array, const int in_num_points) {
  float *dev_points;

  sycl::queue queue = devicemanager::GetCurrentQueue();

  // Before starting the PreProcessing, the device memory has to be reset
  dev_points = sycl::malloc_device<float>(in_num_points * num_box_corners_, queue);
  queue.memcpy(dev_points, in_points_array, in_num_points * num_box_corners_ * sizeof(float));
  queue.memset(dev_sparse_pillar_map_, 0, grid_y_size_ * grid_x_size_ * sizeof(int));
  queue.memset(dev_pillar_x_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_y_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_z_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_i_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_x_coors_, 0, max_num_pillars_ * sizeof(int));
  queue.memset(dev_y_coors_, 0, max_num_pillars_ * sizeof(int));
  queue.memset(dev_num_points_per_pillar_, 0, max_num_pillars_ * sizeof(float));
  queue.memset(dev_anchor_mask_, 0, num_anchor_ * sizeof(int));
  queue.memset(dev_cumsum_workspace_, 0, grid_y_size_ * grid_x_size_ * sizeof(int));

  queue.memset(dev_x_coors_for_sub_shaped_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_y_coors_for_sub_shaped_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));
  queue.memset(dev_pillar_feature_mask_, 0, max_num_pillars_ * max_num_points_per_pillar_ * sizeof(float));

  // wait until all memory operations were completed
  queue.wait();

  // Run the PreProcessing operations and generate the input feature map
  preprocess_points_ptr_->DoPreProcess(dev_points, in_num_points, dev_x_coors_, dev_y_coors_,
                                       dev_num_points_per_pillar_, dev_pillar_x_, dev_pillar_y_, dev_pillar_z_,
                                       dev_pillar_i_, dev_x_coors_for_sub_shaped_, dev_y_coors_for_sub_shaped_,
                                       dev_pillar_feature_mask_, dev_sparse_pillar_map_, host_pillar_count_);

  // remove no longer required memory
  sycl::free(dev_points, devicemanager::GetCurrentQueue());
}

void PointPillars::Detect(const float *in_points_array, const int in_num_points,
                          std::vector<ObjectDetection> &detections) {
  // Run the PointPillar detection algorthim

  // reset the detections
  detections.clear();

  sycl::queue queue = devicemanager::GetCurrentQueue();

  std::cout << "Starting PointPillars\n";
  std::cout << "   PreProcessing";

  // First run the preprocessing to convert the LiDAR pointcloud into the required pillar format
  const auto t0 = std::chrono::high_resolution_clock::now();
  PreProcessing(in_points_array, in_num_points);
  const auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms\n";

  // 2nd step is to create the anchor mask used to optimize the decoding of the RegionProposalNetwork output
  std::cout << "   AnchorMask";
  anchor_grid_ptr_->CreateAnchorMask(dev_sparse_pillar_map_, grid_y_size_, grid_x_size_, pillar_x_size_, pillar_y_size_,
                                     dev_anchor_mask_, dev_cumsum_workspace_);
  const auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";

  // 3rd step is to execture the PillarFeatureExtraction (PFE) network
  // Therefore, first an inference request using OpenVINO has to be created
  // Then, the required input data has to transfered from the SYCL device to be accessible by OpenVINO
  // Next, the inference can be excuted
  // At last, the results have to be copied back to the SYCL device
  std::cout << "   PFE Inference";
  InferenceEngine::InferRequest::Ptr infer_request_ptr = pfe_exe_network_.CreateInferRequestPtr();
  // Fill input tensor with data
  for (auto &item : pfe_exe_network_.GetInputsInfo()) {
    // Make PFE input data be accessible by OpenVINO
    InferenceEngine::Blob::Ptr input_blob = infer_request_ptr->GetBlob(item.first);
    InferenceEngine::SizeVector dims = input_blob->getTensorDesc().getDims();
    auto data =
        input_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    if (devicemanager::GetCurrentDevice().is_gpu()) {
      // copying data directly with SYCL to the input_blob buffer does not work for discrete accelerators
      // copy the data first to the host, and then to the input_blob buffer
      float *temp_buffer_;
      temp_buffer_ = new float[input_blob->size()];
      queue.memcpy(temp_buffer_, pfe_input_map_[item.first], input_blob->size() * sizeof(float)).wait();
      memcpy(data, temp_buffer_, input_blob->size() * sizeof(float));
    } else {
      queue.memcpy(data, pfe_input_map_[item.first], input_blob->size() * sizeof(float)).wait();
    }
  }

  // Launch the inference
  infer_request_ptr->StartAsync();

  // Wait for the inference to finish
  auto inference_result_status = infer_request_ptr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  if (InferenceEngine::OK != inference_result_status) {
    throw std::runtime_error("PFE Inference failed");
  }
  const auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms\n";

  // get PFE inference ouput
  auto output_name = pfe_exe_network_.GetOutputsInfo().begin()->first;
  auto output_blob = infer_request_ptr->GetBlob(output_name)->buffer().as<float *>();

  // 4th step: Perform scatter operation, i.e. convert from pillar features to top view image-like features
  std::cout << "   Scattering";
  queue.memcpy(pfe_output_, output_blob, pfe_output_size_ * sizeof(float));
  queue.memset(dev_scattered_feature_, 0, rpn_input_size_ * sizeof(float));
  queue.wait();
  scatter_ptr_->DoScatter(host_pillar_count_[0], dev_x_coors_, dev_y_coors_, pfe_output_, dev_scattered_feature_);

  const auto t4 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms\n";

  std::cout << "   RPN Inference";
  // 5th step is to execute the RegionProposal (RPN) network
  // Therefore, first an inference request using OpenVINO has to be created
  // Then, the required input data has to transfered from the SYCL device to be accessible by OpenVINO
  // Next, the inference can be excuted
  // At last, the results have to be copied back to the SYCL device
  infer_request_ptr = rpn_exe_network_.CreateInferRequestPtr();
  auto input_blob = infer_request_ptr->GetBlob(rpn_exe_network_.GetInputsInfo().begin()->first);
  // Fill input tensor with data
  auto data =
      input_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
  if (devicemanager::GetCurrentDevice().is_gpu()) {
    // copying data directly with SYCL to the input_blob buffer does not work for discrete accelerators
    // copy the data first to the host, and then to the input_blob buffer
    float *temp_buffer_;
    temp_buffer_ = new float[input_blob->size()];
    queue.memcpy(temp_buffer_, dev_scattered_feature_, input_blob->size() * sizeof(float)).wait();
    memcpy(data, temp_buffer_, input_blob->size() * sizeof(float));
  } else {
    queue.memcpy(data, dev_scattered_feature_, input_blob->size() * sizeof(float)).wait();
  }

  // Start the inference and wait for the results
  infer_request_ptr->StartAsync();
  inference_result_status = infer_request_ptr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  if (InferenceEngine::OK != inference_result_status) {
    throw std::runtime_error("RPN Inference failed");
  }

  // copy inference data to SYCL device
  std::vector<InferenceEngine::Blob::Ptr> output_blobs;
  for (auto &output : rpn_exe_network_.GetOutputsInfo()) {
    output_blobs.push_back(infer_request_ptr->GetBlob(output.first));
  }

  queue.memcpy(
      rpn_1_output_,
      output_blobs[0]->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(),
      rpn_box_output_size_ * sizeof(float));
  queue.memcpy(
      rpn_2_output_,
      output_blobs[1]->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(),
      rpn_cls_output_size_ * sizeof(float));
  queue.memcpy(
      rpn_3_output_,
      output_blobs[2]->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(),
      rpn_dir_output_size_ * sizeof(float));

  queue.memset(dev_filter_count_, 0, sizeof(int));
  queue.wait();

  const auto t5 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << "ms\n";

  std::cout << "   Postprocessing";
  // Last step is to run the PostProcessing operation
  postprocess_ptr_->DoPostProcess(
      rpn_1_output_, rpn_2_output_, rpn_3_output_, dev_anchor_mask_, anchor_grid_ptr_->dev_anchors_px_,
      anchor_grid_ptr_->dev_anchors_py_, anchor_grid_ptr_->dev_anchors_pz_, anchor_grid_ptr_->dev_anchors_dx_,
      anchor_grid_ptr_->dev_anchors_dy_, anchor_grid_ptr_->dev_anchors_dz_, anchor_grid_ptr_->dev_anchors_ro_,
      dev_multiclass_score_, dev_filtered_box_, dev_filtered_score_, dev_filtered_dir_, dev_filtered_class_id_,
      dev_box_for_nms_, dev_filter_count_, detections);

  const auto t6 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << "ms\n";

  std::cout << "Done\n";
}
}  // namespace pointpillars
