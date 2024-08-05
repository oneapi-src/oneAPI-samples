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

#include <sycl/backend/opencl.hpp>

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

  // If the chosen execution device is GPU, also use this for OpenVINO
  if (devicemanager::GetCurrentDevice().is_gpu()) {
    device = "GPU";
  }

  // Enable model caching and read the model
  ov::Core core;
  core.set_property(ov::cache_dir("pointpillars_cache"));
  auto model = core.read_model(pfe_model_file_);

  // Configure input pre-processing
  ov::preprocess::PrePostProcessor ppp(model);
  for (auto input : model->inputs()) {
    auto& input_info = ppp.input(input.get_any_name());
    ov::Layout layout = "NCHW";
    if (input.get_any_name() == "num_points_per_pillar") {
      layout = "NC";
    }
    input_info.tensor()
            .set_element_type(ov::element::f32)
            .set_layout(layout);
    input_info.model().set_layout(layout);
  }
  model = ppp.build();

  // Setup device configuration and load the network onto the execution device
  if (device == "CPU") {
    pfe_exe_network_ = core.compile_model(model, device,
                                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  } else {
    assert(device == "GPU");

    auto queue = sycl::get_native<sycl::backend::opencl>(devicemanager::GetCurrentQueue());
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, queue);
    pfe_exe_network_ = core.compile_model(model, remote_context,
                                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  }

  // Create the inference request
  const auto if_req_t1 = std::chrono::high_resolution_clock::now();
  pfe_infer_request_ = pfe_exe_network_.create_infer_request();
  const auto if_req_t2 = std::chrono::high_resolution_clock::now();
  std::cout << "    PFE InferRequest create " << std::chrono::duration_cast<std::chrono::milliseconds>(if_req_t2 - if_req_t1).count() << "ms\n";

  // create map of network inputs to memory objects
  pfe_input_map_.insert({"pillar_x", dev_pillar_x_});
  pfe_input_map_.insert({"pillar_y", dev_pillar_y_});
  pfe_input_map_.insert({"pillar_z", dev_pillar_z_});
  pfe_input_map_.insert({"pillar_i", dev_pillar_i_});
  pfe_input_map_.insert({"num_points_per_pillar", dev_num_points_per_pillar_});
  pfe_input_map_.insert({"x_sub_shaped", dev_x_coors_for_sub_shaped_});
  pfe_input_map_.insert({"y_sub_shaped", dev_y_coors_for_sub_shaped_});
  pfe_input_map_.insert({"mask", dev_pillar_feature_mask_});

  if (devicemanager::GetCurrentDevice().is_cpu()) {
    for (auto &input : pfe_exe_network_.inputs()) {
      auto input_tensor = ov::Tensor(input.get_element_type(), input.get_shape(),
                                     pfe_input_map_[input.get_any_name()]);
      pfe_input_tensor_map_.insert({input.get_any_name(), input_tensor});
    }
    auto output = pfe_exe_network_.output();
    pfe_output_tensor_ = ov::Tensor(output.get_element_type(), output.get_shape(), pfe_output_);
  }
  else {
    auto remote_context = pfe_exe_network_.get_context()
        .as<ov::intel_gpu::ocl::ClContext>();

    for (auto &input : pfe_exe_network_.inputs()) {
      ov::RemoteTensor input_tensor =
          remote_context.create_tensor(input.get_element_type(), input.get_shape(),
                                    pfe_input_map_[input.get_any_name()]);
      pfe_input_tensor_map_.insert({input.get_any_name(), input_tensor});
    }
    auto output = pfe_exe_network_.output();
    pfe_output_tensor_ = remote_context.create_tensor(
        output.get_element_type(), output.get_shape(), pfe_output_);
  }
}

void PointPillars::SetupRpnNetwork(bool resize_input) {
  std::size_t num_threads = 1;
  std::string device = "CPU";

  // if the chosen exceution device is GPU, also use this for OpenVINO
  if (devicemanager::GetCurrentDevice().is_gpu()) {
    device = "GPU";
  }

  // Enable model caching and read the model
  ov::Core core;
  core.set_property(ov::cache_dir("pointpillars_cache"));
  auto model = core.read_model(rpn_model_file_);

  // Configure input pre-processing
  ov::preprocess::PrePostProcessor ppp(model);
  auto& input = ppp.input();
  input.tensor()
          .set_element_type(ov::element::f32)
          .set_layout("NCHW");
  input.model().set_layout("NCHW");

  if (resize_input) {
    model->reshape({1, config_.pillar_features, config_.grid_y_size, config_.grid_x_size});
  }
  model = ppp.build();

  // Setup device configuration and load the network onto the execution device
  if (device == "CPU") {
    rpn_exe_network_ = core.compile_model(model, device,
                                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  } else {
    assert(device == "GPU");

    auto queue = sycl::get_native<sycl::backend::opencl>(devicemanager::GetCurrentQueue());
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, queue);
    rpn_exe_network_ = core.compile_model(model, remote_context,
                                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  }

  // Create the inference request
  const auto if_req_t1 = std::chrono::high_resolution_clock::now();
  rpn_infer_request_ = rpn_exe_network_.create_infer_request();
  const auto if_req_t2 = std::chrono::high_resolution_clock::now();
  std::cout << "    RPN InferRequest create " << std::chrono::duration_cast<std::chrono::milliseconds>(if_req_t2 - if_req_t1).count() << "ms\n";

  if (devicemanager::GetCurrentDevice().is_cpu()) {
    auto input_info = rpn_exe_network_.input();
    scattered_feature_tensor_ =
        ov::Tensor(input_info.get_element_type(), input_info.get_shape(), dev_scattered_feature_);

    int i = 0;
    float* outputs[] = {rpn_1_output_, rpn_2_output_, rpn_3_output_};
    for (auto& output : rpn_exe_network_.outputs()) {
      rpn_output_tensors_[i] =
          ov::Tensor(output.get_element_type(), output.get_shape(), outputs[i]);
      i++;
    }
  }
  else {
    auto remote_context = rpn_exe_network_.get_context()
        .as<ov::intel_gpu::ocl::ClContext>();

    auto input_info = rpn_exe_network_.input();
    scattered_feature_tensor_ =
        remote_context.create_tensor(input_info.get_element_type(), input_info.get_shape(), dev_scattered_feature_);

    int i = 0;
    float* outputs[] = {rpn_1_output_, rpn_2_output_, rpn_3_output_};
    for (auto& output : rpn_exe_network_.outputs()) {
      rpn_output_tensors_[i] = remote_context.create_tensor(output.get_element_type(), output.get_shape(), outputs[i]);
      i++;
    }
  }
}

void PointPillars::DeviceMemoryMalloc() {
  sycl::queue queue = devicemanager::GetCurrentQueue();

  // Allocate all device memory vector
  dev_x_coors_ = sycl::malloc_device<int>(max_num_pillars_, queue);
  dev_y_coors_ = sycl::malloc_device<int>(max_num_pillars_, queue);
  dev_num_points_per_pillar_ = sycl::malloc_shared<float>(max_num_pillars_, queue);
  dev_sparse_pillar_map_ = sycl::malloc_device<int>(grid_y_size_ * grid_x_size_, queue);

  dev_pillar_x_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_y_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_z_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_i_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);

  dev_x_coors_for_sub_shaped_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_y_coors_for_sub_shaped_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);
  dev_pillar_feature_mask_ = sycl::malloc_shared<float>(max_num_pillars_ * max_num_points_per_pillar_, queue);

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

  if (!devicemanager::GetCurrentDevice().is_gpu()) {
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
  } else {
    // For GPU, the queue.memset waste time, using GPU kernel to assign the value use less.
    auto e = queue.submit([&](auto &h){
      auto dev_pillar_x_auto = dev_pillar_x_;
      auto dev_pillar_y_auto = dev_pillar_y_;
      auto dev_pillar_z_auto = dev_pillar_z_;
      auto dev_pillar_i_auto = dev_pillar_i_;
      auto dev_x_coors_for_sub_shaped_auto = dev_x_coors_for_sub_shaped_;
      auto dev_y_coors_for_sub_shaped_auto = dev_y_coors_for_sub_shaped_;
      auto dev_pillar_feature_mask_auto = dev_pillar_feature_mask_;

      auto x = max_num_pillars_;
      auto y = max_num_points_per_pillar_;

      h.parallel_for(sycl::range<1>(y*x),[=](auto index) {
        dev_pillar_x_auto[index] = 0;
        dev_pillar_y_auto[index] = 0;
        dev_pillar_z_auto[index] = 0;
        dev_pillar_i_auto[index] = 0;
        dev_x_coors_for_sub_shaped_auto[index] = 0;
        dev_y_coors_for_sub_shaped_auto[index] = 0;
        dev_pillar_feature_mask_auto[index] = 0;
      });
    });
    e.wait();
    queue.memset(dev_sparse_pillar_map_, 0, grid_y_size_ * grid_x_size_ * sizeof(int));
    queue.memset(dev_x_coors_, 0, max_num_pillars_ * sizeof(int));
    queue.memset(dev_y_coors_, 0, max_num_pillars_ * sizeof(int));
    queue.memset(dev_num_points_per_pillar_, 0, max_num_pillars_ * sizeof(float));
    queue.memset(dev_anchor_mask_, 0, num_anchor_ * sizeof(int));
    queue.memset(dev_cumsum_workspace_, 0, grid_y_size_ * grid_x_size_ * sizeof(int));
    queue.wait();
  }

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
  // The required input data has to transfered from the SYCL device to be accessible by OpenVINO
  // Next, the inference can be excuted
  // At last, the results have to be copied back to the SYCL device
  std::cout << "   PFE Inference";

  // Fill input tensor with data
  for (auto &input : pfe_exe_network_.inputs()) {
    // Make PFE input data be accessible by OpenVINO
    // TODO: test inefficiency in GPU & CPU
    pfe_infer_request_.set_tensor(input.get_any_name(), pfe_input_tensor_map_[input.get_any_name()]);
  }
  auto pfe_output_name = pfe_exe_network_.output().get_any_name();
  pfe_infer_request_.set_tensor(pfe_output_name, pfe_output_tensor_);

  // Launch the inference
  pfe_infer_request_.start_async();

  cl_command_queue q = sycl::get_native<sycl::backend::opencl>(devicemanager::GetCurrentQueue());
  //clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  // Wait for the inference to finish
  pfe_infer_request_.wait();
  const auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms\n";

  // get PFE inference output
//  auto output_name = pfe_exe_network_.output().get_any_name();
//  auto output_tensor = pfe_infer_request_.get_tensor(pfe_output_name);
//  auto output_blob = output_tensor.data<const float>();

  // 4th step: Perform scatter operation, i.e. convert from pillar features to top view image-like features
  std::cout << "   Scattering";
//  queue.memcpy(pfe_output_, output_blob, pfe_output_size_ * sizeof(float));

  if (!devicemanager::GetCurrentDevice().is_gpu()) {
    queue.memset(dev_scattered_feature_, 0, rpn_input_size_ * sizeof(float));
    queue.wait();
  } else {
    // For GPU, the queue.memset waste time, using GPU kernel to assign the value use less.
    auto e = queue.submit([&](auto &h){
      auto x = rpn_input_size_;
      auto dev_scattered_feature_auto = dev_scattered_feature_;
      h.parallel_for(sycl::range<1>(x),[=](sycl::id<1> id) {
        dev_scattered_feature_auto[id] = 0;
      });
    });
    e.wait();
  }
  scatter_ptr_->DoScatter(host_pillar_count_[0], dev_x_coors_, dev_y_coors_, pfe_output_, dev_scattered_feature_);

  const auto t4 = std::chrono::high_resolution_clock::now();
  std::cout << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms\n";

  std::cout << "   RPN Inference";
  // 5th step is to execute the RegionProposal (RPN) network
  // Therefore, first an inference request using OpenVINO has to be created
  // Then, the required input data has to transfered from the SYCL device to be accessible by OpenVINO
  // Next, the inference can be excuted
  // At last, the results have to be copied back to the SYCL device
  // Fill input tensor with data
  auto input = rpn_infer_request_.get_compiled_model().input();
  rpn_infer_request_.set_input_tensor(input.get_index(),
                                      scattered_feature_tensor_);
  int output_index = 0;
  for (auto &output : rpn_exe_network_.outputs()) {
    rpn_infer_request_.set_tensor(output.get_any_name(), rpn_output_tensors_[output_index++]);
  }

  // Start the inference and wait for the results
  rpn_infer_request_.start_async();
  rpn_infer_request_.wait();

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
