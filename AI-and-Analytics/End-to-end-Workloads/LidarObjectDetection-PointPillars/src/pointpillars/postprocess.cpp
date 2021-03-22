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

#include "pointpillars/postprocess.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include "devicemanager/devicemanager.hpp"

namespace pointpillars {

inline float Sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// FilterKernel decodes the RegionProposalNetwork output and filters the generated detections by threshold
// it also generates a box representation that will later be used during nms
void FilterKernel(const float *box_preds, const float *cls_preds, const float *dir_preds, const int *anchor_mask,
                  const float *dev_anchors_px, const float *dev_anchors_py, const float *dev_anchors_pz,
                  const float *dev_anchors_dx, const float *dev_anchors_dy, const float *dev_anchors_dz,
                  const float *dev_anchors_ro, float *filtered_box, float *filtered_score, float *multiclass_score,
                  int *filtered_dir, int *dev_filtered_class_id, float *box_for_nms, int *filter_count,
                  const float float_min, const float float_max, const float score_threshold,
                  const size_t num_box_corners, const size_t num_output_box_feature, const size_t num_cls,
                  const int index) {
  float class_score_cache[20];  // Asume maximum class size of 20 to avoid runtime allocations

  int tid = index;
  // Decode the class probabilities using the Sigmoid function
  float score = 0.f;
  int class_id = 0;
  for (size_t i = 0; i < num_cls; i++) {
    class_score_cache[i] = Sigmoid(cls_preds[tid * num_cls + i]);

    if (class_score_cache[i] > score) {
      score = class_score_cache[i];
      class_id = i;
    }
  }

  // if there is data inside the anchor
  if (anchor_mask[tid] == 1 && score > score_threshold) {
    int counter = AtomicFetchAdd(filter_count, 1);
    float za = dev_anchors_pz[tid] + dev_anchors_dz[tid] / 2;

    // decode RPN output, the formulas are used according to the encoding used in the paper
    float diagonal = sycl::sqrt(dev_anchors_dx[tid] * dev_anchors_dx[tid] + dev_anchors_dy[tid] * dev_anchors_dy[tid]);
    float box_px = box_preds[tid * num_output_box_feature + 0] * diagonal + dev_anchors_px[tid];
    float box_py = box_preds[tid * num_output_box_feature + 1] * diagonal + dev_anchors_py[tid];
    float box_pz = box_preds[tid * num_output_box_feature + 2] * dev_anchors_dz[tid] + za;
    float box_dx = sycl::exp((float)(box_preds[tid * num_output_box_feature + 3])) * dev_anchors_dx[tid];
    float box_dy = sycl::exp((float)(box_preds[tid * num_output_box_feature + 4])) * dev_anchors_dy[tid];
    float box_dz = sycl::exp((float)(box_preds[tid * num_output_box_feature + 5])) * dev_anchors_dz[tid];
    float box_ro = box_preds[tid * num_output_box_feature + 6] + dev_anchors_ro[tid];

    box_pz = box_pz - box_dz / 2.f;

    // Store the detection x,y,z,l,w,h,theta coordinates
    filtered_box[counter * num_output_box_feature + 0] = box_px;
    filtered_box[counter * num_output_box_feature + 1] = box_py;
    filtered_box[counter * num_output_box_feature + 2] = box_pz;
    filtered_box[counter * num_output_box_feature + 3] = box_dx;
    filtered_box[counter * num_output_box_feature + 4] = box_dy;
    filtered_box[counter * num_output_box_feature + 5] = box_dz;
    filtered_box[counter * num_output_box_feature + 6] = box_ro;
    filtered_score[counter] = score;

    // Copy the class scores
    for (size_t i = 0; i < num_cls; i++) {
      multiclass_score[counter * num_cls + i] = class_score_cache[i];
    }

    // Decode the direction class specified in SecondNet: Sparsely Embedded Convolutional Detection
    int direction_label;
    if (dir_preds[tid * 2 + 0] < dir_preds[tid * 2 + 1]) {
      direction_label = 1;
    } else {
      direction_label = 0;
    }

    filtered_dir[counter] = direction_label;
    dev_filtered_class_id[counter] = class_id;
    // convert normal box(normal boxes: x, y, z, w, l, h, r) to box(xmin, ymin,
    // xmax, ymax) for nms calculation
    // First: dx, dy -> box(x0y0, x0y1, x1y0, x1y1)
    float corners[NUM_3D_BOX_CORNERS_MACRO] = {float(-0.5f * box_dx), float(-0.5f * box_dy), float(-0.5f * box_dx),
                                               float(0.5f * box_dy),  float(0.5f * box_dx),  float(0.5f * box_dy),
                                               float(0.5f * box_dx),  float(-0.5f * box_dy)};

    // Second: Rotate, Offset and convert to point(xmin. ymin, xmax, ymax)
    float rotated_corners[NUM_3D_BOX_CORNERS_MACRO];
    float offset_corners[NUM_3D_BOX_CORNERS_MACRO];
    float sin_yaw = sycl::sin(box_ro);
    float cos_yaw = sycl::cos(box_ro);

    float xmin = float_max;
    float ymin = float_max;
    float xmax = float_min;
    float ymax = float_min;
    for (size_t i = 0; i < num_box_corners; i++) {
      rotated_corners[i * 2 + 0] = cos_yaw * corners[i * 2 + 0] - sin_yaw * corners[i * 2 + 1];
      rotated_corners[i * 2 + 1] = sin_yaw * corners[i * 2 + 0] + cos_yaw * corners[i * 2 + 1];

      offset_corners[i * 2 + 0] = rotated_corners[i * 2 + 0] + box_px;
      offset_corners[i * 2 + 1] = rotated_corners[i * 2 + 1] + box_py;

      xmin = sycl::fmin(xmin, offset_corners[i * 2 + 0]);
      ymin = sycl::fmin(ymin, offset_corners[i * 2 + 1]);
      xmax = sycl::fmax(xmin, offset_corners[i * 2 + 0]);
      ymax = sycl::fmax(ymax, offset_corners[i * 2 + 1]);
    }
    // Store the resulting box, box_for_nms(num_box, 4)
    box_for_nms[counter * num_box_corners + 0] = xmin;
    box_for_nms[counter * num_box_corners + 1] = ymin;
    box_for_nms[counter * num_box_corners + 2] = xmax;
    box_for_nms[counter * num_box_corners + 3] = ymax;
  }
}

// This Kernel uses a list of indices to sort the given input arrays.
void SortBoxesByIndexKernel(float *filtered_box, int *filtered_dir, int *filtered_class_id, float *dev_multiclass_score,
                            float *box_for_nms, int *indexes, int filter_count, float *sorted_filtered_boxes,
                            int *sorted_filtered_dir, int *sorted_filtered_class_id, float *dev_sorted_multiclass_score,
                            float *sorted_box_for_nms, const size_t num_box_corners,
                            const size_t num_output_box_feature, const size_t num_cls, sycl::nd_item<3> item_ct1) {
  int tid = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  if (tid < filter_count) {
    int sort_index = indexes[tid];
    sorted_filtered_boxes[tid * num_output_box_feature + 0] = filtered_box[sort_index * num_output_box_feature + 0];
    sorted_filtered_boxes[tid * num_output_box_feature + 1] = filtered_box[sort_index * num_output_box_feature + 1];
    sorted_filtered_boxes[tid * num_output_box_feature + 2] = filtered_box[sort_index * num_output_box_feature + 2];
    sorted_filtered_boxes[tid * num_output_box_feature + 3] = filtered_box[sort_index * num_output_box_feature + 3];
    sorted_filtered_boxes[tid * num_output_box_feature + 4] = filtered_box[sort_index * num_output_box_feature + 4];
    sorted_filtered_boxes[tid * num_output_box_feature + 5] = filtered_box[sort_index * num_output_box_feature + 5];
    sorted_filtered_boxes[tid * num_output_box_feature + 6] = filtered_box[sort_index * num_output_box_feature + 6];

    for (size_t i = 0; i < num_cls; ++i) {
      dev_sorted_multiclass_score[tid * num_cls + i] = dev_multiclass_score[sort_index * num_cls + i];
    }

    sorted_filtered_dir[tid] = filtered_dir[sort_index];
    sorted_filtered_class_id[tid] = filtered_class_id[sort_index];
    sorted_box_for_nms[tid * num_box_corners + 0] = box_for_nms[sort_index * num_box_corners + 0];
    sorted_box_for_nms[tid * num_box_corners + 1] = box_for_nms[sort_index * num_box_corners + 1];
    sorted_box_for_nms[tid * num_box_corners + 2] = box_for_nms[sort_index * num_box_corners + 2];
    sorted_box_for_nms[tid * num_box_corners + 3] = box_for_nms[sort_index * num_box_corners + 3];
  }
}

PostProcess::PostProcess(const float float_min, const float float_max, const size_t num_anchor_x_inds,
                         const size_t num_anchor_y_inds, const size_t num_anchor_r_inds, const size_t num_cls,
                         const float score_threshold, const size_t num_threads, const float nms_overlap_threshold,
                         const size_t num_box_corners, const size_t num_output_box_feature)
    : float_min_(float_min),
      float_max_(float_max),
      num_anchor_x_inds_(num_anchor_x_inds),
      num_anchor_y_inds_(num_anchor_y_inds),
      num_anchor_r_inds_(num_anchor_r_inds),
      num_cls_(num_cls),
      score_threshold_(score_threshold),
      num_threads_(num_threads),
      num_box_corners_(num_box_corners),
      num_output_box_feature_(num_output_box_feature) {
  nms_ptr_ = std::make_unique<NMS>(num_threads, num_box_corners, nms_overlap_threshold);
}

void PostProcess::DoPostProcess(const float *rpn_box_output, const float *rpn_cls_output, const float *rpn_dir_output,
                                int *dev_anchor_mask, const float *dev_anchors_px, const float *dev_anchors_py,
                                const float *dev_anchors_pz, const float *dev_anchors_dx, const float *dev_anchors_dy,
                                const float *dev_anchors_dz, const float *dev_anchors_ro, float *dev_multiclass_score,
                                float *dev_filtered_box, float *dev_filtered_score, int *dev_filtered_dir,
                                int *dev_filtered_class_id, float *dev_box_for_nms, int *dev_filter_count,
                                std::vector<ObjectDetection> &detections) {
  // filter objects by applying a class confidence threshold

  // Calculate number of boxes in the feature map
  const unsigned int length = num_anchor_x_inds_ * num_cls_ * num_anchor_r_inds_ * num_anchor_y_inds_;

  // Decode the output of the RegionProposalNetwork and store all the boxes with score above the threshold
  sycl::queue queue = devicemanager::GetCurrentQueue();
  queue.submit([&](auto &h) {
    auto float_min_ct18 = float_min_;
    auto float_max_ct19 = float_max_;
    auto score_threshold_ct20 = score_threshold_;
    auto num_box_corners_ct21 = num_box_corners_;
    auto num_output_box_feature_ct22 = num_output_box_feature_;
    auto num_cls_ct23 = num_cls_;

    h.parallel_for(sycl::range<1>{length}, [=](sycl::item<1> it) {
      const int index = it[0];
      FilterKernel(rpn_box_output, rpn_cls_output, rpn_dir_output, dev_anchor_mask, dev_anchors_px, dev_anchors_py,
                   dev_anchors_pz, dev_anchors_dx, dev_anchors_dy, dev_anchors_dz, dev_anchors_ro, dev_filtered_box,
                   dev_filtered_score, dev_multiclass_score, dev_filtered_dir, dev_filtered_class_id, dev_box_for_nms,
                   dev_filter_count, float_min_ct18, float_max_ct19, score_threshold_ct20, num_box_corners_ct21,
                   num_output_box_feature_ct22, num_cls_ct23, index);
    });
  });
  queue.wait();

  int host_filter_count[1];
  queue.memcpy(host_filter_count, dev_filter_count, sizeof(int)).wait();
  if (host_filter_count[0] == 0) {
    return;
  }

  // Create variables to hold the sorted box arrays
  int *dev_indexes;
  float *dev_sorted_filtered_box, *dev_sorted_box_for_nms, *dev_sorted_multiclass_score;
  int *dev_sorted_filtered_dir, *dev_sorted_filtered_class_id;

  dev_indexes = sycl::malloc_device<int>(host_filter_count[0], queue);
  dev_sorted_filtered_box = sycl::malloc_device<float>(num_output_box_feature_ * host_filter_count[0], queue);
  dev_sorted_filtered_dir = sycl::malloc_device<int>(host_filter_count[0], queue);
  dev_sorted_filtered_class_id = sycl::malloc_device<int>(host_filter_count[0], queue);
  dev_sorted_box_for_nms = sycl::malloc_device<float>(num_box_corners_ * host_filter_count[0], queue);
  dev_sorted_multiclass_score = sycl::malloc_device<float>(num_cls_ * host_filter_count[0], queue);

  // Generate an array to hold the box indexes
  sycl::range<1> num_items{static_cast<std::size_t>(host_filter_count[0])};
  auto e = queue.parallel_for(num_items, [=](auto i) { dev_indexes[i] = i; });
  e.wait();

  // Sort the box indexes according to the boxes score
  auto first = oneapi::dpl::make_zip_iterator(dev_filtered_score, dev_indexes);
  auto last = first + std::distance(dev_filtered_score, dev_filtered_score + size_t(host_filter_count[0]));
  std::sort(oneapi::dpl::execution::make_device_policy(queue), first, last,
            [](auto lhs, auto rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });

  const int num_blocks = DIVUP(host_filter_count[0], num_threads_);

  // Use the sorted indexes to sort the boxes and all other decoded information from the RPN
  queue.submit([&](auto &h) {
    auto host_filter_count_ct6 = host_filter_count[0];
    auto num_box_corners_ct12 = num_box_corners_;
    auto num_output_box_feature_ct13 = num_output_box_feature_;
    auto num_cls_ct14 = num_cls_;

    h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, num_threads_),
                                     sycl::range<3>(1, 1, num_threads_)),
                   [=](sycl::nd_item<3> item_ct1) {
                     SortBoxesByIndexKernel(dev_filtered_box, dev_filtered_dir, dev_filtered_class_id,
                                            dev_multiclass_score, dev_box_for_nms, dev_indexes, host_filter_count_ct6,
                                            dev_sorted_filtered_box, dev_sorted_filtered_dir,
                                            dev_sorted_filtered_class_id, dev_sorted_multiclass_score,
                                            dev_sorted_box_for_nms, num_box_corners_ct12, num_output_box_feature_ct13,
                                            num_cls_ct14, item_ct1);
                   });
  });
  queue.wait();

  // Apply NMS to the sorted boxes
  int keep_inds[host_filter_count[0]];
  size_t out_num_objects = 0;
  nms_ptr_->DoNMS(host_filter_count[0], dev_sorted_box_for_nms, keep_inds, out_num_objects);

  // Create arrays to hold the detections in host memory
  float host_filtered_box[host_filter_count[0] * num_output_box_feature_];
  float host_multiclass_score[host_filter_count[0] * num_cls_];

  float host_filtered_score[host_filter_count[0]];
  int host_filtered_dir[host_filter_count[0]];
  int host_filtered_class_id[host_filter_count[0]];

  // Copy memory to host
  queue.memcpy(host_filtered_box, dev_sorted_filtered_box,
               num_output_box_feature_ * host_filter_count[0] * sizeof(float));
  queue.memcpy(host_multiclass_score, dev_sorted_multiclass_score, num_cls_ * host_filter_count[0] * sizeof(float));
  queue.memcpy(host_filtered_class_id, dev_sorted_filtered_class_id, host_filter_count[0] * sizeof(int));
  queue.memcpy(host_filtered_dir, dev_sorted_filtered_dir, host_filter_count[0] * sizeof(int));
  queue.memcpy(host_filtered_score, dev_filtered_score, host_filter_count[0] * sizeof(float));
  queue.wait();

  // Convert the NMS filtered boxes defined by keep_inds to an array of ObjectDetection
  for (size_t i = 0; i < out_num_objects; i++) {
    ObjectDetection detection;
    detection.x = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 0];
    detection.y = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 1];
    detection.z = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 2];
    detection.length = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 3];
    detection.width = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 4];
    detection.height = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 5];

    detection.class_id = static_cast<float>(host_filtered_class_id[keep_inds[i]]);
    detection.likelihood = host_filtered_score[keep_inds[i]];

    // Apply the direction label found by the direction classifier
    if (host_filtered_dir[keep_inds[i]] == 0) {
      detection.yaw = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 6] + M_PI;
    } else {
      detection.yaw = host_filtered_box[keep_inds[i] * num_output_box_feature_ + 6];
    }

    for (size_t k = 0; k < num_cls_; k++) {
      detection.class_probabilities.push_back(host_multiclass_score[keep_inds[i] * num_cls_ + k]);
    }

    detections.push_back(detection);
  }

  sycl::free(dev_indexes, queue);
  sycl::free(dev_sorted_filtered_box, queue);
  sycl::free(dev_sorted_filtered_dir, queue);
  sycl::free(dev_sorted_box_for_nms, queue);
}
}  // namespace pointpillars
