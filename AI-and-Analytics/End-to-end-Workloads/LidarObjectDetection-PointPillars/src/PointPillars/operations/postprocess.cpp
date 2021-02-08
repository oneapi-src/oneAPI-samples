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

#include "PointPillars/operations/postprocess.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

namespace dnn {
__dpct_inline__ float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

void filter_kernel(const float *box_preds, const float *cls_preds, const float *dir_preds, const int *anchor_mask,
                   const float *dev_anchors_px, const float *dev_anchors_py, const float *dev_anchors_pz,
                   const float *dev_anchors_dx, const float *dev_anchors_dy, const float *dev_anchors_dz,
                   const float *dev_anchors_ro, float *filtered_box, float *filtered_score, float *multiclass_score,
                   int *filtered_dir, int *dev_filtered_class_id, float *box_for_nms, int *filter_count,
                   const float FLOAT_MIN, const float FLOAT_MAX, const float score_threshold,
                   const size_t NUM_BOX_CORNERS, const size_t NUM_OUTPUT_BOX_FEATURE, const size_t NUM_CLS,
                   const int index) {
  float classScoreCache[20];  // Asume maximum class size of 20 to avoid runtime
                              // allocations

  int tid = index;
  // sigmoid function
  float score = 0.f;
  int class_id = 0;
  for (size_t i = 0; i < NUM_CLS; i++) {
    classScoreCache[i] = sigmoid(cls_preds[tid * NUM_CLS + i]);

    if (classScoreCache[i] > score) {
      score = classScoreCache[i];
      class_id = i;
    }
  }

  // if there is data inside the anchor
  if (anchor_mask[tid] == 1 && score > score_threshold) {
    int counter = dpct::atomic_fetch_add(filter_count, 1);
    float za = dev_anchors_pz[tid] + dev_anchors_dz[tid] / 2;

    // decode network output
    float diagonal = sycl::sqrt(dev_anchors_dx[tid] * dev_anchors_dx[tid] + dev_anchors_dy[tid] * dev_anchors_dy[tid]);
    float box_px = box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 0] * diagonal + dev_anchors_px[tid];
    float box_py = box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 1] * diagonal + dev_anchors_py[tid];
    float box_pz = box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 2] * dev_anchors_dz[tid] + za;
    float box_dx = sycl::exp((float)(box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 3])) * dev_anchors_dx[tid];
    float box_dy = sycl::exp((float)(box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 4])) * dev_anchors_dy[tid];
    float box_dz = sycl::exp((float)(box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 5])) * dev_anchors_dz[tid];
    float box_ro = box_preds[tid * NUM_OUTPUT_BOX_FEATURE + 6] + dev_anchors_ro[tid];

    box_pz = box_pz - box_dz / 2.f;

    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 0] = box_px;
    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 1] = box_py;
    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 2] = box_pz;
    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 3] = box_dx;
    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 4] = box_dy;
    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 5] = box_dz;
    filtered_box[counter * NUM_OUTPUT_BOX_FEATURE + 6] = box_ro;
    filtered_score[counter] = score;

    for (size_t i = 0; i < NUM_CLS; i++) {
      multiclass_score[counter * NUM_CLS + i] = classScoreCache[i];
    }

    int direction_label;
    if (dir_preds[tid * 2 + 0] < dir_preds[tid * 2 + 1]) {
      direction_label = 1;
    } else {
      direction_label = 0;
    }

    filtered_dir[counter] = direction_label;
    dev_filtered_class_id[counter] = class_id;
    // convrt normal box(normal boxes: x, y, z, w, l, h, r) to box(xmin, ymin,
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

    float xmin = FLOAT_MAX;
    float ymin = FLOAT_MAX;
    float xmax = FLOAT_MIN;
    float ymax = FLOAT_MIN;
    for (size_t i = 0; i < NUM_BOX_CORNERS; i++) {
      rotated_corners[i * 2 + 0] = cos_yaw * corners[i * 2 + 0] - sin_yaw * corners[i * 2 + 1];
      rotated_corners[i * 2 + 1] = sin_yaw * corners[i * 2 + 0] + cos_yaw * corners[i * 2 + 1];

      offset_corners[i * 2 + 0] = rotated_corners[i * 2 + 0] + box_px;
      offset_corners[i * 2 + 1] = rotated_corners[i * 2 + 1] + box_py;

      xmin = sycl::fmin(xmin, offset_corners[i * 2 + 0]);
      ymin = sycl::fmin(ymin, offset_corners[i * 2 + 1]);
      xmax = sycl::fmax(xmin, offset_corners[i * 2 + 0]);
      ymax = sycl::fmax(ymax, offset_corners[i * 2 + 1]);
    }
    // box_for_nms(num_box, 4)
    box_for_nms[counter * NUM_BOX_CORNERS + 0] = xmin;
    box_for_nms[counter * NUM_BOX_CORNERS + 1] = ymin;
    box_for_nms[counter * NUM_BOX_CORNERS + 2] = xmax;
    box_for_nms[counter * NUM_BOX_CORNERS + 3] = ymax;
  }
}

void sort_boxes_by_indexes_kernel(float *filtered_box, int *filtered_dir, int *filtered_class_id,
                                  float *dev_multiclass_score, float *box_for_nms, int *indexes, int filter_count,
                                  float *sorted_filtered_boxes, int *sorted_filtered_dir, int *sorted_filtered_class_id,
                                  float *dev_sorted_multiclass_score, float *sorted_box_for_nms,
                                  const size_t NUM_BOX_CORNERS, const size_t NUM_OUTPUT_BOX_FEATURE,
                                  const size_t NUM_CLS, sycl::nd_item<3> item_ct1) {
  int tid = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  if (tid < filter_count) {
    int sort_index = indexes[tid];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 0] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 0];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 1] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 1];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 2] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 2];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 3] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 3];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 4] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 4];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 5] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 5];
    sorted_filtered_boxes[tid * NUM_OUTPUT_BOX_FEATURE + 6] = filtered_box[sort_index * NUM_OUTPUT_BOX_FEATURE + 6];

    for (size_t i = 0; i < NUM_CLS; ++i) {
      dev_sorted_multiclass_score[tid * NUM_CLS + i] = dev_multiclass_score[sort_index * NUM_CLS + i];
    }

    sorted_filtered_dir[tid] = filtered_dir[sort_index];
    sorted_filtered_class_id[tid] = filtered_class_id[sort_index];
    sorted_box_for_nms[tid * NUM_BOX_CORNERS + 0] = box_for_nms[sort_index * NUM_BOX_CORNERS + 0];
    sorted_box_for_nms[tid * NUM_BOX_CORNERS + 1] = box_for_nms[sort_index * NUM_BOX_CORNERS + 1];
    sorted_box_for_nms[tid * NUM_BOX_CORNERS + 2] = box_for_nms[sort_index * NUM_BOX_CORNERS + 2];
    sorted_box_for_nms[tid * NUM_BOX_CORNERS + 3] = box_for_nms[sort_index * NUM_BOX_CORNERS + 3];
  }
}

PostProcess::PostProcess(const float FLOAT_MIN, const float FLOAT_MAX, const size_t NUM_ANCHOR_X_INDS,
                         const size_t NUM_ANCHOR_Y_INDS, const size_t NUM_ANCHOR_R_INDS, const size_t NUM_CLS,
                         const float score_threshold, const size_t NUM_THREADS, const float nms_overlap_threshold,
                         const size_t NUM_BOX_CORNERS, const size_t NUM_OUTPUT_BOX_FEATURE)
    : FLOAT_MIN_(FLOAT_MIN),
      FLOAT_MAX_(FLOAT_MAX),
      NUM_ANCHOR_X_INDS_(NUM_ANCHOR_X_INDS),
      NUM_ANCHOR_Y_INDS_(NUM_ANCHOR_Y_INDS),
      NUM_ANCHOR_R_INDS_(NUM_ANCHOR_R_INDS),
      NUM_CLS_(NUM_CLS),
      score_threshold_(score_threshold),
      NUM_THREADS_(NUM_THREADS),
      NUM_BOX_CORNERS_(NUM_BOX_CORNERS),
      NUM_OUTPUT_BOX_FEATURE_(NUM_OUTPUT_BOX_FEATURE) {
  nms_ptr_.reset(new NMS(NUM_THREADS, NUM_BOX_CORNERS, nms_overlap_threshold));
}

void PostProcess::doPostProcess(const float *rpn_box_output, const float *rpn_cls_output, const float *rpn_dir_output,
                                int *dev_anchor_mask, const float *dev_anchors_px, const float *dev_anchors_py,
                                const float *dev_anchors_pz, const float *dev_anchors_dx, const float *dev_anchors_dy,
                                const float *dev_anchors_dz, const float *dev_anchors_ro, float *dev_multiclass_score,
                                float *dev_filtered_box, float *dev_filtered_score, int *dev_filtered_dir,
                                int *dev_filtered_class_id, float *dev_box_for_nms, int *dev_filter_count,
                                std::vector<ObjectDetection> &detections) {
  // filter objects by applying a class confidence threshold
  const unsigned int length = NUM_ANCHOR_X_INDS_ * NUM_CLS_ * NUM_ANCHOR_R_INDS_ * NUM_ANCHOR_Y_INDS_;

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto FLOAT_MIN__ct18 = FLOAT_MIN_;
    auto FLOAT_MAX__ct19 = FLOAT_MAX_;
    auto score_threshold__ct20 = score_threshold_;
    auto NUM_BOX_CORNERS__ct21 = NUM_BOX_CORNERS_;
    auto NUM_OUTPUT_BOX_FEATURE__ct22 = NUM_OUTPUT_BOX_FEATURE_;
    auto NUM_CLS__ct23 = NUM_CLS_;

    cgh.parallel_for(sycl::range<1>{length}, [=](sycl::item<1> it) {
      const int index = it[0];
      filter_kernel(rpn_box_output, rpn_cls_output, rpn_dir_output, dev_anchor_mask, dev_anchors_px, dev_anchors_py,
                    dev_anchors_pz, dev_anchors_dx, dev_anchors_dy, dev_anchors_dz, dev_anchors_ro, dev_filtered_box,
                    dev_filtered_score, dev_multiclass_score, dev_filtered_dir, dev_filtered_class_id, dev_box_for_nms,
                    dev_filter_count, FLOAT_MIN__ct18, FLOAT_MAX__ct19, score_threshold__ct20, NUM_BOX_CORNERS__ct21,
                    NUM_OUTPUT_BOX_FEATURE__ct22, NUM_CLS__ct23, index);
    });
  });

  int host_filter_count[1];
  dpct::get_default_queue().memcpy(host_filter_count, dev_filter_count, sizeof(int)).wait();
  if (host_filter_count[0] == 0) {
    return;
  }

  int *dev_indexes;
  float *dev_sorted_filtered_box, *dev_sorted_box_for_nms, *dev_sorted_multiclass_score;
  int *dev_sorted_filtered_dir, *dev_sorted_filtered_class_id;

  dev_indexes = (int *)sycl::malloc_device(host_filter_count[0] * sizeof(int), dpct::get_current_device(),
                                           dpct::get_default_context());

  dev_sorted_filtered_box = (float *)sycl::malloc_device(NUM_OUTPUT_BOX_FEATURE_ * host_filter_count[0] * sizeof(float),
                                                         dpct::get_current_device(), dpct::get_default_context());

  dev_sorted_filtered_dir = (int *)sycl::malloc_device(host_filter_count[0] * sizeof(int), dpct::get_current_device(),
                                                       dpct::get_default_context());

  dev_sorted_filtered_class_id = (int *)sycl::malloc_device(host_filter_count[0] * sizeof(int),
                                                            dpct::get_current_device(), dpct::get_default_context());

  dev_sorted_box_for_nms = (float *)sycl::malloc_device(NUM_BOX_CORNERS_ * host_filter_count[0] * sizeof(float),
                                                        dpct::get_current_device(), dpct::get_default_context());

  dev_sorted_multiclass_score = (float *)sycl::malloc_device(NUM_CLS_ * host_filter_count[0] * sizeof(float),
                                                             dpct::get_current_device(), dpct::get_default_context());

  dpct::iota(dpl::execution::make_device_policy(dpct::get_default_queue()), dev_indexes,
             dev_indexes + host_filter_count[0]);

  dpct::sort(dpl::execution::make_device_policy(dpct::get_default_queue()), dev_filtered_score,
             dev_filtered_score + size_t(host_filter_count[0]), dev_indexes, std::greater<float>());

  const int num_blocks = DIVUP(host_filter_count[0], NUM_THREADS_);

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto host_filter_count_ct6 = host_filter_count[0];
    auto NUM_BOX_CORNERS__ct12 = NUM_BOX_CORNERS_;
    auto NUM_OUTPUT_BOX_FEATURE__ct13 = NUM_OUTPUT_BOX_FEATURE_;
    auto NUM_CLS__ct14 = NUM_CLS_;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, NUM_THREADS_),
                                       sycl::range<3>(1, 1, NUM_THREADS_)),
                     [=](sycl::nd_item<3> item_ct1) {
                       sort_boxes_by_indexes_kernel(
                           dev_filtered_box, dev_filtered_dir, dev_filtered_class_id, dev_multiclass_score,
                           dev_box_for_nms, dev_indexes, host_filter_count_ct6, dev_sorted_filtered_box,
                           dev_sorted_filtered_dir, dev_sorted_filtered_class_id, dev_sorted_multiclass_score,
                           dev_sorted_box_for_nms, NUM_BOX_CORNERS__ct12, NUM_OUTPUT_BOX_FEATURE__ct13, NUM_CLS__ct14,
                           item_ct1);
                     });
  });

  int keep_inds[host_filter_count[0]];
  size_t out_num_objects = 0;
  nms_ptr_->doNMS(host_filter_count[0], dev_sorted_box_for_nms, keep_inds, out_num_objects);

  float host_filtered_box[host_filter_count[0] * NUM_OUTPUT_BOX_FEATURE_];
  float host_multiclass_score[host_filter_count[0] * NUM_CLS_];

  float host_filtered_score[host_filter_count[0]];
  int host_filtered_dir[host_filter_count[0]];
  int host_filtered_class_id[host_filter_count[0]];

  dpct::get_default_queue()
      .memcpy(host_filtered_box, dev_sorted_filtered_box,
              NUM_OUTPUT_BOX_FEATURE_ * host_filter_count[0] * sizeof(float))
      .wait();

  dpct::get_default_queue()
      .memcpy(host_multiclass_score, dev_sorted_multiclass_score, NUM_CLS_ * host_filter_count[0] * sizeof(float))
      .wait();

  dpct::get_default_queue()
      .memcpy(host_filtered_class_id, dev_sorted_filtered_class_id, host_filter_count[0] * sizeof(int))
      .wait();

  dpct::get_default_queue()
      .memcpy(host_filtered_dir, dev_sorted_filtered_dir, host_filter_count[0] * sizeof(int))
      .wait();

  dpct::get_default_queue()
      .memcpy(host_filtered_score, dev_filtered_score, host_filter_count[0] * sizeof(float))
      .wait();

  for (size_t i = 0; i < out_num_objects; i++) {
    ObjectDetection detection;
    detection.x = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 0];
    detection.y = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 1];
    detection.z = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 2];
    detection.length = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 3];
    detection.width = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 4];
    detection.height = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 5];

    detection.classId = static_cast<float>(host_filtered_class_id[keep_inds[i]]);
    detection.likelihood = host_filtered_score[keep_inds[i]];

    if (host_filtered_dir[keep_inds[i]] == 0) {
      detection.yaw = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 6] + M_PI;
    } else {
      detection.yaw = host_filtered_box[keep_inds[i] * NUM_OUTPUT_BOX_FEATURE_ + 6];
    }

    for (size_t k = 0; k < NUM_CLS_; k++) {
      detection.classProbabilities.push_back(host_multiclass_score[keep_inds[i] * NUM_CLS_ + k]);
    }

    detections.push_back(detection);
  }

  sycl::free(dev_indexes, dpct::get_default_context());
  sycl::free(dev_sorted_filtered_box, dpct::get_default_context());
  sycl::free(dev_sorted_filtered_dir, dpct::get_default_context());
  sycl::free(dev_sorted_box_for_nms, dpct::get_default_context());
}
}
