//==============================================================
// Copyright © 2020-2021 Intel Corporation (oneAPI modifications)
//
// SPDX-License-Identifier: MIT
// =============================================================

// ------------------------------------------------------------------
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Modified from MATLAB Faster R-CNN
// (https://github.com/shaoqingren/faster_rcnn)
// ------------------------------------------------------------------

#include "pointpillars/nms.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <set>
#include <vector>
#include "devicemanager/devicemanager.hpp"

namespace pointpillars {

// Intersection over Union (IoU) calculation
// a and b are pointers to the input objects
// @return IoU value = Area of overlap / Area of union
// @details: https://en.wikipedia.org/wiki/Jaccard_index
inline float DevIoU(float const *const a, float const *const b) {
  float left = sycl::max(a[0], b[0]);
  float right = sycl::min(a[2], b[2]);
  float top = sycl::max(a[1], b[1]);
  float bottom = sycl::min(a[3], b[3]);
  float width = sycl::max((float)(right - left + 1), 0.f);
  float height = sycl::max((float)(bottom - top + 1), 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

NMS::NMS(const int num_threads, const int num_box_corners, const float nms_overlap_threshold)
    : num_threads_(num_threads), num_box_corners_(num_box_corners), nms_overlap_threshold_(nms_overlap_threshold) {}

void NMS::DoNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                size_t &out_num_to_keep) {
  // Currently the parallel implementation of NMS only works on the GPU
  // Therefore, in case of a CPU or Host device, we use the sequential implementation
  if (!devicemanager::GetCurrentDevice().is_gpu()) {
    SequentialNMS(host_filter_count, dev_sorted_box_for_nms, out_keep_inds, out_num_to_keep);
  } else {
    ParallelNMS(host_filter_count, dev_sorted_box_for_nms, out_keep_inds, out_num_to_keep);
  }
}

void NMS::SequentialNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                        size_t &out_num_to_keep) {
  std::vector<int> keep_inds_vec;           // vector holding the object indexes that should be kept
  keep_inds_vec.resize(host_filter_count);  // resize vector to maximum possible length

  // fill vector with default indexes 0, 1, 2, ...
  std::iota(keep_inds_vec.begin(), keep_inds_vec.end(), 0);

  // Convert vector to a C++ set
  std::set<int> keep_inds(keep_inds_vec.begin(), keep_inds_vec.end());

  // Filtering overlapping boxes
  for (size_t i = 0; i < host_filter_count; ++i) {
    for (size_t j = i + 1; j < host_filter_count; ++j) {
      auto iou_value =
          DevIoU(dev_sorted_box_for_nms + i * num_box_corners_, dev_sorted_box_for_nms + j * num_box_corners_);
      if (iou_value > nms_overlap_threshold_) {
        // if IoU value to too high, remove the index from the set
        keep_inds.erase(j);
      }
    }
  }

  // fill output data, with the kept indexes
  out_num_to_keep = keep_inds.size();
  int keep_counter = 0;
  for (auto ind : keep_inds) {
    out_keep_inds[keep_counter] = ind;
    keep_counter++;
  }
}

void Kernel(const int n_boxes, const float nms_overlap_thresh, const float *dev_boxes, unsigned long long *dev_mask,
            const int num_box_corners, sycl::nd_item<3> item_ct1, float *block_boxes) {
  const unsigned long row_start = item_ct1.get_group(1);
  const unsigned long col_start = item_ct1.get_group(2);

  const unsigned long block_threads = item_ct1.get_local_range().get(2);

  const unsigned long row_size = sycl::min((unsigned long)(n_boxes - row_start * block_threads), block_threads);
  const unsigned long col_size = sycl::min((unsigned long)(n_boxes - col_start * block_threads), block_threads);

  if (item_ct1.get_local_id(2) < col_size) {
    block_boxes[item_ct1.get_local_id(2) * num_box_corners + 0] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * num_box_corners + 0];
    block_boxes[item_ct1.get_local_id(2) * num_box_corners + 1] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * num_box_corners + 1];
    block_boxes[item_ct1.get_local_id(2) * num_box_corners + 2] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * num_box_corners + 2];
    block_boxes[item_ct1.get_local_id(2) * num_box_corners + 3] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * num_box_corners + 3];
  }

  if (item_ct1.get_local_id(2) < row_size) {
    const int cur_box_idx = block_threads * row_start + item_ct1.get_local_id(2);
    const float cur_box[NUM_2D_BOX_CORNERS_MACRO] = {
        dev_boxes[cur_box_idx * num_box_corners + 0], dev_boxes[cur_box_idx * num_box_corners + 1],
        dev_boxes[cur_box_idx * num_box_corners + 2], dev_boxes[cur_box_idx * num_box_corners + 3]};
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = item_ct1.get_local_id(2) + 1;
    }
    for (size_t i = start; i < col_size; i++) {
      if (DevIoU(cur_box, block_boxes + i * num_box_corners) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, block_threads);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void NMS::ParallelNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                      size_t &out_num_to_keep) {
  const unsigned long col_blocks = DIVUP(host_filter_count, num_threads_);
  sycl::range<3> blocks(DIVUP(host_filter_count, num_threads_), DIVUP(host_filter_count, num_threads_), 1);
  sycl::range<3> threads(num_threads_, 1, 1);

  unsigned long long *dev_mask;
  sycl::queue queue = devicemanager::GetCurrentQueue();
  dev_mask = sycl::malloc_device<unsigned long long>(host_filter_count * col_blocks, queue);

  queue.submit([&](auto &h) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> block_boxes_acc_ct1(
        sycl::range<1>(256), h);

    auto global_range = blocks * threads;

    auto nms_overlap_threshold_ct1 = nms_overlap_threshold_;
    auto num_box_corners_ct4 = num_box_corners_;

    h.parallel_for(sycl::nd_range<3>(sycl::range<3>(global_range.get(2), global_range.get(1), global_range.get(0)),
                                     sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
                   [=](sycl::nd_item<3> item_ct1) {
                     Kernel(host_filter_count, nms_overlap_threshold_ct1, dev_sorted_box_for_nms, dev_mask,
                            num_box_corners_ct4, item_ct1, block_boxes_acc_ct1.get_pointer());
                   });
  });
  queue.wait();

  // postprocess for nms output
  std::vector<unsigned long long> host_mask(host_filter_count * col_blocks);
  queue.memcpy(&host_mask[0], dev_mask, sizeof(unsigned long long) * host_filter_count * col_blocks).wait();
  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  for (size_t i = 0; i < host_filter_count; i++) {
    int nblock = i / num_threads_;
    int inblock = i % num_threads_;

    if (!(remv[nblock] & (1ULL << inblock))) {
      out_keep_inds[out_num_to_keep++] = i;
      unsigned long long *p = &host_mask[0] + i * col_blocks;
      for (size_t j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  // release the dev_mask, as it was only of temporary use
  sycl::free(dev_mask, devicemanager::GetCurrentQueue());
}
}  // namespace pointpillars
