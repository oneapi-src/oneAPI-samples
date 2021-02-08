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

#include "PointPillars/operations/nms.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <dpct/dpct.hpp>
#include <vector>

namespace dnn {
inline float devIoU(float const *const a, float const *const b) {
  float left = sycl::max(a[0], b[0]), right = sycl::min(a[2], b[2]);
  float top = sycl::max(a[1], b[1]), bottom = sycl::min(a[3], b[3]);
  float width = sycl::max((float)(right - left + 1), 0.f), height = sycl::max((float)(bottom - top + 1), 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

void nms_kernel(const int n_boxes, const float nms_overlap_thresh, const float *dev_boxes, unsigned long long *dev_mask,
                const int NUM_BOX_CORNERS, sycl::nd_item<3> item_ct1, float *block_boxes) {
  const unsigned long row_start = item_ct1.get_group(1);
  const unsigned long col_start = item_ct1.get_group(2);

  const unsigned long block_threads = item_ct1.get_local_range().get(2);

  const unsigned long row_size = sycl::min((unsigned long)(n_boxes - row_start * block_threads), block_threads);
  const unsigned long col_size = sycl::min((unsigned long)(n_boxes - col_start * block_threads), block_threads);

  if (item_ct1.get_local_id(2) < col_size) {
    block_boxes[item_ct1.get_local_id(2) * NUM_BOX_CORNERS + 0] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * NUM_BOX_CORNERS + 0];
    block_boxes[item_ct1.get_local_id(2) * NUM_BOX_CORNERS + 1] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * NUM_BOX_CORNERS + 1];
    block_boxes[item_ct1.get_local_id(2) * NUM_BOX_CORNERS + 2] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * NUM_BOX_CORNERS + 2];
    block_boxes[item_ct1.get_local_id(2) * NUM_BOX_CORNERS + 3] =
        dev_boxes[(block_threads * col_start + item_ct1.get_local_id(2)) * NUM_BOX_CORNERS + 3];
  }
  // item_ct1.barrier(); // @todo: re-enable if not using host device

  if (item_ct1.get_local_id(2) < row_size) {
    const int cur_box_idx = block_threads * row_start + item_ct1.get_local_id(2);
    const float cur_box[NUM_2D_BOX_CORNERS_MACRO] = {
        dev_boxes[cur_box_idx * NUM_BOX_CORNERS + 0], dev_boxes[cur_box_idx * NUM_BOX_CORNERS + 1],
        dev_boxes[cur_box_idx * NUM_BOX_CORNERS + 2], dev_boxes[cur_box_idx * NUM_BOX_CORNERS + 3]};
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = item_ct1.get_local_id(2) + 1;
    }
    for (size_t i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * NUM_BOX_CORNERS) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, block_threads);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

NMS::NMS(const int NUM_THREADS, const int NUM_BOX_CORNERS, const float nms_overlap_threshold)
    : NUM_THREADS_(NUM_THREADS), NUM_BOX_CORNERS_(NUM_BOX_CORNERS), nms_overlap_threshold_(nms_overlap_threshold) {}

void NMS::doNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                size_t &out_num_to_keep) {
  if (!dpct::get_current_device().is_gpu()) {
    sequentialNMS(host_filter_count, dev_sorted_box_for_nms, out_keep_inds, out_num_to_keep);
  } else {
    parallelNMS(host_filter_count, dev_sorted_box_for_nms, out_keep_inds, out_num_to_keep);
  }
}

void NMS::sequentialNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                        size_t &out_num_to_keep) {
  std::vector<int> keep_inds_vec;
  keep_inds_vec.resize(host_filter_count);
  std::iota(keep_inds_vec.begin(), keep_inds_vec.end(), 0);
  std::set<int> keep_inds(keep_inds_vec.begin(), keep_inds_vec.end());

  // Filtering overlapping boxes
  for (size_t i = 0; i < host_filter_count; ++i) {
    for (size_t j = i + 1; j < host_filter_count; ++j) {
      auto iou_value =
          devIoU(dev_sorted_box_for_nms + i * NUM_BOX_CORNERS_, dev_sorted_box_for_nms + j * NUM_BOX_CORNERS_);
      if (iou_value > nms_overlap_threshold_) {
        keep_inds.erase(j);
      }
    }
  }

  out_num_to_keep = keep_inds.size();
  int keep_counter = 0;
  for (auto ind : keep_inds) {
    out_keep_inds[keep_counter] = ind;
    keep_counter++;
  }
}

void NMS::parallelNMS(const size_t host_filter_count, float *dev_sorted_box_for_nms, int *out_keep_inds,
                      size_t &out_num_to_keep) {
  const unsigned long col_blocks = DIVUP(host_filter_count, NUM_THREADS_);
  sycl::range<3> blocks(DIVUP(host_filter_count, NUM_THREADS_), DIVUP(host_filter_count, NUM_THREADS_), 1);
  sycl::range<3> threads(NUM_THREADS_, 1, 1);

  unsigned long long *dev_mask;
  dev_mask = (unsigned long long *)sycl::malloc_device(host_filter_count * col_blocks * sizeof(unsigned long long),
                                                       dpct::get_current_device(), dpct::get_default_context());

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> block_boxes_acc_ct1(
        sycl::range<1>(256 /*NUM_THREADS_MACRO * NUM_2D_BOX_CORNERS_MACRO*/), cgh);

    auto dpct_global_range = blocks * threads;

    auto nms_overlap_threshold__ct1 = nms_overlap_threshold_;
    auto NUM_BOX_CORNERS__ct4 = NUM_BOX_CORNERS_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                          sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          nms_kernel(host_filter_count, nms_overlap_threshold__ct1, dev_sorted_box_for_nms, dev_mask,
                     NUM_BOX_CORNERS__ct4, item_ct1, block_boxes_acc_ct1.get_pointer());
        });
  });

  // postprocess for nms output
  std::vector<unsigned long long> host_mask(host_filter_count * col_blocks);
  dpct::get_default_queue()
      .memcpy(&host_mask[0], dev_mask, sizeof(unsigned long long) * host_filter_count * col_blocks)
      .wait();
  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  for (size_t i = 0; i < host_filter_count; i++) {
    int nblock = i / NUM_THREADS_;
    int inblock = i % NUM_THREADS_;

    if (!(remv[nblock] & (1ULL << inblock))) {
      out_keep_inds[out_num_to_keep++] = i;
      unsigned long long *p = &host_mask[0] + i * col_blocks;
      for (size_t j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  sycl::free(dev_mask, dpct::get_default_context());
}
}
