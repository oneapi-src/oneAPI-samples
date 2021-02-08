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

#include "PointPillars/operations/scan.hpp"
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace dnn {

void scan_x(int *output, const int *input, int n, sycl::nd_item<3> item_ct1, uint8_t *dpct_local) {
  auto temp = (int *)dpct_local;  // allocated on invocation
  int thid = item_ct1.get_local_id(2);
  int bid = item_ct1.get_group(2);
  int bdim = item_ct1.get_local_range().get(2);
  int offset = 1;

  temp[2 * thid] = input[bid * bdim * 2 + 2 * thid];  // load input into shared memory
  temp[2 * thid + 1] = input[bid * bdim * 2 + 2 * thid + 1];

  for (int d = n >> 1; d > 0; d >>= 1)  // build sum in place up the tree
  {
    item_ct1.barrier();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0) {
    temp[n - 1] = 0;
  }                               // clear the last element
  for (int d = 1; d < n; d *= 2)  // traverse down tree & build scan
  {
    offset >>= 1;
    item_ct1.barrier();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  item_ct1.barrier();
  output[bid * bdim * 2 + 2 * thid] = temp[2 * thid + 1];  // write results to device memory
  if (thid + 1 == bdim) {
    output[bid * bdim * 2 + 2 * thid + 1] = temp[2 * thid + 1] + input[bid * bdim * 2 + 2 * thid + 1];
  } else {
    output[bid * bdim * 2 + 2 * thid + 1] = temp[2 * thid + 2];
  }
}

void scan_y(int *output, const int *input, int n, sycl::nd_item<3> item_ct1, uint8_t *dpct_local) {
  auto temp = (int *)dpct_local;  // allocated on invocation
  int thid = item_ct1.get_local_id(2);
  int bid = item_ct1.get_group(2);
  int bdim = item_ct1.get_local_range().get(2);
  int gdim = item_ct1.get_group_range(2);
  int offset = 1;
  temp[2 * thid] = input[bid + 2 * thid * gdim];  // load input into shared memory
  temp[2 * thid + 1] = input[bid + 2 * thid * gdim + gdim];
  for (int d = n >> 1; d > 0; d >>= 1)  // build sum in place up the tree
  {
    item_ct1.barrier();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0) {
    temp[n - 1] = 0;
  }                               // clear the last element
  for (int d = 1; d < n; d *= 2)  // traverse down tree & build scan
  {
    offset >>= 1;
    item_ct1.barrier();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  item_ct1.barrier();
  output[bid + 2 * thid * gdim] = temp[2 * thid + 1];  // write results to device memory
  int second_ind = 2 * thid + 2;
  if (second_ind == bdim * 2) {
    output[bid + 2 * thid * gdim + gdim] = temp[2 * thid + 1] + input[bid + 2 * thid * gdim + gdim];
  } else {
    output[bid + 2 * thid * gdim + gdim] = temp[2 * thid + 2];
  }
}

void scanX(int *devOutput, const int *devInput, int w, int h, int n) {
  if (!dpct::get_current_device().is_cpu()) {
    // For host and GPU (due to worker limitations) we use a sequential
    // implementation
    int *hostInput = new int[w * h];
    int *hostOutput = new int[w * h];
    dpct::get_default_queue().memcpy(hostInput, devInput, w * h * sizeof(int)).wait();

    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        if (j == 0) {
          hostOutput[i * w + j] = hostInput[i * w + j];
        } else {
          hostOutput[i * w + j] = hostInput[i * w + j] + hostOutput[i * w + j - 1];
        }
      }
    }

    dpct::get_default_queue().memcpy(devOutput, hostOutput, w * h * sizeof(int)).wait();
  } else {
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(
          sycl::range<1>(n * sizeof(int)), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, h) * sycl::range<3>(1, 1, w / 2), sycl::range<3>(1, 1, w / 2)),
          [=](sycl::nd_item<3> item_ct1) {
            scan_x(devOutput, devInput, n, item_ct1, dpct_local_acc_ct1.get_pointer());
          });
    });
  }
}

void scanY(int *devOutput, const int *devInput, int w, int h, int n) {
  if (!dpct::get_current_device().is_cpu()) {
    // For host and GPU (due to worker limitations) we use a sequential
    // implementation
    int *hostInput = new int[w * h];
    int *hostOutput = new int[w * h];
    dpct::get_default_queue().memcpy(hostInput, devInput, w * h * sizeof(int)).wait();

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        if (j == 0) {
          hostOutput[i + j * w] = hostInput[i + j * w];
        } else {
          hostOutput[i + j * w] = hostInput[i + j * w] + hostOutput[i + (j - 1) * w];
        }
      }
    }

    dpct::get_default_queue().memcpy(devOutput, hostOutput, w * h * sizeof(int)).wait();
  } else {
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(
          sycl::range<1>(n * sizeof(int)), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, w) * sycl::range<3>(1, 1, h / 2), sycl::range<3>(1, 1, h / 2)),
          [=](sycl::nd_item<3> item_ct1) {
            scan_y(devOutput, devInput, n, item_ct1, dpct_local_acc_ct1.get_pointer());
          });
    });
  }
}

}  // namespace dnn
