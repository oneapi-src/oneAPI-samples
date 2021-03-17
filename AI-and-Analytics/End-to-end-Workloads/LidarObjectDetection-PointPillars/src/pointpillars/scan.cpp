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

#include "pointpillars/scan.hpp"
#include <CL/sycl.hpp>
#include "devicemanager/devicemanager.hpp"

namespace pointpillars {

void ScanXKernel(int *output, const int *input, int n, sycl::nd_item<3> item_ct1, uint8_t *local) {
  auto temp = (int *)local;  // allocated on invocation
  int thid = item_ct1.get_local_id(2);
  int bid = item_ct1.get_group(2);
  int bdim = item_ct1.get_local_range().get(2);
  int offset = 1;

  temp[2 * thid] = input[bid * bdim * 2 + 2 * thid];  // load input into shared memory
  temp[2 * thid + 1] = input[bid * bdim * 2 + 2 * thid + 1];

  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
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
  }                                 // clear the last element
  for (int d = 1; d < n; d *= 2) {  // traverse down tree & build scan
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

void ScanYKernel(int *output, const int *input, int n, sycl::nd_item<3> item_ct1, uint8_t *local) {
  auto temp = (int *)local;  // allocated on invocation
  int thid = item_ct1.get_local_id(2);
  int bid = item_ct1.get_group(2);
  int bdim = item_ct1.get_local_range().get(2);
  int gdim = item_ct1.get_group_range(2);
  int offset = 1;
  temp[2 * thid] = input[bid + 2 * thid * gdim];  // load input into shared memory
  temp[2 * thid + 1] = input[bid + 2 * thid * gdim + gdim];
  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
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
  }                                 // clear the last element
  for (int d = 1; d < n; d *= 2) {  // traverse down tree & build scan
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

void ScanX(int *dev_output, const int *dev_input, int w, int h, int n) {
  sycl::queue queue = devicemanager::GetCurrentQueue();
  if (!devicemanager::GetCurrentDevice().is_cpu()) {
    // For host and GPU (due to worker limitations) we use a sequential
    // implementation
    int *host_input = new int[w * h];
    int *host_output = new int[w * h];
    queue.memcpy(host_input, dev_input, w * h * sizeof(int)).wait();

    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        if (j == 0) {
          host_output[i * w + j] = host_input[i * w + j];
        } else {
          host_output[i * w + j] = host_input[i * w + j] + host_output[i * w + j - 1];
        }
      }
    }

    queue.memcpy(dev_output, host_output, w * h * sizeof(int)).wait();
  } else {
    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_acc_ct1(
          sycl::range<1>(n * sizeof(int)), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, h) * sycl::range<3>(1, 1, w / 2), sycl::range<3>(1, 1, w / 2)),
          [=](sycl::nd_item<3> item_ct1) {
            ScanXKernel(dev_output, dev_input, n, item_ct1, local_acc_ct1.get_pointer());
          });
    });
  }
  queue.wait();
}

void ScanY(int *dev_output, const int *dev_input, int w, int h, int n) {
  sycl::queue queue = devicemanager::GetCurrentQueue();
  if (!devicemanager::GetCurrentDevice().is_cpu()) {
    // For host and GPU (due to worker limitations) we use a sequential
    // implementation
    int *host_input = new int[w * h];
    int *host_output = new int[w * h];
    queue.memcpy(host_input, dev_input, w * h * sizeof(int)).wait();

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        if (j == 0) {
          host_output[i + j * w] = host_input[i + j * w];
        } else {
          host_output[i + j * w] = host_input[i + j * w] + host_output[i + (j - 1) * w];
        }
      }
    }

    queue.memcpy(dev_output, host_output, w * h * sizeof(int)).wait();
  } else {
    queue.submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_acc_ct1(
          sycl::range<1>(n * sizeof(int)), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, w) * sycl::range<3>(1, 1, h / 2), sycl::range<3>(1, 1, h / 2)),
          [=](sycl::nd_item<3> item_ct1) {
            ScanYKernel(dev_output, dev_input, n, item_ct1, local_acc_ct1.get_pointer());
          });
    });
  }
  queue.wait();
}

}  // namespace pointpillars
