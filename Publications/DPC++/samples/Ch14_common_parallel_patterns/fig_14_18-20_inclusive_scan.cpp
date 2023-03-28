// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;

int main() {
  queue q;

  const size_t N = 128;
  const size_t L = 16;
  const size_t G = N / L;

  int32_t* input = malloc_shared<int32_t>(N, q);
  int32_t* output = malloc_shared<int32_t>(N, q);
  std::iota(input, input + N, 1);
  std::fill(output, output + N, 0);

  // Create a temporary allocation that will only be used by the device
  int32_t* tmp = malloc_device<int32_t>(G, q);

  // Phase 1: Compute local scans over input blocks
  q.submit([&](handler& h) {
     auto local = local_accessor<int32_t, 1>(L, h);
     h.parallel_for(nd_range<1>(N, L), [=](nd_item<1> it) {
       int i = it.get_global_id(0);
       int li = it.get_local_id(0);

       // Copy input to local memory
       local[li] = input[i];
       group_barrier(it.get_group());

       // Perform inclusive scan in local memory
       for (int32_t d = 0; d <= log2((float)L) - 1; ++d) {
         uint32_t stride = (1 << d);
         int32_t update = (li >= stride) ? local[li - stride] : 0;
         group_barrier(it.get_group());
         local[li] += update;
         group_barrier(it.get_group());
       }

       // Write the result for each item to the output buffer
       // Write the last result from this block to the temporary buffer
       output[i] = local[li];
       if (li == it.get_local_range()[0] - 1) {
         tmp[it.get_group(0)] = local[li];
       }
     });
   }).wait();

  // Phase 2: Compute scan over partial results
  q.submit([&](handler& h) {
     auto local = local_accessor<int32_t, 1>(G, h);
     h.parallel_for(nd_range<1>(G, G), [=](nd_item<1> it) {
       int i = it.get_global_id(0);
       int li = it.get_local_id(0);

       // Copy input to local memory
       local[li] = tmp[i];
       group_barrier(it.get_group());

       // Perform inclusive scan in local memory
       for (int32_t d = 0; d <= log2((float)G) - 1; ++d) {
         uint32_t stride = (1 << d);
         int32_t update = (li >= stride) ? local[li - stride] : 0;
         group_barrier(it.get_group());
         local[li] += update;
         group_barrier(it.get_group());
       }

       // Overwrite result from each work-item in the temporary buffer
       tmp[i] = local[li];
     });
   }).wait();

  // Phase 3: Update local scans using partial results
  q.parallel_for(nd_range<1>(N, L), [=](nd_item<1> it) {
     int g = it.get_group(0);
     if (g > 0) {
       int i = it.get_global_id(0);
       output[i] += tmp[g - 1];
     }
   }).wait();

  // Check that all outputs match serial execution
  bool passed = true;
  int32_t gold = 0;
  for (int i = 0; i < N; ++i) {
    gold += input[i];
    if (output[i] != gold) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(tmp, q);
  free(output, q);
  free(input, q);
  return (passed) ? 0 : 1;
}
