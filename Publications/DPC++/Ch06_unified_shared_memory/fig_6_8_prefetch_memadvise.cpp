// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

// Appropriate values depend on your HW
constexpr int BLOCK_SIZE = 42;
constexpr int NUM_BLOCKS = 2500;
constexpr int N = NUM_BLOCKS * BLOCK_SIZE;

int main() {
  queue Q;
  int *data = malloc_shared<int>(N, Q);
  int *read_only_data = malloc_shared<int>(BLOCK_SIZE, Q);

  // Never updated after initialization
  for (int i = 0; i < BLOCK_SIZE; i++)
    read_only_data[i] = i;

  // Mark this data as "read only" so the runtime can copy it
  // to the device instead of migrating it from the host.
  // Real values will be documented by your DPC++ backend.
  int HW_SPECIFIC_ADVICE_RO = 0;

  Q.mem_advise(read_only_data, BLOCK_SIZE, HW_SPECIFIC_ADVICE_RO);
  event e = Q.prefetch(data, BLOCK_SIZE);

  for (int b = 0; b < NUM_BLOCKS; b++) {
    Q.parallel_for(range{BLOCK_SIZE}, e,
                   [=](id<1> i) { data[b * BLOCK_SIZE + i] += data[i]; });
    if ((b + 1) < NUM_BLOCKS) {
      // Prefetch next block
      e = Q.prefetch(data + (b + 1) * BLOCK_SIZE, BLOCK_SIZE);
    }
  }
  Q.wait();

  free(data, Q);
  free(read_only_data, Q);
  return 0;
}
