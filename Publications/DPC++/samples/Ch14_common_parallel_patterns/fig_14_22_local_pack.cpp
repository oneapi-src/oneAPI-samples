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
  queue Q;

  // Set parameters to control neighborhood size
  const float CUTOFF = 3.0f;
  const uint32_t MAX_K = 150;

  // Initialize input and output on the host
  const uint32_t Nx = 8, Ny = 8, Nz = 8;
  const uint32_t N = Nx * Ny * Nz;
  float3* position = malloc_shared<float3>(N, Q);
  uint32_t* num_neighbors = malloc_shared<uint32_t>(N, Q);
  uint32_t* neighbors = malloc_shared<uint32_t>(N * MAX_K, Q);
  for (uint32_t x = 0; x < Nx; ++x) {
    for (uint32_t y = 0; y < Ny; ++y) {
      for (uint32_t z = 0; z < Nz; ++z) {
        position[z * Ny * Nx + y * Nx + x] = {x, y, z};
      }
    }
  }
  std::fill(num_neighbors, num_neighbors + N, 0);
  std::fill(neighbors, neighbors + N * MAX_K, 0);

  range<2> global(N, 8);
  range<2> local(1, 8);
  Q.parallel_for(
       nd_range<2>(global, local),
       [=](nd_item<2> it) [[intel::reqd_sub_group_size(8)]] {
         int i = it.get_global_id(0);
         sub_group sg = it.get_sub_group();
         int sglid = sg.get_local_id()[0];
         int sgrange = sg.get_max_local_range()[0];

         uint32_t k = 0;
         for (int j = sglid; j < N; j += sgrange) {

           // Compute distance between i and neighbor j
           float r = distance(position[i], position[j]);

           // Pack neighbors that require post-processing into a list
           uint32_t pack = (i != j) and (r <= CUTOFF);
           uint32_t offset = exclusive_scan_over_group(sg, pack, plus<>());
           if (pack) {
             neighbors[i * MAX_K + k + offset] = j;
           }

           // Keep track of how many neighbors have been packed so far
           k += reduce_over_group(sg, pack, plus<>());
         }
         num_neighbors[i] = reduce_over_group(sg, k, maximum<>());
       })
      .wait();

  // Check that all outputs match serial execution
  bool passed = true;
  for (int i = 0; i < N; ++i) {
    uint32_t k = 0;
    for (int j = 0; j < N; ++j) {
      float r = distance(position[i], position[j]);
      if (i != j and r <= CUTOFF) {
        if (neighbors[i * MAX_K + k] != j) {
          passed = false;
        }
        k++;
      }
    }
    if (num_neighbors[i] != k) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";
  free(neighbors, Q);
  free(num_neighbors, Q);
  free(position, Q);
  return (passed) ? 0 : 1;
}
