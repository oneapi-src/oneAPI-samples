//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q{sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{}};

  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  srand(time(NULL));

  constexpr int SIMD_SIZE = 32;
  constexpr int SUBGROUPS = 16;
  constexpr int GLOBAL_WORK_SIZE = SIMD_SIZE * SUBGROUPS;

  constexpr int ITEMS_PER_LANE = 2048;
  constexpr int ITEMS_PER_SUBGROUP = ITEMS_PER_LANE * SIMD_SIZE;
  constexpr int DATA_SIZE = ITEMS_PER_SUBGROUP * SUBGROUPS;

  float *result = sycl::malloc_shared<float>(GLOBAL_WORK_SIZE, q);
  float *values = sycl::malloc_shared<float>(DATA_SIZE, q);
  int *indices = sycl::malloc_shared<int>(DATA_SIZE, q);
  for (auto i = 0; i < DATA_SIZE; ++i) {
    values[i] = static_cast<float>(i);
    // Fill indices in random order.
    indices[i] = rand() % DATA_SIZE;
  }

  // Snippet begin
  auto e = q.submit([&](auto &h) {
    h.parallel_for(
        sycl::nd_range(sycl::range{GLOBAL_WORK_SIZE},
                       sycl::range{GLOBAL_WORK_SIZE}),
        [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SIMD_SIZE)]] {
          const int i = it.get_global_linear_id();
          const int lane = it.get_sub_group().get_local_id()[0];
          const int subgroup = it.get_sub_group().get_group_id()[0];

          // Index starting position
          int *indexCurrent = indices + lane + subgroup * ITEMS_PER_SUBGROUP;

          float dx = 0.0f;
#pragma unroll(0)
          for (int j = 0; j < ITEMS_PER_LANE; ++j) {
            // Load index for indirect addressing
            int index = *indexCurrent;
            // Waits for load to finish, high latency
            float v = values[index];
            for (int k = 0; k < 64; ++k)
              dx += sycl::sqrt(v + k);

            indexCurrent += SIMD_SIZE;
          }

          result[i] = dx;
        });
  });
  // Snippet end
  q.wait();
  std::cout << "Kernel time = "
            << (e.template get_profiling_info<
                    sycl::info::event_profiling::command_end>() -
                e.template get_profiling_info<
                    sycl::info::event_profiling::command_start>())
            << " ns" << std::endl;
  return 0;
}
