//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <sycl/sycl.hpp>

enum LSC_LDCC {
  LSC_LDCC_DEFAULT,
  LSC_LDCC_L1UC_L3UC, // 1 // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C,  // 2 // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC,  // 3 // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C,   // 4 // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC,  // 5 // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C,   // 6 // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C, // 7 // Override to L1 invalidate-after-read, and L3
                      // cached
};

extern "C" {
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uint(const __attribute__((opencl_global))
                                      uint32_t *base,
                                      int immElemOff, enum LSC_LDCC cacheOpt);
}

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
          int *indexNext = indexCurrent + SIMD_SIZE;

          float dx = 0.0f;
#pragma unroll(0)
          for (int j = 0; j < ITEMS_PER_LANE; ++j) {
        // Prefetch next index to cache
#if __SYCL_DEVICE_ONLY__
            if (j < ITEMS_PER_LANE - 1)
              __builtin_IB_lsc_prefetch_global_uint(
                  (const __attribute__((opencl_global)) uint32_t *)indexNext, 0,
                  LSC_LDCC_L1C_L3C);
#endif
            // Load index, might be cached
            int index = *indexCurrent;
            // Latency might be reduced if index was cached
            float v = values[index];
            for (int k = 0; k < 64; ++k)
              dx += sycl::sqrt(v + k);

            indexCurrent = indexNext;
            indexNext += SIMD_SIZE;
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
