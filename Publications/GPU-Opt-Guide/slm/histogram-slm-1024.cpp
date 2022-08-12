//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <vector>

int main() {
  constexpr int N = 4096 * 4096;

  std::vector<unsigned long> input(N);
  srand(2009);
  for (int i = 0; i < N; ++i) {
    input[i] = (long)rand() % 1024;
    input[i] |= ((long)rand() % 1024) << 16;
    input[i] |= ((long)rand() % 1024) << 32;
    input[i] |= ((long)rand() % 1024) << 48;
  }

  sycl::queue q{sycl::gpu_selector{},
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // Snippet begin
  constexpr int NUM_BINS = 1024;
  constexpr int blockSize = 256;

  std::vector<unsigned long> hist(NUM_BINS, 0);
  sycl::buffer<unsigned long, 1> mbuf(input.data(), N);
  sycl::buffer<unsigned long, 1> hbuf(hist.data(), NUM_BINS);

  auto e = q.submit([&](auto &h) {
    sycl::accessor macc(mbuf, h, sycl::read_only);
    auto hacc = hbuf.get_access<sycl::access::mode::atomic>(h);
    sycl::accessor<unsigned int, 1, sycl::access::mode::atomic,
                   sycl::access::target::local>
        local_histogram(sycl::range(NUM_BINS), h);
    h.parallel_for(
        sycl::nd_range(sycl::range{N / blockSize}, sycl::range{64}),
        [=](sycl::nd_item<1> it) {
          int group = it.get_group()[0];
          int gSize = it.get_local_range()[0];
          sycl::ext::oneapi::sub_group sg = it.get_sub_group();
          int sgSize = sg.get_local_range()[0];
          int sgGroup = sg.get_group_id()[0];

          int factor = NUM_BINS / gSize;
          int local_id = it.get_local_id()[0];
          if ((factor <= 1) && (local_id < NUM_BINS)) {
            local_histogram[local_id].store(0);
          } else {
            for (int k = 0; k < factor; k++) {
              local_histogram[gSize * k + local_id].store(0);
            }
          }
          it.barrier(sycl::access::fence_space::local_space);

          for (int k = 0; k < blockSize; k++) {
            unsigned long x =
                sg.load(macc.get_pointer() + group * gSize * blockSize +
                        sgGroup * sgSize * blockSize + sgSize * k);
            local_histogram[x & 0x3FFU].fetch_add(1);
            local_histogram[(x >> 16) & 0x3FFU].fetch_add(1);
            local_histogram[(x >> 32) & 0x3FFU].fetch_add(1);
            local_histogram[(x >> 48) & 0x3FFU].fetch_add(1);
          }
          it.barrier(sycl::access::fence_space::local_space);

          if ((factor <= 1) && (local_id < NUM_BINS)) {
            hacc[local_id].fetch_add(local_histogram[local_id].load());
          } else {
            for (int k = 0; k < factor; k++) {
              hacc[gSize * k + local_id].fetch_add(
                  local_histogram[gSize * k + local_id].load());
            }
          }
        });
  });
  // Snippet end
  q.wait();

  size_t kernel_ns = (e.template get_profiling_info<
                          sycl::info::event_profiling::command_end>() -
                      e.template get_profiling_info<
                          sycl::info::event_profiling::command_start>());
  std::cout << "Kernel Execution Time Average: total = " << kernel_ns * 1e-6
            << " msec" << std::endl;

  return 0;
}
