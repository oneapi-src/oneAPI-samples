//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  constexpr int N = 4096 * 4096;

  std::vector<unsigned long> input(N);
  srand(2009);
  for (int i = 0; i < N; ++i) {
    input[i] = (long)rand() % 256;
    input[i] |= ((long)rand() % 256) << 8;
    input[i] |= ((long)rand() % 256) << 16;
    input[i] |= ((long)rand() % 256) << 24;
    input[i] |= ((long)rand() % 256) << 32;
    input[i] |= ((long)rand() % 256) << 40;
    input[i] |= ((long)rand() % 256) << 48;
    input[i] |= ((long)rand() % 256) << 56;
  }

  sycl::queue q{sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // Snippet begin
  constexpr int NUM_BINS = 256;
  constexpr int blockSize = 256;

  std::vector<unsigned long> hist(NUM_BINS, 0);
  sycl::buffer<unsigned long, 1> mbuf(input.data(), N);
  sycl::buffer<unsigned long, 1> hbuf(hist.data(), NUM_BINS);

  auto e = q.submit([&](auto &h) {
    sycl::accessor macc(mbuf, h, sycl::read_only);
    sycl::accessor hacc(hbuf, h, sycl::read_write);
    sycl::local_accessor<unsigned int> local_histogram(sycl::range(NUM_BINS),
                                                       h);
    h.parallel_for(
        sycl::nd_range(sycl::range{N / blockSize}, sycl::range{64}),
        [=](sycl::nd_item<1> it) {
          int group = it.get_group()[0];
          int gSize = it.get_local_range()[0];
          sycl::sub_group sg = it.get_sub_group();
          int sgSize = sg.get_local_range()[0];
          int sgGroup = sg.get_group_id()[0];

          int factor = NUM_BINS / gSize;
          int local_id = it.get_local_id()[0];
          if ((factor <= 1) && (local_id < NUM_BINS)) {
            sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::local_space>
                local_bin(local_histogram[local_id]);
            local_bin.store(0);
          } else {
            for (int k = 0; k < factor; k++) {
              sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  local_bin(local_histogram[gSize * k + local_id]);
              local_bin.store(0);
            }
          }
          sycl::group_barrier(it.get_group());

          for (int k = 0; k < blockSize; k++) {
            unsigned long x =
                sg.load(macc.get_pointer() + group * gSize * blockSize +
                        sgGroup * sgSize * blockSize + sgSize * k);
#pragma unroll
            for (std::uint8_t shift : {0, 8, 16, 24, 32, 40, 48, 56}) {
              constexpr unsigned long mask = 0xFFU;
              sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  local_bin(local_histogram[(x >> shift) & mask]);
              local_bin += 1;
            }
          }
          sycl::group_barrier(it.get_group());

          if ((factor <= 1) && (local_id < NUM_BINS)) {
            sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::local_space>
                local_bin(local_histogram[local_id]);
            sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                global_bin(hacc[local_id]);
            global_bin += local_bin.load();
          } else {
            for (int k = 0; k < factor; k++) {
              sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  local_bin(local_histogram[gSize * k + local_id]);
              sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  global_bin(hacc[gSize * k + local_id]);
              global_bin += local_bin.load();
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
