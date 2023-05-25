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

  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int blockSize = 256;
  constexpr int NUM_BINS = 256;

  std::vector<unsigned long> hist(NUM_BINS, 0);

  sycl::buffer<unsigned long, 1> mbuf(input.data(), N);
  sycl::buffer<unsigned long, 1> hbuf(hist.data(), NUM_BINS);

  auto e = q.submit([&](auto &h) {
    sycl::accessor macc(mbuf, h, sycl::read_only);
    auto hacc = hbuf.get_access<sycl::access::mode::atomic>(h);
    h.parallel_for(
        sycl::nd_range(sycl::range{N / blockSize}, sycl::range{64}), [=
    ](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
          int group = it.get_group()[0];
          int gSize = it.get_local_range()[0];
          sycl::sub_group sg = it.get_sub_group();
          int sgSize = sg.get_local_range()[0];
          int sgGroup = sg.get_group_id()[0];

          unsigned int histogram[NUM_BINS];
          for (int k = 0; k < NUM_BINS; k++) {
            histogram[k] = 0;
          }
          for (int k = 0; k < blockSize; k++) {
            unsigned long x =
                sg.load(macc.get_pointer() + group * gSize * blockSize +
                        sgGroup * sgSize * blockSize + sgSize * k);
#pragma unroll
            for (int i = 0; i < 8; i++) {
              unsigned int c = x & 0x1FU;
              histogram[c] += 1;
              x = x >> 8;
            }
          }

          for (int k = 0; k < NUM_BINS; k++) {
            hacc[k].fetch_add(histogram[k]);
          }
        });
  });

  q.wait();

  size_t kernel_ns = (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>());
  std::cout << "Kernel Execution Time Average: total = " << kernel_ns * 1e-6 << " msec" << std::endl;

  return 0;
}
