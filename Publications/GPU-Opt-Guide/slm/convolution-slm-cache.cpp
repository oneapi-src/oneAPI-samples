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
  constexpr size_t N = 8192 * 8192;
  constexpr size_t M = 257;

  std::vector<int> input(N);
  std::vector<int> output(N);
  std::vector<int> kernel(M);

  srand(2009);
  for (size_t i = 0; i < N; ++i) {
    input[i] = rand();
  }

  for (size_t i = 0; i < M; ++i) {
    kernel[i] = rand();
  }

  sycl::queue q{sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  {
    // Snippet begin
    sycl::buffer<int> ibuf(input.data(), N);
    sycl::buffer<int> obuf(output.data(), N);
    sycl::buffer<int> kbuf(kernel.data(), M);

    auto e = q.submit([&](auto &h) {
      sycl::accessor iacc(ibuf, h, sycl::read_only);
      sycl::accessor oacc(obuf, h);
      sycl::accessor kacc(kbuf, h, sycl::read_only);
      sycl::local_accessor<int, 1> ciacc(sycl::range(256 + (M / 2) * 2), h);

      h.parallel_for(
          sycl::nd_range(sycl::range{N}, sycl::range{256}),
          [=](sycl::nd_item<1> it) {
            int i = it.get_global_linear_id();
            int group = it.get_group()[0];
            int gSize = it.get_local_range()[0];
            int local_id = it.get_local_id()[0];
            int _M = static_cast<int>(M);

            ciacc[local_id + M / 2] = iacc[i];

            if (local_id == 0) {
              if (group == 0) {
                for (int j = 0; j < _M / 2; ++j) {
                  ciacc[j] = 0;
                }
              } else {
                for (int j = 0, k = i - _M / 2; j < _M / 2; ++j, ++k) {
                  ciacc[j] = iacc[k];
                }
              }
            }
            if (local_id == gSize - 1) {
              if (group == static_cast<int>(it.get_group_range()[0]) - 1) {
                for (int j = gSize + _M / 2; j < gSize + _M / 2 + _M / 2; ++j) {
                  ciacc[j] = 0;
                }
              } else {
                for (int j = gSize + _M / 2, k = i + 1;
                     j < gSize + _M / 2 + _M / 2; ++j, ++k) {
                  ciacc[j] = iacc[k];
                }
              }
            }

            it.barrier(sycl::access::fence_space::local_space);

            int t = 0;
            for (int j = 0, k = local_id; j < _M; ++j, ++k) {
              t += ciacc[k] * kacc[j];
            }

            oacc[i] = t;
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
  }

  return 0;
}
