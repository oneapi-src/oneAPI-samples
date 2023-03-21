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

      h.parallel_for(sycl::nd_range(sycl::range{N}, sycl::range{256}),
                     [=](sycl::nd_item<1> it) {
                       int i = it.get_global_linear_id();
                       int t = 0;
                       int startj = sycl::max<int>(M / 2 - i, 0);
                       int endj = sycl::min<int>(M / 2 + N - i, M);
                       int startk = sycl::max<int>(i - M / 2, 0);
                       for (int j = startj, k = startk; j < endj; j++, k++) {
                         t += iacc[k] * kacc[j];
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
