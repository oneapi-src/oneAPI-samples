//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  constexpr size_t N = 8192 * 8192;
  constexpr size_t M = 257;

  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<int> input(N + M / 2 + M / 2);
  std::vector<int> output(N);
  std::vector<int> kernel(M);

  srand(2009);
  for (int i = M / 2; i < N + M / 2; ++i) {
    input[i] = rand();
  }

  for (int i = 0; i < M / 2; ++i) {
    input[i] = 0;
    input[i + N + M / 2] = 0;
  }

  for (int i = 0; i < M; ++i) {
    kernel[i] = rand();
  }

  {
    sycl::buffer<int> ibuf(input.data(), N + M / 2 + M / 2);
    sycl::buffer<int> obuf(output.data(), N);
    sycl::buffer<int> kbuf(kernel.data(), M);

    auto e = q.submit([&](auto &h) {
      sycl::accessor iacc(ibuf, h, sycl::read_only);
      sycl::accessor oacc(obuf, h);
      sycl::accessor kacc(kbuf, h, sycl::read_only);

      h.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
           int i = it.get_global_linear_id();
           int t = 0;

           for (int j = 0; j < M; j++) {
             t += iacc[i + j] * kacc[j];
           }

           oacc[i] = t;
         });
    });
    q.wait();

    size_t kernel_ns = (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time Average: total = " << kernel_ns * 1e-6 << " msec\n";
  }

  return 0;
}
