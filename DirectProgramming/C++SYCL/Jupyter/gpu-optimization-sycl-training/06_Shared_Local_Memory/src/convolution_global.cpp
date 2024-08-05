//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  constexpr size_t N = 8192 * 8192;
  constexpr size_t M = 257;

  std::vector<int> input(N);
  std::vector<int> output(N);
  std::vector<int> kernel(M);

  srand(2009);
  for (int i = 0; i < N; ++i) {
    input[i] = rand();
  }

  for (int i = 0; i < M; ++i) {
    kernel[i] = rand();
  }

  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  {
    sycl::buffer<int> ibuf(input.data(), N);
    sycl::buffer<int> obuf(output.data(), N);
    sycl::buffer<int> kbuf(kernel.data(), M);

    auto e = q.submit([&](auto &h) {
      sycl::accessor iacc(ibuf, h, sycl::read_only);
      sycl::accessor oacc(obuf, h);
      sycl::accessor kacc(kbuf, h, sycl::read_only);

      h.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
           int i = it.get_global_linear_id();
           int group = it.get_group()[0];
           int gSize = it.get_local_range()[0];

           int t = 0;

           if ((group == 0) || (group == N / gSize - 1)) {
             if (i < M / 2) {
               for (int j = M / 2 - i, k = 0; j < M; j++, k++) {
                 t += iacc[k] * kacc[j];
               }
             } else {
               if (i + M / 2 >= N) {
                 for (int j = 0, k = i - M / 2; j < M / 2 + N - i;
                      j++, k++) {
                   t += iacc[k] * kacc[j];
                 }
               } else {
                 for (int j = 0, k = i - M / 2; j < M; j++, k++) {
                   t += iacc[k] * kacc[j];
                 }
               }
             }
           } else {
             for (int j = 0, k = i - M / 2; j < M; j++, k++) {
               t += iacc[k] * kacc[j];
             }
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
