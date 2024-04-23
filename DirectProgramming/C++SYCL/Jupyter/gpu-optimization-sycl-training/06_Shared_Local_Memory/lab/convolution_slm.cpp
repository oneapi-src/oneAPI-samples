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
      sycl::local_accessor<int, 1> ciacc(sycl::range(256 + (M / 2) * 2), h);

      h.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
           int i = it.get_global_linear_id();
           int group = it.get_group()[0];
           int gSize = it.get_local_range()[0];
           int local_id = it.get_local_id()[0];

           ciacc[local_id + M / 2] = iacc[i];

           if (local_id == 0) {
             if (group == 0) {
               for (int j = 0; j < M / 2; j++) {
                 ciacc[j] = 0;
               }
             } else {
               for (int j = 0, k = i - M / 2; j < M / 2; j++, k++) {
                 ciacc[j] = iacc[k];
               }
             }
           }
           if (local_id == gSize - 1) {
             if (group == it.get_group_range()[0] - 1) {
               for (int j = gSize + M / 2;
                    j < gSize + M / 2 + M / 2; j++) {
                 ciacc[j] = 0;
               }
             } else {
               for (int j = gSize + M / 2, k = i + 1;
                    j < gSize + M / 2 + M / 2; j++, k++) {
                 ciacc[j] = iacc[k];
               }
             }
           }

	   sycl::group_barrier(it.get_group());

           int t = 0;
           for (int j = 0, k = local_id; j < M; j++, k++) {
             t += ciacc[k] * kacc[j];
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
