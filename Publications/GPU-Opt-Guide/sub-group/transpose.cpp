//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iomanip>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

constexpr size_t N = 16;
typedef unsigned int uint;

int main() {
  sycl::queue q{sycl::gpu_selector{},
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  std::vector<uint> matrix(N * N);
  for (uint i = 0; i < N * N; ++i) {
    matrix[i] = i;
  }

  std::cout << "Matrix: " << std::endl;
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      std::cout << std::setw(3) << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  {

    // Snippet begin
    constexpr size_t blockSize = 16;
    sycl::buffer<uint, 2> m(matrix.data(), sycl::range<2>(N, N));

    auto e = q.submit([&](auto &h) {
      sycl::accessor marr(m, h);
      sycl::accessor<uint, 2, sycl::access::mode::read_write,
                     sycl::access::target::local>
          barr1(sycl::range<2>(blockSize, blockSize), h);
      sycl::accessor<uint, 2, sycl::access::mode::read_write,
                     sycl::access::target::local>
          barr2(sycl::range<2>(blockSize, blockSize), h);

      h.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(N / blockSize, N),
                            sycl::range<2>(1, blockSize)),
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
            int gi = it.get_group(0);
            int gj = it.get_group(1);

            sycl::ext::oneapi::sub_group sg = it.get_sub_group();
            uint sgId = sg.get_local_id()[0];

            uint bcol[blockSize];
            int ai = blockSize * gi;
            int aj = blockSize * gj;

            for (uint k = 0; k < blockSize; k++) {
              bcol[k] = sg.load(marr.get_pointer() + (ai + k) * N + aj);
            }

            uint tcol[blockSize];
            for (uint n = 0; n < blockSize; n++) {
              if (sgId == n) {
                for (uint k = 0; k < blockSize; k++) {
                  tcol[k] = sg.shuffle(bcol[n], k);
                }
              }
            }

            for (uint k = 0; k < blockSize; k++) {
              sg.store(marr.get_pointer() + (ai + k) * N + aj, tcol[k]);
            }
          });
    });
    // Snippet end
    q.wait();

    size_t kernel_time = (e.template get_profiling_info<
                              sycl::info::event_profiling::command_end>() -
                          e.template get_profiling_info<
                              sycl::info::event_profiling::command_start>());
    std::cout << std::endl
              << "Kernel Execution Time: " << kernel_time * 1e-6 << " msec"
              << std::endl;
  }

  std::cout << std::endl << "Transposed Matrix: " << std::endl;
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      std::cout << std::setw(3) << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
