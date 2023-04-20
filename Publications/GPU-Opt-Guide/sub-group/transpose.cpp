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
  sycl::queue q{sycl::gpu_selector_v,
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
    constexpr size_t BLOCK_SIZE = 16;
    sycl::buffer<uint, 2> m(matrix.data(), sycl::range<2>(N, N));

    auto e = q.submit([&](auto &h) {
      sycl::accessor marr(m, h);
      sycl::local_accessor<uint, 2> barr1(
          sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
      sycl::local_accessor<uint, 2> barr2(
          sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);

      h.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(N / BLOCK_SIZE, N),
                            sycl::range<2>(1, BLOCK_SIZE)),
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
            int gi = it.get_group(0);
            int gj = it.get_group(1);

            auto sg = it.get_sub_group();
            uint sgId = sg.get_local_id()[0];

            uint bcol[BLOCK_SIZE];
            int ai = BLOCK_SIZE * gi;
            int aj = BLOCK_SIZE * gj;

            for (uint k = 0; k < BLOCK_SIZE; k++) {
              bcol[k] = sg.load(marr.get_pointer() + (ai + k) * N + aj);
            }

            uint tcol[BLOCK_SIZE];
            for (uint n = 0; n < BLOCK_SIZE; n++) {
              if (sgId == n) {
                for (uint k = 0; k < BLOCK_SIZE; k++) {
                  tcol[k] = sg.shuffle(bcol[n], k);
                }
              }
            }

            for (uint k = 0; k < BLOCK_SIZE; k++) {
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
