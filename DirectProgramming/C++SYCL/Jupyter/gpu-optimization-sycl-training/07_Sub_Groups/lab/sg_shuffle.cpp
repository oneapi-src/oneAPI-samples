//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <iomanip>

constexpr size_t N = 16;

int main() {
  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<unsigned int> matrix(N * N);
  for (int i = 0; i < N * N; ++i) {
    matrix[i] = i;
  }

  std::cout << "Matrix: " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::setw(3) << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  {
    constexpr size_t blockSize = 16;
    sycl::buffer<unsigned int, 2> m(matrix.data(), sycl::range<2>(N, N));

    auto e = q.submit([&](auto &h) {
      sycl::accessor marr(m, h);
      sycl::local_accessor<unsigned int, 2> barr1(sycl::range<2>(blockSize, blockSize), h);
      sycl::local_accessor<unsigned int, 2> barr2(sycl::range<2>(blockSize, blockSize), h);

      h.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(N / blockSize, N),
                            sycl::range<2>(1, blockSize)),
          [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
            int gi = it.get_group(0);
            int gj = it.get_group(1);

            sycl::sub_group sg = it.get_sub_group();
            int sgId = sg.get_local_id()[0];

            unsigned int bcol[blockSize];
            int ai = blockSize * gi;
            int aj = blockSize * gj;

            for (int k = 0; k < blockSize; k++) {
              bcol[k] = sg.load(marr.get_pointer() + (ai + k) * N + aj);
            }

            unsigned int tcol[blockSize];
            for (int n = 0; n < blockSize; n++) {
              if (sgId == n) {
                for (int k = 0; k < blockSize; k++) {
                  tcol[k] = sycl::select_from_group(sg, bcol[n], k);
                }
              }
            }

            for (int k = 0; k < blockSize; k++) {
              sg.store(marr.get_pointer() + (ai + k) * N + aj, tcol[k]);
            }
          });
    });
    q.wait();

    size_t kernel_time = (e.template get_profiling_info< sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>());
    std::cout << "\nKernel Execution Time: " << kernel_time * 1e-6 << " msec\n";
  }

  std::cout << std::endl << "Transposed Matrix: " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << std::setw(3) << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
