//==============================================================
// The simple-dgemm is a simple program that adds multiplies two matrices of
// double precision floats and verifies the results. This program is implemented
// using C++ and Data Parallel C++ (DPC++) for Intel(R) CPU and accelerators
// and the oneMKL library.
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

#include <oneapi/mkl.hpp>

using namespace std;
using namespace sycl;

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

int main() {
  cout << "oneMKL DGEMM demo\n";

  const int iterations = 10;
  const int order = 10000;

  cout << "Number of iterations: " << iterations << "\n";
  cout << "Matrix order:         " << order << "\n";

  queue q(default_selector{}, property::queue::in_order{});

  const auto d = q.get_device();
  const auto p = d.get_platform();

  cout << "Device:   " << d.get_info<sycl::info::device::name>() << "\n";
  cout << "Platform: " << p.get_info<sycl::info::platform::name>() << "\n";

  // Allocate space for matrices.
  const size_t alignment = 128 + 4096 * 16;
  const size_t nelems = (size_t)order * (size_t)order;
  const size_t bytes = nelems * sizeof(double);

  double* h_a = sycl::aligned_alloc_host<double>(alignment, nelems, q);
  double* h_b = sycl::aligned_alloc_host<double>(alignment, nelems, q);
  double* h_c = sycl::aligned_alloc_host<double>(alignment, nelems, q);

  for (int i = 0; i < order; ++i) {
    for (int j = 0; j < order; ++j) {
      h_a[i * order + j] = i;
      h_b[i * order + j] = i;
      h_c[i * order + j] = 0;
    }
  }

  // Copy input from host to device.
  double* d_a = sycl::aligned_alloc_device<double>(alignment, nelems, q);
  double* d_b = sycl::aligned_alloc_device<double>(alignment, nelems, q);
  double* d_c = sycl::aligned_alloc_device<double>(alignment, nelems, q);
  q.wait();

  q.memcpy(d_a, &(h_a[0]), bytes);
  q.memcpy(d_b, &(h_b[0]), bytes);
  q.memcpy(d_c, &(h_c[0]), bytes);
  q.wait();

  free(h_a, q);
  free(h_b, q);

  double dgemm_time(0);

  {
    dpc_common::TimeInterval timer;
    double start_time = 0.0;

    for (int i = 0; i <= iterations; i++) {
      if (i == 1) start_time = timer.Elapsed();

      const double alpha = 1.0;
      const double beta = 1.0;
      const auto nt = oneapi::mkl::transpose::nontrans;
      oneapi::mkl::blas::gemm(q, nt, nt,            // opA, opB
                              order, order, order,  // m, n, k
                              alpha,                // alpha
                              d_a, order,           // d_a, lda
                              d_b, order,           // d_b, ldb
                              beta,                 // beta
                              d_c, order);          // d_c, ldc
    }
    q.wait();

    double stop_time = timer.Elapsed();
    dgemm_time = stop_time - start_time;
  }

  // Copy output back to host.
  q.memcpy(&(h_c[0]), d_c, bytes).wait();

  free(d_c, q);
  free(d_b, q);
  free(d_a, q);

  // Analyze and output results.
  const double epsilon = 1.0e-8;
  const double forder = static_cast<double>(order);
  const double reference =
      0.25 * pow(forder, 3) * pow(forder - 1.0, 2) * (iterations + 1);
  const double checksum = reduce(&(h_c[0]), &(h_c[nelems]), 0.0);
  const double residuum = abs(checksum - reference) / reference;

  free(h_c, q);

  if (residuum < epsilon) {
    cout << "Solution validates\n";

    auto avg_time = dgemm_time / iterations;
    auto nflops = 2.0 * pow(forder, 3);

    cout << "Rate (GF/s):  " << (1.0e-9 * nflops / avg_time) << "\n";
    cout << "Avg time (s): " << avg_time << "\n";
  } else {
    cout << "Reference checksum: " << reference << "\n";
    cout << "Residuum:           " << residuum << "\n";

    return 1;
  }

  return 0;
}
