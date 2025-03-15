//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include "mkl.h"
#include "oneapi/mkl/blas.hpp"
#include <sycl/sycl.hpp>

// Helper functions
template <typename fp> fp set_fp_value(fp arg1, fp arg2 = 0.0) {
  return (arg1 + 0.0 * arg2);
}

template <typename fp>
std::complex<fp> set_fp_value(std::complex<fp> arg1,
                              std::complex<fp> arg2 = 0.0) {
  return std::complex<fp>(arg1.real(), arg2.real());
}

template <typename fp>
void init_matrix(fp *M, oneapi::mkl::transpose trans, int64_t m, int64_t n,
                 int64_t ld) {
  if (trans == oneapi::mkl::transpose::nontrans) {
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = 0; i < m; i++)
        M[i + j * ld] = fp(i + j);
  } else {
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++)
        M[j + i * ld] = fp(i + j);
  }
}

// GEMM operation is performed as shown below.
// C = alpha * op(A) * op(B) + beta * C

template <typename fp> void run_gemm(const sycl::device &dev) {
  // Initialize data for GEMM
  oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

  // Matrix data sizes
  int64_t m = 1024;
  int64_t n = 1024;
  int64_t k = 1024;

  // Leading dimensions of matrices
  int64_t ldA = 1024;
  int64_t ldB = 1024;
  int64_t ldC = 1024;

  // Set scalar fp values
  fp alpha = set_fp_value(2.0, -0.5);
  fp beta = set_fp_value(3.0, -1.5);

  // Create devices for multi-device run using
  // the same platform as input device
  auto devices = dev.get_platform().get_devices();
  sycl::queue device0_queue;
  sycl::queue device1_queue;

  int64_t nb_device = devices.size();

  // nb_device = 1    Example will use 1 device
  // nb_device = 2    Example will use 2 devices with explicit scaling
  // nb_device > 2    Example will use only 2 devices with explicit scaling

  // Catch asynchronous exceptions
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                  << e.what() << std::endl;
      }
    }
  };

  // Create context and execution queue
  devices.push_back(dev);
  sycl::context cxt(devices);
  sycl::queue main_queue(cxt, dev, exception_handler);

  // If two devices are detected, create queue for each device
  if (nb_device > 1) {
    device0_queue = sycl::queue(cxt, devices[0], exception_handler);
    device1_queue = sycl::queue(cxt, devices[1], exception_handler);
  }

  // Allocate and initialize arrays for matrices
  int64_t sizea, sizeb;
  if (transA == oneapi::mkl::transpose::nontrans)
    sizea = ldA * k;
  else
    sizea = ldA * m;
  if (transB == oneapi::mkl::transpose::nontrans)
    sizeb = ldB * n;
  else
    sizeb = ldB * k;
  int64_t sizec = ldC * n;

  auto A_host = sycl::malloc_host<fp>(sizea, main_queue);
  auto B_host = sycl::malloc_host<fp>(sizeb, main_queue);
  auto C_host = sycl::malloc_host<fp>(sizec, main_queue);

  init_matrix(A_host, transA, m, k, ldA);
  init_matrix(B_host, transB, k, n, ldB);
  init_matrix(C_host, oneapi::mkl::transpose::nontrans, m, n, ldC);

  // Copy A/B/C from host to device(s)
  // When two devices are detected,
  // GEMM operation is split between devices in n direction
  // All A matrix is copied to both devices.
  // B and C matrices are split between devices, so only half of B and C are
  // copied to each device.

  fp *A_dev0, *A_dev1, *B_dev0, *B_dev1, *C_dev0, *C_dev1, *A_dev, *B_dev,
      *C_dev;

  if (nb_device > 1) {
    A_dev0 = sycl::malloc_device<fp>(sizea, device0_queue);
    A_dev1 = sycl::malloc_device<fp>(sizea, device1_queue);
    B_dev0 = sycl::malloc_device<fp>(sizeb / 2, device0_queue);
    B_dev1 = sycl::malloc_device<fp>(sizeb / 2, device1_queue);
    C_dev0 = sycl::malloc_device<fp>(sizec / 2, device0_queue);
    C_dev1 = sycl::malloc_device<fp>(sizec / 2, device1_queue);
    // Entire A matrix is copied to both devices
    device0_queue.memcpy(A_dev0, A_host, sizea * sizeof(fp));
    device1_queue.memcpy(A_dev1, A_host, sizea * sizeof(fp));
    // Half of B and C are copied to each device
    device0_queue.memcpy(B_dev0, B_host, (sizeb / 2) * sizeof(fp));
    device1_queue.memcpy(B_dev1, B_host + ldB * n / 2,
                         (sizeb / 2) * sizeof(fp));
    device0_queue.memcpy(C_dev0, C_host, (sizec / 2) * sizeof(fp));
    device1_queue.memcpy(C_dev1, C_host + ldC * n / 2,
                         (sizec / 2) * sizeof(fp));
    device0_queue.wait();
    device1_queue.wait();
  } else {
    A_dev = sycl::malloc_device<fp>(sizea, main_queue);
    B_dev = sycl::malloc_device<fp>(sizeb, main_queue);
    C_dev = sycl::malloc_device<fp>(sizec, main_queue);
    main_queue.memcpy(A_dev, A_host, sizea * sizeof(fp));
    main_queue.memcpy(B_dev, B_host, sizeb * sizeof(fp));
    main_queue.memcpy(C_dev, C_host, sizec * sizeof(fp));
    main_queue.wait();
  }

  // Call oneMKL GEMM API
  // When two devices are detected, GEMM call is launched on each device
  sycl::event gemm_done0;
  sycl::event gemm_done1;
  std::vector<sycl::event> gemm_dependencies;
  try {
    if (nb_device > 1) {
      // Split B and C for each device
      int64_t n_half = n / 2;
      gemm_done0 = oneapi::mkl::blas::gemm(
          device0_queue, transA, transB, m, n_half, k, alpha, A_dev0, ldA,
          B_dev0, ldB, beta, C_dev0, ldC, gemm_dependencies);
      gemm_done1 = oneapi::mkl::blas::gemm(
          device1_queue, transA, transB, m, n_half, k, alpha, A_dev1, ldA,
          B_dev1, ldB, beta, C_dev1, ldC, gemm_dependencies);
    } else {
      gemm_done0 = oneapi::mkl::blas::gemm(main_queue, transA, transB, m, n, k,
                                           alpha, A_dev, ldA, B_dev, ldB, beta,
                                           C_dev, ldC, gemm_dependencies);
    }
  } catch (sycl::exception const &e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
              << e.what() << std::endl;
  }

  // Wait for GEMM calls to finish
  gemm_done0.wait();
  if (nb_device > 1)
    gemm_done1.wait();

  // Copy C from device(s) to host
  if (nb_device > 1) {
    device0_queue.memcpy(C_host, C_dev0, (sizec / 2) * sizeof(fp));
    device1_queue.memcpy(C_host + ldC * n / 2, C_dev1,
                         (sizec / 2) * sizeof(fp));
    device0_queue.wait();
    device1_queue.wait();
  } else {
    main_queue.memcpy(C_host, C_dev, sizec * sizeof(fp));
    main_queue.wait();
  }

  // Clean-up
  free(A_host, cxt);
  free(B_host, cxt);
  free(C_host, cxt);
  if (nb_device > 1) {
    sycl::free(A_dev0, device0_queue);
    sycl::free(A_dev1, device1_queue);
    sycl::free(B_dev0, device0_queue);
    sycl::free(B_dev1, device1_queue);
    sycl::free(C_dev0, device0_queue);
    sycl::free(C_dev1, device1_queue);
  } else {
    sycl::free(A_dev, main_queue);
    sycl::free(B_dev, main_queue);
    sycl::free(C_dev, main_queue);
  }

  if (nb_device > 1)
    std::cout << "\tGEMM operation is complete on 2 devices" << std::endl;
  else
    std::cout << "\tGEMM operation is complete on 1 device" << std::endl;
}

// Main entry point for example.
// GEMM example is run on GPU for 4 data types.

int main() {
  // Create GPU device
  sycl::device dev = sycl::device(sycl::gpu_selector_v);

  std::cout << "Running with single precision real data type:" << std::endl;
  run_gemm<float>(dev);

  std::cout << "Running with double precision real data type:" << std::endl;
  run_gemm<double>(dev);

  std::cout << "Running with single precision complex data type:" << std::endl;
  run_gemm<std::complex<float>>(dev);

  std::cout << "Running with double precision complex data type:" << std::endl;
  run_gemm<std::complex<double>>(dev);
  return 0;
}
// Snippet end
