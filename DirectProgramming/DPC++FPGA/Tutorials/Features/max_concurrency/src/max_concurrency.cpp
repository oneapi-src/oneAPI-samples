//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <array>
#include <iomanip>
#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

constexpr size_t kSize = 8192;
constexpr size_t kMaxIter = 50000;
constexpr size_t kTotalOps = 2 * kMaxIter * kSize;
constexpr size_t kMaxValue = 128;

using IntArray = std::array<int, kSize>;
using IntScalar = std::array<int, 1>;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int concurrency>
class Kernel;

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// performance impact of the max_concurrency attribute.
template <int concurrency>
void PartialSumWithShift(const device_selector &selector, const IntArray &array,
                         int shift, IntScalar &result) {
  double kernel_time = 0.0;

  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array(array);
    buffer<int, 1> buffer_result(result.data(), 1);

    event e = q.submit([&](handler &h) {
      accessor accessor_array(buffer_array, h, read_only);
      accessor accessor_result(buffer_result, h, write_only, noinit);

      h.single_task<Kernel<concurrency>>([=]() [[intel::kernel_args_restrict]] {
        int r = 0;

        // At most concurrency iterations of the outer loop will be
        // active at one time.
        // This limits memory usage, since each iteration of the outer
        // loop requires its own copy of a1.
        [[intel::max_concurrency(concurrency)]] for (size_t i = 0; i < kMaxIter;
                                                     i++) {
          int a1[kSize];
          for (size_t j = 0; j < kSize; j++)
            a1[j] = accessor_array[(i * 4 + j) % kSize] * shift;
          for (size_t j = 0; j < kSize; j++) r += a1[j];
        }
        accessor_result[0] = r;
      });
    });

    // SYCL event profiling allows the kernel execution to be timed
    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();
    kernel_time = (double)(end - start) * 1e-6;

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // The performance of the kernel is measured in MIPS, based on:
  // 1) the number of arithmetic operations performed by the kernel.
  //    This can be calculated easily for the simple example kernel.
  // 2) the kernel execution time reported by SYCL event profiling.
  std::cout << "Max concurrency " << concurrency << " "
            << "kernel time : " << kernel_time << " ms\n";
  std::cout << "Throughput for kernel with max_concurrency " << concurrency
            << ": ";
  std::cout << std::fixed << std::setprecision(3)
            << ((double)(kTotalOps) / kernel_time) / 1e3f << " MIPS\n";
}

// Calculates the expected results. Used to verify that the kernel
// is functionally correct.
int GoldenResult(const IntArray &A, int shift) {
  int gr = 0;
  for (size_t i = 0; i < kMaxIter; i++) {
    int a1[kSize];
    for (size_t j = 0; j < kSize; j++) a1[j] = A[(i * 4 + j) % kSize] * shift;
    for (size_t j = 0; j < kSize; j++) gr += a1[j];
  }
  return gr;
}

int main() {
  bool success = true;

  IntArray A;
  IntScalar R0, R1, R2, R3, R4, R5;

  int shift = rand() % kMaxValue;

  // initialize the input data
  for (size_t i = 0; i < kSize; i++) A[i] = rand() % kMaxValue;

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // Run the kernel with different values of the max_concurrency
  // attribute, to determine the optimal concurrency.
  // In this case, the optimal max_concurrency is 2 since this
  // achieves the highest throughput. Higher values of max_concurrency
  // consume additional RAM without increasing throughput.
  PartialSumWithShift<0>(selector, A, shift, R0);
  PartialSumWithShift<1>(selector, A, shift, R1);
  PartialSumWithShift<2>(selector, A, shift, R2);
  PartialSumWithShift<4>(selector, A, shift, R3);
  PartialSumWithShift<8>(selector, A, shift, R4);
  PartialSumWithShift<16>(selector, A, shift, R5);

  // compute the actual result here
  int gr = GoldenResult(A, shift);

  // verify the results are correct
  if (gr != R0[0]) {
    std::cout << "Max Concurrency 0: mismatch: " << R0[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R1[0]) {
    std::cout << "Max Concurrency 1: mismatch: " << R1[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R2[0]) {
    std::cout << "Max Concurrency 2: mismatch: " << R2[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R3[0]) {
    std::cout << "Max Concurrency 4: mismatch: " << R3[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R4[0]) {
    std::cout << "Max Concurrency 8: mismatch: " << R4[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R5[0]) {
    std::cout << "Max Concurrency 16: mismatch: " << R5[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}
