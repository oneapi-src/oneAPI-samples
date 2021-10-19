//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <math.h>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
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
template <int num_copies>
class Kernel;

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// performance impact of the private_copies attribute.
template <int num_copies>
void SimpleMathWithShift(const device_selector &selector, const IntArray &array,
                         int shift, IntScalar &result) {
  double kernel_time = 0.0;

  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array(array);
    buffer<int, 1> buffer_result(result.data(), 1);

    event e = q.submit([&](handler &h) {
      accessor accessor_array(buffer_array, h, read_only);
      accessor accessor_result(buffer_result, h, write_only, no_init);

      h.single_task<Kernel<num_copies>>([=]() [[intel::kernel_args_restrict]] {
        int r = 0;

        for (size_t i = 0; i < kMaxIter; i++) {
          // Request num_copies private copies for array a. This limits the
          // concurrency of the outer loop to num_copies and also limits the
          // memory use of a.
          [[intel::private_copies(num_copies)]] int a[kSize];
          for (size_t j = 0; j < kSize; j++) {
            a[j] = accessor_array[(i * 4 + j) % kSize] * shift;
          }
          for (size_t j = 0; j < kSize; j++) r += a[j];
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
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // The performance of the kernel is measured in GFlops, based on:
  // 1) the number of operations performed by the kernel.
  //    This can be calculated easily for the simple example kernel.
  // 2) the kernel execution time reported by SYCL event profiling.
  std::cout << "Kernel time when private_copies is set to " << num_copies
            << ": " << kernel_time << " ms\n";
  std::cout << "Kernel throughput when private_copies is set to " << num_copies
            << ": ";
  std::cout << std::fixed << std::setprecision(3)
            << ((double)(kTotalOps) / kernel_time) / 1e6f << " GFlops\n";
}

// Calculates the expected results. Used to verify that the kernel
// is functionally correct.
int GoldenResult(const IntArray &input_arr, int shift) {
  int gr = 0;

  for (size_t i = 0; i < kMaxIter; i++) {
    int a[kSize];
    for (size_t j = 0; j < kSize; j++) {
      a[j] = input_arr[(i * 4 + j) % kSize] * shift;
    }
    for (size_t j = 0; j < kSize; j++) gr += a[j];
  }

  return gr;
}

int main() {
  bool success = true;

  IntArray a;
  IntScalar R0, R1, R2, R3, R4;

  int shift = rand() % kMaxValue;

  // initialize the input data
  for (size_t i = 0; i < kSize; i++) a[i] = rand() % kMaxValue;

#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  // Run the kernel with different values of the private_copies
  // attribute to determine the optimal private_copies number.
  SimpleMathWithShift<0>(selector, a, shift, R0);
  SimpleMathWithShift<1>(selector, a, shift, R1);
  SimpleMathWithShift<2>(selector, a, shift, R2);
  SimpleMathWithShift<3>(selector, a, shift, R3);
  SimpleMathWithShift<4>(selector, a, shift, R4);

  // compute the actual result here
  int gr = GoldenResult(a, shift);

  // verify the results are correct
  if (gr != R0[0]) {
    std::cout << "Private copies 0: mismatch: " << R0[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R1[0]) {
    std::cout << "Private copies 1: mismatch: " << R1[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R2[0]) {
    std::cout << "Private copies 2: mismatch: " << R2[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R3[0]) {
    std::cout << "Private copies 3: mismatch: " << R3[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R4[0]) {
    std::cout << "Private copies 4: mismatch: " << R4[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}
