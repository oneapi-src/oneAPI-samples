//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <array>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>

#include "exception_handler.hpp"

using namespace sycl;

constexpr size_t kSize = 8;
constexpr float kErrorThreshold = 0.001;
constexpr int kTotalOps = kSize * (2 * kSize + 1);

using FloatArray = std::array<float, kSize>;
using TwoDimFloatArray = std::array<float, kSize*kSize>;
using FloatScalar = std::array<float, 1>;

// an example complicated operation that creates a long critical path of
// combinational logic from the use of the parameter values to the result
float SomethingComplicated(float x, float y) { return sycl::sqrt(x) * sycl::sqrt(y); }

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int interleaving>
class KernelCompute;

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// performance impact of the max_interleaving attribute.
template <int interleaving>
void Transform(const TwoDimFloatArray &array_a, const FloatArray &array_b, 
               FloatArray &array_r) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  double kernel_time = 0.0;

  try {
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    buffer array_a_buffer(array_a);
    buffer array_b_buffer(array_b);
    buffer array_r_buffer(array_r);

    event e = q.submit([&](handler &h) {
      accessor array_a_accessor(array_a_buffer, h, read_only);
      accessor array_b_accessor(array_b_buffer, h, read_only);
      accessor accessor_array_r(array_r_buffer, h, write_only, no_init);

      h.single_task<KernelCompute<interleaving>>([=]() 
                                                 [[intel::kernel_args_restrict]] {
        float temp_a[kSize*kSize];
        float temp_b[kSize];
        float temp_r[kSize];

        for (size_t i = 0; i < kSize; i++) {
          for (size_t j = 0; j < kSize; j++) {
            temp_a[i*kSize+j] = array_a_accessor[i*kSize+j];
          }
          temp_b[i] = array_b_accessor[i];
          temp_r[i] = 1.0;
        }

        for (size_t i = 0; i < kSize; i++) {
          // only one iteration of the outer loop can be executing the
          // inner loop at a time so that accesses to temp_r occur
          // in the correct order -- use max_interleaving to simplify
          // the datapath and reduce hardware resource usage
          [[intel::max_interleaving(interleaving)]] 
          for (size_t j = 0; j < kSize; j++) {
            temp_r[j] = SomethingComplicated(temp_a[i*kSize+j], temp_r[j]);
          }
          temp_r[i] += temp_b[i];
        }

        for (size_t i = 0; i < kSize; i++) {
          accessor_array_r[i] = temp_r[i];
        }
      });
    });

    // SYCL event profiling allows the kernel execution to be timed
    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();
    kernel_time = (double)(end - start) * 1e-6f;

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:" << '\n' << e.what() << '\n';

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
  // 1) the number of floating-point operations performed by the kernel.
  //    This can be calculated easily for the simple example kernel.
  // 2) the kernel execution time reported by SYCL event profiling.
  std::cout << "Max interleaving " << interleaving << " "
            << "kernel time : " << kernel_time << " ms\n";
  std::cout << "Throughput for kernel with max_interleaving " << interleaving
            << ": ";
  std::cout << std::fixed << std::setprecision(3)
#if defined(FPGA_SIMULATOR)
            << ((double)(kTotalOps) / kernel_time) << " KFlops\n";
#else
            << ((double)(kTotalOps) / kernel_time) / 1e6f << " GFlops\n";
#endif
}

// Calculates the expected results. Used to verify that the kernel
// is functionally correct.
void GoldenResult(const TwoDimFloatArray &A, const FloatArray &B,
                  FloatArray &R) {
  for (size_t i = 0; i < kSize; i++) {
    for (size_t j = 0; j < kSize; j++) {
      R[j] = SomethingComplicated(A[i*kSize+j], R[j]);
    }
    R[i] += B[i];
  }
}

int main() {

  TwoDimFloatArray indata_A;
  FloatArray indata_B;
  FloatArray outdata_R_compute_0;
  FloatArray outdata_R_compute_1;
  FloatArray outdata_R_golden;

  // initialize the input data
  srand(7);
  for (size_t i = 0; i < kSize; i++) {
    indata_B[i] = (float)(rand() % 256);
    for (size_t j = 0; j < kSize; j++) {
      indata_A[i*kSize+j] = (float)(rand() % 256);
    }
    outdata_R_golden[i] = 1.0;
  }

  // Run the kernel with two different values of the max_interleaving
  // attribute. In this case, unlimited interleaving (max_interleaving
  // set to 0) gives no improvement in runtime performance over
  // restricted interleaving (max_interleaving set to 1), despite
  // requiring more hardware resources (see README.md for details
  // on confirming this difference in hardware resource usage in
  // the reports).
  Transform<0>(indata_A, indata_B, outdata_R_compute_0);
  Transform<1>(indata_A, indata_B, outdata_R_compute_1);

  // compute the actual result here
  GoldenResult(indata_A, indata_B, outdata_R_golden);

  // error check for Transform<0>
  bool failed = false;
  for (unsigned i = 0; i < kSize; i++) {
    if (std::abs(outdata_R_compute_0[i] - outdata_R_golden[i]) >
        kErrorThreshold) {
      std::cout << "error at [" << i << "]: " << outdata_R_compute_0[i]
                << " != " << outdata_R_golden[i] << '\n';
      failed = true;
    }
  }

  // error check for Transform<1>
  for (unsigned i = 0; i < kSize; i++) {
    if (std::abs(outdata_R_compute_1[i] - outdata_R_golden[i]) >
        kErrorThreshold) {
      std::cout << "error at [" << i << "]: " << outdata_R_compute_1[i]
                << " != " << outdata_R_golden[i] << '\n';
      failed = true;
    }
  }

  if (failed) {
    std::cout << "FAILED: The results are incorrect\n";
    return 1;
  } else {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }
}
