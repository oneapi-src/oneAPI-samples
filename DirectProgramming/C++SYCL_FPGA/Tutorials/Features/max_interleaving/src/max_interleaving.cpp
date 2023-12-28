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

#if defined(FPGA_SIMULATOR) || defined(FPGA_EMULATOR)
// Simulator runs too slowly for large array sizes
// Emulator has stack issues for large array sizes -
// (malloc can be used but is out of scope of this tutorial)
constexpr size_t kSize = 32;
#else
constexpr size_t kSize = 128;
#endif
constexpr float kErrorThreshold = 0.5;
constexpr int kTotalOps = 4 * kSize * kSize;

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
void Transform(const TwoDimFloatArray &array_a, FloatArray &array_r) {
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
    buffer array_r_buffer(array_r);

    event e = q.submit([&](handler &h) {
      accessor array_a_accessor(array_a_buffer, h, read_only);
      accessor accessor_array_r(array_r_buffer, h, write_only, no_init);

      h.single_task<KernelCompute<interleaving>>([=]() 
                                                 [[intel::kernel_args_restrict]] {
        float temp_a[kSize*kSize];
        float temp_r[kSize];

        for (size_t i = 0; i < kSize; i++) {
          for (size_t j = 0; j < kSize; j++) {
            temp_a[i*kSize+j] = array_a_accessor[i*kSize+j];
          }
          temp_r[i] = 1.0;
        }

        // A simple row reduction where row i of temp_a is summed and 
        // stored in temp_r[i].
        // Notice how temp_r[i] is a loop carried dependency in the inner loop as it is updated 
        // every iteration. As a result, the *inner loop II is very high* as a new iteration from 
        // the *same* outer loop invocation must wait for the previous iteration to finish updating
        // temp_r[i].
        // However, notice how *no* loop carried memory dependency exists with respect to 
        // the outer loop - for different i-iterations, the temp_r array is read and written to
        // in different locations. 
        // The lack of outer loop carried memory dependencies and a high inner loop II is what 
        // allows interleaving to happen - where multiple invocations of the inner loop
        // concurrently execute on the same inner loop hardware. This is like pipelining, 
        // but each iteration executing in the inner loop is from *different* invocations.
        outer: 
        for (int i = kSize - 1; i >= 0; i--) {
          // You can explicitly disable interleaving with this attribute by providing `1` as 
          // a parameter. `0` keeps it enabled, so long as interleaving is possible.
          // This may result in area savings at the cost of throughput, which could 
          // be useful for non-critical data paths in low-area settings. 
          inner: 
          [[intel::max_interleaving(interleaving)]]
          for (int j = kSize - 1; j >= 0; j--) {
            temp_r[i] +=
                SomethingComplicated(temp_a[i * kSize + j], temp_r[i]);
          }
          // One final note - the loop induction variables decrease (i--) instead of increase (i++)
          // in these two loops to prevent loop fusion optimizations, which makes it harder to 
          // keep track of loops in the optimization reports. Interleaving will still occur if  
          // `i` and `j` were instead incremented.
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
void GoldenResult(const TwoDimFloatArray &A, FloatArray &R) {
  outer: for (int i = kSize - 1; i >= 0; i--) {
    inner: for (int j = kSize - 1; j >= 0; j--) {
      R[i] +=
          SomethingComplicated(A[i * kSize + j], R[i]);
    }
  }
}

int main() {

  TwoDimFloatArray indata_A;
  FloatArray outdata_R_compute_0;
  FloatArray outdata_R_compute_1;
  FloatArray outdata_R_golden;

  // initialize the input data
  srand(7);
  for (size_t i = 0; i < kSize; i++) {
    for (size_t j = 0; j < kSize; j++) {
      indata_A[i*kSize+j] = (float)(rand() % 32);
    }
    outdata_R_golden[i] = 1.0;
  }

  // Run the kernel with two different values of the max_interleaving
  // attribute: 
  //   Enabled - max_interleaving = 0
  //   Disabled - max_interleaving = 1
  // When interleaving is disabled, runtime performance may decrease while
  // hardware resources may increase (see README.md for details
  // on confirming this difference in hardware resource usage in
  // the reports).
  Transform<0>(indata_A, outdata_R_compute_0);
  Transform<1>(indata_A, outdata_R_compute_1);

  // compute the actual result here
  GoldenResult(indata_A, outdata_R_golden);

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
