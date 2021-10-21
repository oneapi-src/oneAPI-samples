//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <chrono>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

constexpr int kLUTSize = 512;         // Size of the LUT.
constexpr int kNumOutputs = 524288;   // Number of outputs.
constexpr double kNs = 1000000000.0;  // number of nanoseconds in a second

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class SqrtTest;

void runSqrtTest(sycl::queue &q, const std::vector<float> &sqrt_lut_vec,
                 std::vector<float> &output_vec, event &e) {
  range<1> N{output_vec.size()};
  buffer sqrt_lut_buf(sqrt_lut_vec);
  buffer output_buf(output_vec.data(), N);

  // Enqueue  kernel
  e = q.submit([&](handler &h) {
    // Get accessors to the SYCL buffers
    accessor sqrt_lut(sqrt_lut_buf, h, read_only,
                      accessor_property_list{no_alias});
    accessor output(output_buf, h, write_only,
                    accessor_property_list{no_alias, no_init});

    h.single_task<SqrtTest>([=]() {
      uint16_t index = 0xFFFu;
      uint16_t bits = 0;

      for (int i = 0; i < kNumOutputs; i++) {
        bits = ((index >> 0) ^ (index >> 3) ^ (index >> 4) ^ (index >> 5)) & 1u;
        index = (index >> 1) | (bits << 15);
        output[i] = sqrt_lut[index % kLUTSize];
      }

      for (int i = 0; i < kNumOutputs; i++) {
        bits = ((index >> 0) ^ (index >> 1) ^ (index >> 2) ^ (index >> 3)) & 1u;
        index = (index >> 1) | (bits << 15);
        output[i] += sqrt_lut[index % kLUTSize];
      }

      for (int i = 0; i < kNumOutputs; i++) {
        bits = ((index >> 0) ^ (index >> 1) ^ (index >> 2) ^ (index >> 5)) & 1u;
        index = (index >> 1) | (bits << 15);
        output[i] += sqrt_lut[index % kLUTSize];
      }
    });
  });
}

int main() {
  // Host and kernel profiling
  event e;
  ulong t1_kernel, t2_kernel;
  double time_kernel;

  // Create input and output vectors
  std::vector<float> sqrt_lut_vec(kLUTSize);
  std::vector<float> output_vec(kNumOutputs);
  for (int i = 0; i < kLUTSize; ++i) {
    sqrt_lut_vec[i] = sqrt(i);
  }

// Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
  std::cout << "\nEmulator output does not demonstrate true hardware "
               "performance. The design may need to run on actual hardware "
               "to observe the performance benefit of the optimization "
               "exemplified in this tutorial.\n\n";
#else
  ext::intel::fpga_selector device_selector;
#endif
  try {
    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};

    sycl::queue q(device_selector, dpc_common::exception_handler, prop_list);

    platform platform = q.get_context().get_platform();
    device device = q.get_device();
    std::cout << "Platform name: "
              << platform.get_info<info::platform::name>().c_str() << "\n";
    std::cout << "Device name: "
              << device.get_info<info::device::name>().c_str() << "\n\n\n";

    std::cout << "\nSQRT LUT size: " << kLUTSize << "\n";
    std::cout << "Number of outputs: " << kNumOutputs << "\n";

    runSqrtTest(q, sqrt_lut_vec, output_vec, e);

    // Wait for kernels to finish
    q.wait();

    // Compute kernel execution time
    t1_kernel = e.get_profiling_info<info::event_profiling::command_start>();
    t2_kernel = e.get_profiling_info<info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / kNs;

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

  // Compute the reference solution
  uint16_t index = 0xFFFu;
  uint16_t bits = 0;
  float gold[kNumOutputs];
  for (int i = 0; i < kNumOutputs; ++i) {
    bits = ((index >> 0) ^ (index >> 3) ^ (index >> 4) ^ (index >> 5)) & 1u;
    index = (index >> 1) | (bits << 15);
    gold[i] = sqrt_lut_vec[index % kLUTSize];
  }

  for (int i = 0; i < kNumOutputs; ++i) {
    bits = ((index >> 0) ^ (index >> 1) ^ (index >> 2) ^ (index >> 3)) & 1u;
    index = (index >> 1) | (bits << 15);
    gold[i] += sqrt_lut_vec[index % kLUTSize];
  }

  for (int i = 0; i < kNumOutputs; ++i) {
    bits = ((index >> 0) ^ (index >> 1) ^ (index >> 2) ^ (index >> 5)) & 1u;
    index = (index >> 1) | (bits << 15);
    gold[i] += sqrt_lut_vec[index % kLUTSize];
  }

  // Verify output and print pass/fail
  bool passed = true;
  int num_errors = 0;
  for (int b = 0; b < kNumOutputs; b++) {
    if (num_errors < 10 && output_vec[b] != gold[b]) {
      passed = false;
      std::cout << " (mismatch, expected " << gold[b] << ")\n";
      num_errors++;
    }
  }

  if (passed) {
    std::cout << "Verification PASSED\n\n";

    // Report host execution time and throughput
    std::cout.setf(std::ios::fixed);
    double N_MB =
        (kNumOutputs * sizeof(uint32_t)) / (1024 * 1024); // Input size in MB

    // Report kernel execution time and throughput
    std::cout << "Kernel execution time: " << time_kernel << " seconds\n";
    std::cout << "Kernel throughput " << N_MB / time_kernel << " MB/s\n\n";
  } else {
    std::cout << "Verification FAILED\n";
    return 1;
  }
  return 0;
}
