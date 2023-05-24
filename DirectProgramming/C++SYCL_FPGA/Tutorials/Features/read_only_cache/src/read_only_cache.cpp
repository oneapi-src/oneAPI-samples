//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <chrono>

#include "exception_handler.hpp"

using namespace sycl;
namespace ext_oneapi = sycl::ext::oneapi;

constexpr int kLUTSize = 512;       // Size of the LUT.
constexpr int kNumOutputs = 131072; // Number of outputs.
constexpr double kNs = 1e9;         // number of nanoseconds in a second

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class SqrtTest;

// Below are three different random number generators that are used to generate
// arbitrary indices to be used when accessing the LUT. Each generator is
// implemented using a "Linear-Feedback Shift Register" (LFSR).
uint16_t rand1(uint16_t &index, uint16_t &bits) {
  bits = ((index >> 0) ^ (index >> 3) ^ (index >> 4) ^ (index >> 5)) & 1u;
  return index = (index >> 1) | (bits << 15);
}

uint16_t rand2(uint16_t &index, uint16_t &bits) {
  bits = ((index >> 0) ^ (index >> 1) ^ (index >> 2) ^ (index >> 3)) & 1u;
  return index = (index >> 1) | (bits << 15);
}

uint16_t rand3(uint16_t &index, uint16_t &bits) {
  bits = ((index >> 0) ^ (index >> 1) ^ (index >> 2) ^ (index >> 5)) & 1u;
  return index = (index >> 1) | (bits << 15);
}

event runSqrtTest(sycl::queue &q, const std::vector<float> &sqrt_lut_vec,
                  std::vector<float> &output_vec) {
  buffer sqrt_lut_buf(sqrt_lut_vec);
  buffer output_buf(output_vec);

  event e = q.submit([&](handler &h) {
    accessor sqrt_lut(sqrt_lut_buf, h, read_only,
                      ext_oneapi::accessor_property_list{ext_oneapi::no_alias});
    accessor output(
        output_buf, h, write_only,
        ext_oneapi::accessor_property_list{ext_oneapi::no_alias, no_init});

    h.single_task<SqrtTest>([=]() {
      uint16_t index = 0xFFFu; // An arbitrary non-zero starting state
      uint16_t bits = 0;

      for (int i = 0; i < kNumOutputs; i++)
        output[i] = sqrt_lut[rand1(index, bits) % kLUTSize];

      for (int i = 0; i < kNumOutputs; i++)
        output[i] += sqrt_lut[rand2(index, bits) % kLUTSize];

      for (int i = 0; i < kNumOutputs; i++)
        output[i] += sqrt_lut[rand3(index, bits) % kLUTSize];
    });
  });
  return e;
}

int main() {
  // Host and kernel profiling
  event e;
  unsigned long t1_kernel, t2_kernel;
  double time_kernel;

  // Create input and output vectors
  std::vector<float> sqrt_lut_vec(kLUTSize);
  std::vector<float> output_vec(kNumOutputs);
  for (int i = 0; i < kLUTSize; ++i) {
    sqrt_lut_vec[i] = sqrt(i);
  }

// Create queue, get platform and device
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};

    sycl::queue q(selector, fpga_tools::exception_handler, prop_list);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    std::cout << "\nSQRT LUT size: " << kLUTSize << "\n";
    std::cout << "Number of outputs: " << kNumOutputs << "\n";

    e = runSqrtTest(q, sqrt_lut_vec, output_vec);
    e.wait();

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
  uint16_t index = 0xFFFu; // An arbitrary non-zero starting state
  uint16_t bits = 0;
  float gold[kNumOutputs];
  for (int i = 0; i < kNumOutputs; ++i)
    gold[i] = sqrt_lut_vec[rand1(index, bits) % kLUTSize];

  for (int i = 0; i < kNumOutputs; ++i)
    gold[i] += sqrt_lut_vec[rand2(index, bits) % kLUTSize];

  for (int i = 0; i < kNumOutputs; ++i)
    gold[i] += sqrt_lut_vec[rand3(index, bits) % kLUTSize];

  // Verify output and print pass/fail
  bool passed = true;
  int num_errors = 0;
  for (int b = 0; b < kNumOutputs; b++) {
    if (num_errors < 10 && output_vec[b] != gold[b]) {
      passed = false;
      std::cerr << " (mismatch, expected " << gold[b] << ")\n";
      num_errors++;
    }
  }

  if (passed) {
    std::cout << "Verification PASSED\n\n";

    // Report host execution time and throughput
    std::cout.setf(std::ios::fixed);

    // Input size in MB
    constexpr double num_mb =
        (static_cast<double>(kNumOutputs * sizeof(uint32_t))) / (1024 * 1024);

    // Report kernel execution time and throughput
    std::cout << "Kernel execution time: " << time_kernel << " seconds\n";
#if defined(CACHE_ENABLED)
    std::cout << "Kernel throughput with the read-only cache: "
#else
    std::cout << "Kernel throughput: "
#endif
              << (num_mb / time_kernel) << " MB/s\n\n";
  } else {
    std::cerr << "Verification FAILED\n";
    return 1;
  }
  return 0;
}
