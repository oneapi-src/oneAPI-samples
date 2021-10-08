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

constexpr int kLUTSize = 1024;       // Default number of inputs.
constexpr int kNumInputs = 1024; // Default number of inputs.
constexpr int kInitSeed = 42;        // Seed for randomizing data inputs
constexpr int kNumRuns = 2;          // runs twice to show the impact of cache
constexpr double kNs = 1000000000.0; // number of nanoseconds in a second

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class SqrtWithoutCaching;
class SqrtWithCaching;

void ComputeSqrtWithoutCaching(sycl::queue &q, buffer<float> &sqrt_lut_buf,
                               buffer<uint32_t> &input_buf,
                               buffer<float> &output_buf, event &e) {
  // Enqueue  kernel
  e = q.submit([&](handler &h) {
    // Get accessors to the SYCL buffers
    accessor sqrt_lut(sqrt_lut_buf, h, read_only,
                      accessor_property_list{no_alias});
    accessor input(input_buf, h, read_write, accessor_property_list{no_alias});
    accessor output(output_buf, h, write_only,
                    accessor_property_list{no_alias, no_init});

    h.single_task<SqrtWithoutCaching>([=]() {
      for (int i = 0; i < kNumInputs; ++i) {
        output[i] = sqrt_lut[input[i]];
      }
    });
  });
}

void ComputeSqrtWithCaching(sycl::queue &q, buffer<float> &sqrt_lut_buf,
                            buffer<uint32_t> &input_buf,
                            buffer<float> &output_buf, event &e) {
  // Enqueue  kernel
  e = q.submit([&](handler &h) {
    // Get accessors to the SYCL buffers
    accessor sqrt_lut(sqrt_lut_buf, h, read_write,
                      accessor_property_list{no_alias});
    accessor input(input_buf, h, read_write, accessor_property_list{no_alias});
    accessor output(output_buf, h, write_only,
                    accessor_property_list{no_alias, no_init});

    h.single_task<SqrtWithCaching>([=]() {
      for (int i = 0; i < kNumInputs; i++) {
        output[i] = sqrt_lut[input[i]];
      }
    });
  });
}

int main() {
  // Host and kernel profiling
  event e;
  ulong t1_kernel, t2_kernel;
  double time_kernel;

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

    std::cout << "\nSQRT LUT Size: " << kLUTSize << "\n";
    std::cout << "Number of inputs: " << kNumInputs << "\n";

    // Create input and output buffers
    auto sqrt_lut_buf = buffer<float>(range<1>(kLUTSize));
    auto input_buf = buffer<uint32_t>(range<1>(kNumInputs));
    auto output_buf = buffer<float>(range<1>(kNumInputs));

    srand(kInitSeed);

    // Compute the reference solution
    float gold[kNumInputs];

    {
      // Get host-side accessors to the SYCL buffers
      host_accessor sqrt_lut_host(sqrt_lut_buf, write_only);
      host_accessor input_host(input_buf, write_only);
      // Initialize random input
      for (int i = 0; i < kNumInputs; ++i) {
        sqrt_lut_host[i] = sqrt(i);
      }

      for (int i = 0; i < kNumInputs; ++i) {
        input_host[i] = rand() % kLUTSize;
      }

      for (int i = 0; i < kNumInputs; ++i) {
        gold[i] = sqrt_lut_host[input_host[i]];
      }
    }

    // Host accessor is now out-of-scope and is destructed. This is required
    // in order to unblock the kernel's subsequent accessor to the same buffer.

    for (int i = 0; i < kNumRuns; i++) {
      switch (i) {
      case 0: {
        std::cout << "Beginning run with caching disabled.\n\n";
        ComputeSqrtWithoutCaching(q, sqrt_lut_buf, input_buf, output_buf, e);
        break;
      }
      case 1: {
        std::cout << "Beginning run with caching enabled.\n\n";
        ComputeSqrtWithCaching(q, sqrt_lut_buf, input_buf, output_buf, e);
        break;
      }
      default: {
        ComputeSqrtWithoutCaching(q, sqrt_lut_buf, input_buf, output_buf, e);
      }
      }

      // Wait for kernels to finish
      q.wait();

      // Compute kernel execution time
      t1_kernel = e.get_profiling_info<info::event_profiling::command_start>();
      t2_kernel = e.get_profiling_info<info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / kNs;

      // Get accessor to output buffer. Accessing the buffer at this point in
      // the code will block on kernel completion.
      host_accessor output_host(output_buf, read_only);

      // Verify output and print pass/fail
      bool passed = true;
      int num_errors = 0;
      for (int b = 0; b < kNumInputs; b++) {
        if (num_errors < 10 && output_host[b] != gold[b]) {
          passed = false;
          std::cout << " (mismatch, expected " << gold[b] << ")\n";
          num_errors++;
        }
      }

      if (passed) {
        std::cout << "Verification PASSED\n\n";

        // Report host execution time and throughput
        std::cout.setf(std::ios::fixed);
        double N_MB = (kNumInputs * sizeof(uint32_t)) /
                      (1024 * 1024); // Input size in MB

        // Report kernel execution time and throughput
        std::cout << "Kernel execution time: " << time_kernel << " seconds\n";
        std::cout << "Kernel throughput " << (i == 0 ? "without" : "with")
                  << " caching: " << N_MB / time_kernel << " MB/s\n\n";
      } else {
        std::cout << "Verification FAILED\n";
        return 1;
      }
    }
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
  return 0;
}
