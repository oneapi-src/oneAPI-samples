//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <chrono>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

constexpr int kInitNumInputs = 16 * 1024 * 1024;  // Default number of inputs.
constexpr int kNumOutputs = 64;                   // Number of outputs
constexpr int kInitSeed = 42;         // Seed for randomizing data inputs
constexpr int kCacheDepth = 5;        // Depth of the cache.
constexpr int kNumRuns = 2;           // runs twice to show the impact of cache
constexpr double kNs = 1000000000.0;  // number of nanoseconds in a second

template<bool use_cache>
class Task;

// This kernel function implements two data paths: with and without caching.
// use_cache specifies which path to take.
template<bool use_cache>
void Histogram(std::unique_ptr<queue>& q, buffer<uint32_t>& input_buf,
               buffer<uint32_t>& output_buf, event& e) {
  // Enqueue  kernel
  e = q->submit([&](handler& h) {
    // Get accessors to the SYCL buffers
    auto input = input_buf.get_access<access::mode::read>(h);
    auto output = output_buf.get_access<access::mode::discard_write>(h);

    h.single_task<Task<use_cache>>([=]() [[intel::kernel_args_restrict]] {

      // On-chip memory for Histogram
      uint32_t local_output[kNumOutputs];
      uint32_t local_output_with_cache[kNumOutputs];

      // Register-based cache of recently-accessed memory locations
      uint32_t last_sum[kCacheDepth + 1];
      uint32_t last_sum_index[kCacheDepth + 1];

      // Initialize Histogram to zero
      for (uint32_t b = 0; b < kNumOutputs; ++b) {
        local_output[b] = 0;
        local_output_with_cache[b] = 0;
      }

      // Compute the Histogram
      if (!use_cache) {  // Without cache
        for (uint32_t n = 0; n < kInitNumInputs; ++n) {
          // Compute the Histogram index to increment
          uint32_t b = input[n] % kNumOutputs;
          local_output[b]++;
        }
      } else {  // With cache

        // Specify that the minimum dependence-distance of
        // loop carried variables is kCacheDepth.
        [[intelfpga::ivdep(kCacheDepth)]] for (uint32_t n = 0;
                                               n < kInitNumInputs; ++n) {
          // Compute the Histogram index to increment
          uint32_t b = input[n] % kNumOutputs;

          // Get the value from the on-chip mem at this index.
          uint32_t val = local_output_with_cache[b];

          // However, if this location in on-chip mem was recently
          // written to, take the value from the cache.
          #pragma unroll
          for (int i = 0; i < kCacheDepth + 1; i++) {
            if (last_sum_index[i] == b) val = last_sum[i];
          }

          // Write the new value to both the cache and the on-chip mem.
          last_sum[kCacheDepth] = local_output_with_cache[b] = val + 1;
          last_sum_index[kCacheDepth] = b;

          // Cache is just a shift register, so shift the shift reg. Pushing
          // into the back of the shift reg is done above.
          #pragma unroll
          for (int i = 0; i < kCacheDepth; i++) {
            last_sum[i] = last_sum[i + 1];
            last_sum_index[i] = last_sum_index[i + 1];
          }
        }
      }

      // Write output to global memory
      for (uint32_t b = 0; b < kNumOutputs; ++b) {
        if (!use_cache) {
          output[b] = local_output[b];
        } else {
          output[b] = local_output_with_cache[b];
        }
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
  INTEL::fpga_emulator_selector device_selector;
  std::cout << "\nEmulator output does not demonstrate true hardware "
               "performance. The design may need to run on actual hardware "
               "to observe the performance benefit of the optimization "
               "exemplified in this tutorial.\n\n";
#else
  INTEL::fpga_selector device_selector;
#endif
  try {
    auto prop_list =
        property_list{property::queue::enable_profiling()};

    std::unique_ptr<queue> q;
    q.reset(new queue(device_selector, dpc_common::exception_handler, prop_list));

    platform platform = q->get_context().get_platform();
    device device = q->get_device();
    std::cout << "Platform name: "
              << platform.get_info<info::platform::name>().c_str() << "\n";
    std::cout << "Device name: "
              << device.get_info<info::device::name>().c_str() << "\n\n\n";

    std::cout << "\nNumber of inputs: " << kInitNumInputs << "\n";
    std::cout << "Number of outputs: " << kNumOutputs << "\n\n";

    // Create input and output buffers
    auto input_buf = buffer<uint32_t>(range<1>(kInitNumInputs));
    auto output_buf = buffer<uint32_t>(range<1>(kNumOutputs));

    srand(kInitSeed);

    // Compute the reference solution
    uint32_t gold[kNumOutputs];

    {
      // Get host-side accessors to the SYCL buffers
      auto input_host = input_buf.get_access<access::mode::write>();
      // Initialize random input
      for (int i = 0; i < kInitNumInputs; ++i) {
        input_host[i] = rand();
      }

      for (int b = 0; b < kNumOutputs; ++b) {
        gold[b] = 0;
      }
      for (int i = 0; i < kInitNumInputs; ++i) {
        int b = input_host[i] % kNumOutputs;
        gold[b]++;
      }
    }

    // Host accessor is now out-of-scope and is destructed. This is required
    // in order to unblock the kernel's subsequent accessor to the same buffer.

    for (int i = 0; i < kNumRuns; i++) {
      switch (i) {
        case 0: {
          std::cout << "Beginning run without on-chip memory caching.\n\n";
          Histogram<false>(q, input_buf, output_buf, e);
          break;
        }
        case 1: {
          std::cout << "Beginning run with on-chip memory caching.\n\n";
          Histogram<true>(q, input_buf, output_buf, e);
          break;
        }
        default: {
          Histogram<false>(q, input_buf, output_buf, e);
        }
      }

      // Wait for kernels to finish
      q->wait();

      // Compute kernel execution time
      t1_kernel = e.get_profiling_info<info::event_profiling::command_start>();
      t2_kernel = e.get_profiling_info<info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / kNs;

      // Get accessor to output buffer. Accessing the buffer at this point in
      // the code will block on kernel completion.
      auto output_host = output_buf.get_access<access::mode::read>();

      // Verify output and print pass/fail
      bool passed = true;
      int num_errors = 0;
      for (int b = 0; b < kNumOutputs; b++) {
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
        double N_MB = (kInitNumInputs * sizeof(uint32_t)) /
                      (1024 * 1024);  // Input size in MB

        // Report kernel execution time and throughput
        std::cout << "Kernel execution time: " << time_kernel << " seconds\n";
        std::cout << "Kernel throughput " << (i == 0 ? "without" : "with")
                  << " caching: " << N_MB / time_kernel << " MB/s\n\n";
      } else {
        std::cout << "Verification FAILED\n";
        return 1;
      }
    }
  } catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
  return 0;
}
