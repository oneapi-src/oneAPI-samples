//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <algorithm>
#include <chrono>

#include "onchip_memory_with_cache.hpp" // DirectProgramming/C++SYCL_FPGA/include
#include "unrolled_loop.hpp"            // DirectProgramming/C++SYCL_FPGA/include

#include "exception_handler.hpp"

#if defined(FPGA_SIMULATOR)
// Smaller size to keep the runtime reasonable
constexpr int kInitNumInputs = 16 * 1024;  // Default number of inputs
// Only test a single cache depth in simulation mode
constexpr int kMaxCacheDepth = 5; // max cache depth to test
constexpr int kMinCacheDepth = 5; // min cache depth to test
#else
constexpr int kInitNumInputs = 16 * 1024 * 1024;  // Default number of inputs
constexpr int kMaxCacheDepth = MAX_CACHE_DEPTH; // max cache depth to test
constexpr int kMinCacheDepth = MIN_CACHE_DEPTH; // min cache depth to test
#endif

constexpr int kNumOutputs = 64;           // Number of outputs
constexpr int kInitSeed = 42;             // Seed for randomizing data inputs

constexpr double kNs = 1000000000.0;      // number of nanoseconds in a second

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template<size_t cache_depth> class HistogramID;

template<size_t k_cache_depth>
void ComputeHistogram(sycl::queue &q, sycl::buffer<uint32_t>& input_buf,
                      sycl::buffer<uint32_t>& output_buf, sycl::event& e) {
  // Enqueue  kernel
  e = q.submit([&](sycl::handler& h) {
    // Get accessors to the SYCL buffers
    sycl::accessor input(input_buf, h, sycl::read_only);
    sycl::accessor output(output_buf, h, sycl::write_only, sycl::no_init);

    h.single_task<HistogramID<k_cache_depth>>(
    [=]() [[intel::kernel_args_restrict]] {

      // On-chip memory for Histogram
      // A k_cache_depth of 0 is equivalent to a standard array with no cache
      fpga_tools::OnchipMemoryWithCache<uint32_t, kNumOutputs, k_cache_depth> 
        histogram(0);
      // Compute the Histogram
      for (uint32_t n = 0; n < kInitNumInputs; ++n) {
        uint32_t hist_group = input[n] % kNumOutputs;
        auto hist_count = histogram.read(hist_group);
        hist_count++;
        histogram.write(hist_group, hist_count);
      }

      // Write output to global memory
      for (uint32_t hist_group = 0; hist_group < kNumOutputs; ++hist_group) {
        output[hist_group] = histogram.read(hist_group);
      }
    });
  });
}

int main() {
  // Host and kernel profiling
  sycl::event e;
  unsigned long t1_kernel, t2_kernel;
  double time_kernel;

// Create queue, get platform and device
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

#ifndef FPGA_HARDWARE
  std::cout << "\nEmulator and simulator outputs do not demonstrate "
               "true hardware performance. The design may need to run "
               "on actual hardware to observe the performance benefit "
               "of the optimization exemplified in this tutorial.\n\n";
#endif

  try {
    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};

    sycl::queue q(selector, fpga_tools::exception_handler, prop_list);

    sycl::platform platform = q.get_context().get_platform();
    sycl::device device = q.get_device();
    std::cout << "Platform name: "
              << platform.get_info<sycl::info::platform::name>().c_str() 
              << "\n";
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    std::cout << "\nNumber of inputs: " << kInitNumInputs << "\n";
    std::cout << "Number of outputs: " << kNumOutputs << "\n\n";

    // Create input and output buffers
    auto input_buf = sycl::buffer<uint32_t>(sycl::range<1>(kInitNumInputs));
    auto output_buf = sycl::buffer<uint32_t>(sycl::range<1>(kNumOutputs));

    srand(kInitSeed);

    // Compute the reference solution
    uint32_t gold[kNumOutputs];

    {
      // Get host-side accessors to the SYCL buffers
      sycl::host_accessor input_host(input_buf, sycl::write_only);
      // Initialize random input
      for (int i = 0; i < kInitNumInputs; ++i) {
        input_host[i] = rand();
      }

      for (int hist_group = 0; hist_group < kNumOutputs; ++hist_group) {
        gold[hist_group] = 0;
      }
      for (int i = 0; i < kInitNumInputs; ++i) {
        int hist_group = input_host[i] % kNumOutputs;
        gold[hist_group]++;
      }
    }

    // Host accessor is now out-of-scope and is destructed. This is required
    // in order to unblock the kernel's subsequent accessor to the same buffer.

    // iterate over the cache depths
    for (int i = kMinCacheDepth; i < kMaxCacheDepth + 1; i++) {

      std::cout << "Beginning run with cache depth " << i;
      if (i == 0) { std::cout << " (no cache)"; }
      std::cout << std::endl;

      // ComputeHistogram is templated on the cache depth, and template
      // parameters must be compile time constants. This unrolled loop allows
      // us to convert the runtime variable i into a compile time constant j.
      fpga_tools::UnrolledLoop<kMinCacheDepth, kMaxCacheDepth+1>([&](auto j) {
        if (j == i) {
          ComputeHistogram<j>(q, input_buf, output_buf, e);
        }
      });

      // Wait for kernel to finish
      q.wait();

      // Compute kernel execution time
      t1_kernel = 
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
      t2_kernel = 
        e.get_profiling_info<sycl::info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / kNs;

      // Get accessor to output buffer. Accessing the buffer at this point in
      // the code will block on kernel completion.
      sycl::host_accessor output_host(output_buf);

      // Verify output and print pass/fail, and clear the output buffer
      bool passed = true;
      int num_errors = 0;
      for (int hist_group = 0; hist_group < kNumOutputs; hist_group++) {
        if (num_errors < 10 && output_host[hist_group] != gold[hist_group]) {
          passed = false;
          std::cout << " data mismatch in bucket: " << hist_group
                    << ", expected " << gold[hist_group]
                    << ", received from kernel: " << output_host[hist_group]
                    << std::endl;
          num_errors++;
        }
        output_host[hist_group] = 0;
      }

      if (passed) {
        std::cout << "Data check succeeded for cache depth " << i 
                  << std::endl;
        std::cout.setf(std::ios::fixed);
        double N_MB = (kInitNumInputs * sizeof(uint32_t)) /
                      (1024 * 1024);  // Input size in MB
        std::cout << "Kernel execution time: " << time_kernel << " seconds" 
                  << std::endl;
        std::cout << "Kernel throughput for cache depth " << i << ": "
                  << (N_MB / time_kernel) << " MB/s" << std::endl << std::endl;
      } else {
        std::cout << "Verification FAILED" << std::endl;
        return 1;
      }
    }
    std::cout << "Verification PASSED" << std::endl;

  } catch (sycl::exception const& e) {
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
