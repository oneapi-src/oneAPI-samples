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

// Seed for randomizing data inputs
constexpr int kInitSeed = 42;

// This tutorial runs twice to show the impact with
// and without the optimization.
constexpr int kNumRuns = 2;

// number of nanoseconds in a second
constexpr double kNs = 1000000000.0;

// Number of inputs. Don't set this too large, otherwise
// computation of the reference solution will take a long time on
// the host (the time is proportional to kSize^2)
constexpr int kSize = 8 * 1024;

// >=1. Minimum number of iterations of the inner loop that must be
// executed in the optimized implementation. Set this approximately
// equal to the ii of inner loop in the unoptimized implementation.
constexpr int kM = 50;

// do not use with unary operators, e.g., kMin(x++, y++)
constexpr int Min(int X, int Y) { return (((X) < (Y)) ? (X) : (Y)); };

// Forward declaration of kernel
class Task;

// This method represents the operation you perform on the loop-carried variable
// in the triangular loop (i.e. a dot product or something that may take many
// cycles to complete).
int SomethingComplicated(int x) { return (int)sycl::sqrt((float)x); }

// This kernel function implements two data paths: with and without the
// optimization. 'optimize' specifies which path to take.
void TriangularLoop(std::unique_ptr<queue>& q, buffer<uint32_t>& input_buf,
                    buffer<uint32_t>& output_buf, uint32_t n, event& e,
                    bool optimize) {
  // Enqueue kernel
  e = q->submit([&](handler& h) {
    // Get accessors to the SYCL buffers
    auto input = input_buf.get_access<access::mode::read>(h);
    auto output = output_buf.get_access<access::mode::discard_write>(h);

    h.single_task<Task>([=]() [[intel::kernel_args_restrict]] {
      // See README for description of the loop_bound calculation.
      const int real_iterations = (n * (n + 1) / 2 - 1);
      const int extra_dummy_iterations = (kM - 2) * (kM - 1) / 2;
      const int loop_bound = real_iterations + extra_dummy_iterations;

      // Local memory for the buffer to be operated on
      uint32_t local_buf[kSize];

      // Read the input_buf from global mem and load it into the local mem
      for (uint32_t i = 0; i < kSize; i++) {
        local_buf[i] = input[i];
      }

      // Perform the triangular loop computation

      if (!optimize) {  // Unoptimized loop.

        for (int x = 0; x < n; x++) {
          for (int y = x + 1; y < n; y++) {
            local_buf[y] = local_buf[y] + SomethingComplicated(local_buf[x]);
          }
        }

      } else {  // Optimized loop.

        // Indices to track the execution inside the single, merged loop.
        int x = 0, y = 1;

        // Specify that the minimum dependence-distance of loop-carried
        // variables is kM iterations. We ensure this is true by modifying the y
        // index such that a minimum of kM iterations are always executed.
        [[intelfpga::ivdep(kM)]] for (int i = 0; i < loop_bound; i++) {
          // Determine if this iteration is a dummy iteration or a real
          // iteration in which the computation should be performed.
          bool compute = y > x;
          // Perform the computation if needed.
          if (compute) {
            local_buf[y] = local_buf[y] + SomethingComplicated(local_buf[x]);
          }
          // Figure out the next value for the indices.
          y++;

          // If we've hit the end, set y such that a minimum of kM
          // iterations are exected.
          if (y == n) {
            x++;
            y = Min(n - kM, x + 1);
          }
        }
      }

      // Write the output to global mem
      for (uint32_t i = 0; i < kSize; i++) {
        output[i] = local_buf[i];
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

    // Create input and output buffers
    auto input_buf = buffer<uint32_t>(range<1>(kSize));
    auto output_buf = buffer<uint32_t>(range<1>(kSize));

    srand(kInitSeed);

    // Compute the reference solution
    uint32_t gold[kSize];

    {
      // Get host-side accessors to the SYCL buffers.
      auto input_host = input_buf.get_access<access::mode::write>();

      // Initialize random input
      for (int i = 0; i < kSize; ++i) {
        input_host[i] = rand() % 256;
      }

      for (int i = 0; i < kSize; ++i) {
        gold[i] = input_host[i];
      }
    }

    // Host accessor now out-of-scope and is destructed. This is required in
    // order to unblock the kernel's subsequent accessor to the same buffer.

    for (int x = 0; x < kSize; x++) {
      for (int y = x + 1; y < kSize; y++) {
        gold[y] += SomethingComplicated(gold[x]);
      }
    }

    std::cout << "Length of input array: " << kSize << "\n\n";

    for (int i = 0; i < kNumRuns; i++) {
      switch (i) {
        case 0: {
          std::cout
              << "Beginning run without triangular loop optimization.\n\n";
          TriangularLoop(q, input_buf, output_buf, kSize, e, false);
          break;
        }
        case 1: {
          std::cout << "Beginning run with triangular loop optimization.\n\n";
          TriangularLoop(q, input_buf, output_buf, kSize, e, true);
          break;
        }
        default: {
          TriangularLoop(q, input_buf, output_buf, kSize, e, false);
        }
      }

      // Wait for kernels to finish
      q->wait();

      t1_kernel = e.get_profiling_info<info::event_profiling::command_start>();
      t2_kernel = e.get_profiling_info<info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / kNs;

      // Get accessor to output buffer. Accessing the buffer at this point in
      // the code will block on kernel completion.
      auto output_host = output_buf.get_access<access::mode::read>();

      // Verify output and print pass/fail
      bool passed = true;
      int num_errors = 0;
      for (int b = 0; b < kSize; b++) {
        if (num_errors < 10 && output_host[b] != gold[b]) {
          passed = false;
          std::cout << " Mismatch at element " << b << ". expected " << gold[b]
                    << ")\n";
          num_errors++;
        }
      }

      if (passed) {
        std::cout << "Verification PASSED\n\n";

        // Report host execution time and throughput
        std::cout.setf(std::ios::fixed);
        std::cout << "Execution time: " << time_kernel << " seconds\n";
        int num_iterations =
            kSize * (kSize + 1) / 2 -
            1;  // One piece of data is processed on each iteration. This
                // formula is taken from the loop_bound calculation.
        double N_MB = (sizeof(uint32_t) * num_iterations) /
                      (1024 * 1024);  // Amount of data processed, in mB
        std::cout << "Throughput " << (i == 0 ? "without" : "with")
                  << " optimization: " << N_MB / time_kernel << " MB/s\n\n";
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
