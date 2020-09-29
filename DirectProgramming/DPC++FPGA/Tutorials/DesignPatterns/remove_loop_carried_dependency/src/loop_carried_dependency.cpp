//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

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
using namespace std;

// Forward declare the kernel names
// (This will become unnecessary in a future compiler version.)
class UnOptKernel;
class OptKernel;

event Unoptimized(queue &q, const vector<double> &vec_a,
                  const vector<double> &vec_b, double &result, size_t N) {
  buffer b_a(vec_a);
  buffer b_b(vec_b);
  buffer b_result(&result, range(1));

  auto e = q.submit([&](handler &h) {
    auto a = b_a.get_access<access::mode::read>(h);
    auto b = b_b.get_access<access::mode::read>(h);
    auto result = b_result.get_access<access::mode::discard_write>(h);

    h.single_task<UnOptKernel>([=]() {
      double sum = 0;
      for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
          sum += a[i * N + j];
        }
        sum += b[i];
      }
      result[0] = sum;
    });
  });
  return e;
}

event Optimized(queue &q, const vector<double> &vec_a,
                const vector<double> &vec_b, double &result, size_t N) {
  buffer b_a(vec_a);
  buffer b_b(vec_b);
  buffer b_result(&result, range(1));

  auto e = q.submit([&](handler &h) {
    auto a = b_a.get_access<access::mode::read>(h);
    auto b = b_b.get_access<access::mode::read>(h);
    auto result = b_result.get_access<access::mode::discard_write>(h);

    h.single_task<OptKernel>([=]() [[intel::kernel_args_restrict]] {
      double sum = 0;

      for (size_t i = 0; i < N; i++) {
        // Step 1: Definition
        double sum_2 = 0;

        // Step 2: Accumulation of array A values for one outer loop iteration
        for (size_t j = 0; j < N; j++) {
          sum_2 += a[i * N + j];
        }

        // Step 3: Addition of array B value for an outer loop iteration
        sum += sum_2;
        sum += b[i];
      }

      result[0] = sum;
    });
  });
  return e;
}

void PrintTime(const event &e, queue &q, const char *kind) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (double)(end_k - start_k) * 1e-6;

  cout << "Run: " << kind << ":\n";
  cout << "kernel time : " << kernel_time << " ms\n";
}

int main(int argc, char *argv[]) {
  size_t n = 16000;

  if (argc > 1) {
    string option(argv[1]);
    if (option == "-h" || option == "--help") {
      cout << "Usage: <executable> <data size>\n\nFAILED\n";
      return 1;
    } else {
      n = stoi(option);
    }
  }
  // Cap the value of n.
  n = std::max(std::min((size_t)n, (size_t)16000), (size_t)100);
  cout << "Number of elements: " << n << '\n';

  vector<double> vec_a(n * n);
  vector<double> vec_b(n);

  double answer = 0;

  // initialize data and compute golden result
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      vec_a[i * n + j] = i + j;
      answer += i + j;
    }
    vec_b[i] = i;
    answer += i;
  }

  // Initialize queue with device selector and enabling profiling
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
  cout << "\nEmulator output does not demonstrate true hardware "
          "performance. The design may need to run on actual hardware "
          "to observe the performance benefit of the optimization "
          "exemplified in this tutorial.\n\n";
#else
  INTEL::fpga_selector selector;
#endif

  double unopt_sum = -1, opt_sum = -1;

  try {
    // Create a profiling queue
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    // compute result on device
    PrintTime(Unoptimized(q, vec_a, vec_b, unopt_sum, n), q, "Unoptimized");
    PrintTime(Optimized(q, vec_a, vec_b, opt_sum, n), q, "Optimized");

    // q's destructor invokes q's exception handler on any device exceptions.
  } catch (sycl::exception const &e) {
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

  // Check the results
  bool failed = false;
  if (unopt_sum != answer) {
    cout << "Unoptimized: expected: " << answer << ", result: " << unopt_sum
         << '\n';
    failed = true;
  }
  if (opt_sum != answer) {
    cout << "Optimized: expected: " << answer << ", result: " << opt_sum
         << '\n';
    failed = true;
  }

  if (failed) {
    cout << "FAILED\n";
    return 1;
  }
  cout << "PASSED\n";
  return 0;
}
