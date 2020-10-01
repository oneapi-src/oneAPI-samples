//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

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

// Artificial coefficient and offset data for our math function
constexpr size_t kSize = 64;
constexpr std::array<int, kSize> kCoeff = {
            1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
constexpr std::array<int, kSize> kOffset = {
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1};

// The function our kernel will compute
// The "golden result" will be computed on the host to check the kernel result.
vector<int> GoldenResult(vector<int> vec) {

  // The coefficients will be modified with each iteration of the outer loop.
  std::array coeff = kCoeff;

  for (int &val : vec) {
    // Do some arithmetic
    int acc = 0;
    for (size_t i = 0; i < kSize; i++) {
      acc += coeff[i] * (val + kOffset[i]);
    }

    // Update coeff by rotating the values of the array
    int tmp = coeff[0];
    for (size_t i = 0; i < kSize - 1; i++) {
      coeff[i] = coeff[i + 1];
    }
    coeff[kSize - 1] = tmp;

    // Result
    val = acc;
  }

  return vec;
}

// Forward declaration of the kernel name
// (This will become unnecessary in a future compiler version.)
class SimpleMath;

void RunKernel(const device_selector &selector,
               const std::vector<int> &vec_a,
               std::vector<int> &vec_r) {

  size_t input_size = vec_a.size();

  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer device_a(vec_a);
    // Use verbose SYCL 1.2 syntax for the output buffer.
    // (This will become unnecessary in a future compiler version.)
    buffer<int, 1> device_r(vec_r.data(), input_size);

    event e = q.submit([&](handler &h) {
      auto a = device_a.get_access<access::mode::read>(h);
      auto r = device_r.get_access<access::mode::discard_write>(h);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<class SimpleMath>([=]() [[intel::kernel_args_restrict]] {

        // Force the compiler to implement the coefficient array in FPGA
        // pipeline registers rather than in on-chip memory.
        [[intelfpga::register]] std::array coeff = kCoeff;

        // The compiler will pipeline the outer loop.
        for (size_t i = 0; i < input_size; ++i) {
          int acc = 0;
          int val = a[i];

          // Fully unroll the accumulator loop.
          // All of the unrolled operations can be freely scheduled by the
          // DPC++ compiler's FPGA backend as part of a common data pipeline.
          #pragma unroll
          for (size_t j = 0; j < kSize; j++) {
#ifdef USE_FPGA_REG
            // Use fpga_reg to insert a register between the copy of val used
            // in each unrolled iteration.
            val = INTEL::fpga_reg(val);
            // Since val is held constant across the kSize unrolled iterations,
            // the FPGA hardware structure of val's distribution changes from a
            // kSize-way fanout (without fpga_reg) to a chain of of registers
            // with intermediate tap offs. Refer to the diagram in the README.

            // Use fpga_reg to insert a register between each step in the acc
            // adder chain.
            acc = INTEL::fpga_reg(acc) + (coeff[j] * (val + kOffset[j]));
            // This transforms a compiler-inferred adder tree into an adder
            // chain, altering the structure of the pipeline. Refer to the
            // diagram in the README.
#else
            // Without fpga_reg, the compiler schedules the operations here
            // according to its default optimization heuristics.
            acc += (coeff[j] * (val + kOffset[j]));
#endif
          }

          // Rotate the values of the coefficient array.
          // The loop is fully unrolled. This is a cannonical code structure;
          // the DPC++ compiler's FPGA backend infers a shift register here.
          int tmp = coeff[0];
          #pragma unroll
          for (size_t j = 0; j < kSize - 1; j++) {
            coeff[j] = coeff[j + 1];
          }
          coeff[kSize - 1] = tmp;

          // Result
          r[i] = acc;
        }
      });
    });

    // Measure kernel execution time
    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();
    // Convert from nanoseconds to milliseconds.
    double kernel_time = (end - start) * 1e-6;

    // Kernel consists of two nested loops with 3 operations in the innermost
    // loop: 2 additions and 1 multiplication operation.
    size_t num_ops_per_kernel = input_size * kSize * 3;
    cout << "Throughput for kernel with input size " << input_size
         << " and coefficient array size " << kSize << ": ";
    cout << std::fixed << std::setprecision(6)
         << ((double)num_ops_per_kernel / kernel_time) / 1.0e6 << " GFlops\n";

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
}

int main(int argc, char *argv[]) {
  size_t input_size = 1e6;

  // Optional command line override of default input size
  if (argc > 1) {
    string option(argv[1]);
    if (option == "-h" || option == "--help") {
      cout << "Usage: \n<executable> <input data size>\n\nFAILED\n";
      return 1;
    } else {
      input_size = stoi(option);
    }
  }

  // Initialize input vector
  constexpr int max_val = 1<<10; // Conservative max to avoid integer overflow
  vector<int> vec_a(input_size);
  for (size_t i = 0; i < input_size; i++) {
    vec_a[i] = rand() % max_val;
  }
  // Kernel result vector
  vector<int> vec_r(input_size);

  // Run the kernel on either the FPGA emulator, or FPGA
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif
  RunKernel(selector, vec_a, vec_r);

  // Test the results.
  vector<int> golden_ref = GoldenResult(vec_a);
  bool correct = true;
  for (size_t i = 0; i < input_size; i++) {
    if (vec_r[i] != golden_ref[i]) {
      cout << "Found mismatch at " << i << ", "
           << vec_r[i] << " != " << golden_ref[i] << "\n";
      correct = false;
    }
  }

  if (correct) {
    cout << "PASSED: Results are correct.\n";
  } else {
    cout << "FAILED: Results are incorrect.\n";
    return 1;
  }

  return 0;
}
