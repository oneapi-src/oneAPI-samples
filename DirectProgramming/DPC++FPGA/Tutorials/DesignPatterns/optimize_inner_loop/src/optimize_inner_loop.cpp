//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// allow the maximum random number value to be controlled from the command line
#ifndef RAND_RANGE_MAX
#define RAND_RANGE_MAX 3
#endif

//// constants
constexpr int kNumKernels = 3;
constexpr int kRandRangeMax = RAND_RANGE_MAX;
constexpr double kProbSuccess = 1.0 / kRandRangeMax;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
// Templating allows us to instantiate multiple versions of the kernel.
template <int version> class Producer;
template <int version> class Consumer;

// Declare the pipe class name globally to reduce name mangling.
// Templating allows us to instantiate multiple versions of pipes for each 
// version of the kernel.
template <int version> class PipeClass;

//
// Submits the kernel, which is templated on the variables:
//    version             - The version ID of the kernel
//    in_element_upper_bound - The upperbound (inclusive) on the elements of the
//                          'in' vector (a negative value implies no bound). In
//                          other words: if in_element_upper_bound >= 0, then
//                          in[i] <= in_element_upper_bound, for all elements
//                          of 'in'
//    spec_iters           - The number of speculated iterations to set for the
//                          inner loop
//
template <int version, int in_element_upper_bound, int spec_iters>
void SubmitKernels(const device_selector &selector, std::vector<int> &in,
                   int &res, double &kernel_time_ms) {
  // static asserts: these cause the compiler to fail if the conditions fail
  static_assert(version >= 0, "Invalid kernel version");
  static_assert(spec_iters >= 0, "spec_iters must be positive");

  // the pipe
  using Pipe = pipe<PipeClass<version>, bool>;

  kernel_time_ms = 0.0;
  int size = in.size();

  try {
    // create the device queue with profiling enabled
    auto prop_list = property_list{ property::queue::enable_profiling() };
    queue q(selector, dpc_common::exception_handler, prop_list);

    // The input data buffer
    buffer in_buf(in);

    // The output data buffer
    // Scalar inputs are passed to the kernel using the lambda capture,
    // but a SYCL buffer must be used to return a scalar from the kernel.
    buffer<int, 1> res_buf(&res, 1);

    // submit the Producer kernel
    event p_e = q.submit([&](handler &h) {
      // the input buffer accessor
      accessor in_a(in_buf, h, read_only);

      h.single_task<Producer<version>>([=]() [[intel::kernel_args_restrict]] {
        for (int i = 0; i < size; i++) {
          // read the input value, which is in the range [0,InnerLoopBound]
          int val = in_a[i];

          // 'in_element_upper_bound' is a constant (a template variable).
          // Therefore, the condition 'in_element_upper_bound < 0', and therefore
          // the taken branch of this if-else statement, can be determined at
          // compile time. This results in the branch that is NOT taken being
          // optimized away. Both versions of the inner loop apply the
          // speculated_iterations attribute, where the number of speculated
          // iterations is determined by the template variable 'spec_iters'.
          if (in_element_upper_bound < 0) {
            // In this version of the inner loop, we do NOT provide an
            // upperbound on the loop index variable 'j'. While it may be easy
            // for you to read the code and reason that 'j<in_element_upper_bound'
            // is always true by looking at the rest of the program, it is much
            // more difficult for the compiler. As a result, the compiler will
            // be conservative and assume this inner loop may have a large trip
            // count and decide to make (or not make) optimizations accordingly.
            [[intel::speculated_iterations(spec_iters)]]
            for (int j = 0; j < val; j++) {
              Pipe::write(true);
            }
          } else {
            // In this version of the inner loop, we provide an upper bound
            // on the loop index variable 'j' by adding the
            // 'j<in_element_upper_bound' loop exit condition. This provides the
            // compiler with a constant upperbound on the trip count and allows
            // it to make optimizations accordingly.
            [[intel::speculated_iterations(spec_iters)]]
            for (int j = 0; j < val && j <= in_element_upper_bound; j++) {
              Pipe::write(true);
            }
          }
        }

        // tell the consumer that we are done producing data
        Pipe::write(false);
      });
    });

    // submit the Consumer kernel
    event c_e = q.submit([&](handler &h) {
      // the output buffer accessor
      accessor res_a(res_buf, h, write_only, noinit);

      h.single_task<Consumer<version>>([=]() [[intel::kernel_args_restrict]] {
        // local register to accumulate into
        int local_sum = 0;

        // keep grabbing data from the Producer until it tells us to stop
        while (Pipe::read()) {
          local_sum++;
        }

        // copy back the result to global memory
        res_a[0] = local_sum;
      });
    });

    // get the kernel time in milliseconds
    // this excludes memory transfer and queuing overhead
    double startk =
        p_e.template get_profiling_info<info::event_profiling::command_start>();
    double endk =
        c_e.template get_profiling_info<info::event_profiling::command_end>();
    kernel_time_ms = (endk - startk) * 1e-6f;

  } catch (exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}

//
// main function
//
int main(int argc, char *argv[]) {
  // the device selector
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // set the input size based on whether we are in emulation or FPGA hardware
#if defined(FPGA_EMULATOR)
  int size = 5000;
#else
  int size = 5000000;
#endif

  // Allow the size to be changed by a command line argument
  if (argc > 1) {
    size = atoi(argv[1]);
  }

  // check that the size makes sense
  if (size <= 0) {
    std::cerr << "ERROR: 'size' must be strictly positive\n";
    return 1;
  }

  // generate random input data and compute golden result
  std::vector<int> in(size);
  int golden_result = 0;

  std::cout << "generating " << size << " random numbers in the range "
            << "[0," << kRandRangeMax << "]\n";

  // The random number generator (rng)
  std::default_random_engine rng;

  // A binomial distribution will generate random numbers in the range
  // [0,kRandRangeMax], where the expected value is kRandRangeMax*kProbSuccess.
  // We have set these constants such that the expected value is 1. This
  // means that the number of inner loop iterations in the Producer kernel
  // is in the range [0,kRandRangeMax], but is 1 on average. For more info see:
  //    https://en.cppreference.com/w/cpp/numeric/random/binomial_distribution
  std::binomial_distribution<int> bin_dist(kRandRangeMax, kProbSuccess);

  // generate the random input data
  std::generate(in.begin(), in.end(), [&] { return bin_dist(rng); });

  // compute the golden result
  golden_result = std::accumulate(in.begin(), in.end(), 0);

  // the result variables from the kernels
  std::array<int, kNumKernels> result;
  std::array<double, kNumKernels> ktime;

  // version 0
  //
  // For the inner loop, this version has the bounding of the inner loop
  // disabled (-1 for in_element_upper_bound disables inner loop bounding)
  // and sets 2 speculated iterations.
  std::cout << "Running kernel 0\n";
  SubmitKernels<0, -1, 2>(selector, in, result[0], ktime[0]);

  // version 1
  //
  // For the inner loop, this version has the bounding of the inner loop
  // disabled (-1 for in_element_upper_bound disables inner loop bounding)
  // and sets 0 speculated iterations.
  std::cout << "Running kernel 1\n";
  SubmitKernels<1, -1, 0>(selector, in, result[1], ktime[1]);

  // version 2
  //
  // For the inner loop, this version bounds the inner loop (the max value
  // generated by our RNG above, kRandRangeMax) and has 0 speculated iterations.
  std::cout << "Running kernel 2\n";
  SubmitKernels<2, kRandRangeMax, 0>(selector, in, result[2], ktime[2]);

  // validate the results
  bool success = true;
  for (int i = 0; i < kNumKernels; i++) {
    if (result[i] != golden_result) {
      std::cerr << "ERROR: Kernel " << i << " result mismatch: " << result[i]
                << " != " << golden_result << " (result != expected)\n";
      success = false;
    }
  }

  if (success) {
    // the emulator does not accurately represent real hardware performance.
    // Therefore, we don't show performance results when running in emulation.
#if !defined(FPGA_EMULATOR)
    double input_size_bytes = size * sizeof(int);

    // only display two decimal points
    std::cout << std::fixed << std::setprecision(2);

    // compute and print the performance results
    for (int i = 0; i < kNumKernels; i++) {
      std::cout << "Kernel " << i
                << " throughput: " << (input_size_bytes / ktime[i]) * 1e-3
                << " MB/s \n";
    }
#endif

    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}
