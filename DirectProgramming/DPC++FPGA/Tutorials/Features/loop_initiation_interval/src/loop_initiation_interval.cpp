//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <vector>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// short initialization loop trip count
constexpr size_t kInitLoopSize = 10;
// long-running loop trip count
constexpr size_t kLongLoopSize = 10000;
// problem input size
constexpr size_t kInputSize = 1000000;

// Forward declare the kernel name in the global scope
// This FPGA best practice reduces name mangling in the optimization reports
class SimpleMath;

int SomethingComplicated(int x) { return (int)sycl::sqrt((float)x); }

// The function the kernel will compute
// The golden result will be computed on the host to check the kernel result
int GoldenResult(int num) {
  for (size_t i = 0; i < kInitLoopSize; i++) {
    num += 1;
    num ^= 1;
    num += 1;
    num ^= 1;
    num += 1;
    num ^= 1;
    num += 1;
    num ^= 1;
    num += 1;
    num ^= 1;
  }

  int sum = 0;
  for (size_t j = 0; j < kLongLoopSize; j++) {
    sum += SomethingComplicated(num + j);
  }

  return sum;
}

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9;  // ns to s
  return kernel_time;
}

void RunKernel(std::vector<int> &in, std::vector<int> &out) {
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  try {
    // create the SYCL device queue
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer in_buf(in);
    buffer out_buf(out);

    // submit the kernel
    auto e = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(out_buf, h, write_only, noinit);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<SimpleMath>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < kInputSize; i++) {
          int num = in_acc[i];

          // All kernels share a common clock domain, thus this design needs to
          // be compiled twice to showcase the same design with different fMAX
          // If ENABLE_II is defined, the intel::initiation_interval attribute will be set for
          // the next loop, the short initialization loop Explicitly setting the
          // II for a loop will tell the compiler to schedule the loop while
          // enforcing the set II, overriding the default heuristic of finding
          // the minimum II * (1/fMAX) Relaxing the II on a short loop with a
          // long feedback path will remove the bottleneck the loop had on the
          // maximum achievable fMAX of the design The default targeted fMAX is
          // 240MHz for Arria 10 and 480MHz for Stratix 10, so different IIs
          // need to be specified so the compiler can schedule the loop such
          // that it does not restrict the maximum fMAX
#if defined(ENABLE_II)
#if defined(A10)
          [[intel::initiation_interval(3)]]
#elif defined(S10)
          [[intel::initiation_interval(5)]]
#elif defined(Agilex)
          [[intel::initiation_interval(5)]]
#else
          static_assert(false, "Unknown FPGA Architecture!");
#endif
#endif
          // ---------------------------
          // Short initialization loop
          // ---------------------------
          // The variable "num" has a loop carried dependency with a long
          // feedback path: The first operation on "num" in a given iteration
          // depends on the value of "num" calculated in the last operation of a
          // previous iteration. This leads to a classic fMAX-II tradeoff
          // situation. The compiler can achieve an II of 1 and a low fMAX, or
          // it can pipeline the arithmetic logic to improve fMAX at the expense
          // of II. By default the compiler will select an II of 1 after
          // optimizing for minimum II * (1/fMAX), which is not optimal for
          // whole design as fMAX is a system-wide constraint and this loop has
          // few iterations.
          for (size_t j = 0; j < kInitLoopSize; j++) {
            num += 1;
            num ^= 1;
            num += 1;
            num ^= 1;
            num += 1;
            num ^= 1;
            num += 1;
            num ^= 1;
            num += 1;
            num ^= 1;
          }

          int sum = 0;

          // The intel::initiation_interval attribute is added here to "assert" that II=1 for
          // this loop. Even though we fully expect the compiler to achieve
          // II=1 here by default, some developers find it helful to include
          // the attribute to "document" this expectation. If a future code
          // change causes an unexpected II regression, the compiler will error
          // out. Without the intel::initiation_interval attribute, an II regression may go
          // unnoticed.
#if defined(ENABLE_II)
          [[intel::initiation_interval(1)]]
#endif
          // ---------------------------
          // Long running loop
          // ---------------------------
          // The variable "sum" has a loop carried dependency with a feedback
          // path, although not as long as the feedback path of "num". The first
          // operation on "sum" in a given iteration depends on the value of
          // "sum" calculated in the last operation of a previous iteration The
          // compiler is able to achieve an II of 1 and the default targeted
          // fMAX for Arria 10, but falls a little short on Stratix 10. The
          // intel::initiation_interval attribute should not be used to relax the II of this loop
          // as the drop in occupancy of the long loop is not worth achieving a
          // slightly higher fMAX
          for (size_t k = 0; k < kLongLoopSize; k++) {
            sum += SomethingComplicated(num + k);
          }

          out_acc[i] = sum;
        }
      });
    });

    double exec_time = GetExecutionTime(e);
    double inputMB = (kInputSize * sizeof(int)) / (double)(1024 * 1024);

#if defined(ENABLE_II)
    std::cout << "Kernel_ENABLE_II Throughput: " << (inputMB / exec_time)
              << "MB/s\n";
#else
    std::cout << "Kernel Throughput: " << (inputMB / exec_time) << "MB/s\n";
#endif
    std::cout << "Exec Time: " << exec_time << "s , InputMB: " << inputMB
              << "MB\n";

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

int main() {
  // seed random number generator
  srand(0);

  // input/output data
  std::vector<int> in(kInputSize);
  std::vector<int> out(kInputSize);

  // Conservative max to avoid addition overflow
  constexpr int kRandMax = 1 << 10;
  // generate random input data
  for (size_t i = 0; i < kInputSize; i++) {
    in[i] = rand() % kRandMax;
  }

  // Run kernel once. Since fMAX is a global constraint, we cannot run two
  // kernels demonstrating the use of the intel::initiation_interval attribute since the kernel
  // without the intel::initiation_interval attribute would restrict the global fMAX, thus
  // affecting the design with the intel::initiation_interval attribute. Rely on the preprocessor
  // defines to change the kernel behaviour.
  RunKernel(in, out);

  // validate results
  for (size_t i = 0; i < kInputSize; i++) {
    if (out[i] != GoldenResult(in[i])) {
      std::cout << "FAILED: mismatch at entry " << i
                << " of 'SimpleMath' kernel output\n";
      return 1;
    }
  }

  std::cout << "PASSED\n";

  return 0;
}
