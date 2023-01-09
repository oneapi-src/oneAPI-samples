//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "exception_handler.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

class Kernel;

constexpr int kInputSize = 1000;

typedef int RGBType;
typedef std::vector<RGBType> RGBVec;
typedef float GreyType;
typedef std::vector<GreyType> GreyVec;

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

GreyType Compute(RGBType r, RGBType g, RGBType b) {
  GreyType output =
      (GreyType)r * 0.3f + (GreyType)g * 0.59f + (GreyType)b * 0.11f;
  return output;
}

void RunKernel(const RGBVec &r, const RGBVec &g, const RGBVec &b,
               GreyVec &out) {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#elif defined(FPGA_SIMULATOR)
  ext::intel::fpga_simulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  try {
    // create the SYCL device queue
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    buffer r_buf(r);
    buffer g_buf(g);
    buffer b_buf(b);
    buffer out_buf(out);

    // submit the kernel
    auto e = q.submit([&](handler &h) {
      accessor r_acc(r_buf, h, read_only);
      accessor g_acc(g_buf, h, read_only);
      accessor b_acc(b_buf, h, read_only);
      accessor out_acc(out_buf, h, write_only, no_init);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]] {
#if defined(MANUAL_REVERT)
#if defined(S10)
        [[intel::speculated_iterations(4)]]
#endif
#endif
        for (size_t i = 0; i < kInputSize; i++) {
          out_acc[i] = Compute(r_acc[i], g_acc[i], b_acc[i]);
        }
      });
    });

    double exec_time = GetExecutionTime(e);
    double inputMB = (kInputSize * sizeof(RGBType)) / (double)(1024 * 1024);

    std::cout << "Kernel Throughput: " << (inputMB / exec_time) << "MB/s\n";
    std::cout << "Exec Time: " << exec_time << "s, InputMB: " << inputMB
              << "MB\n";

  } catch (exception const &e) {
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
}

int main() {
  // input/output data
  RGBVec r(kInputSize);
  RGBVec g(kInputSize);
  RGBVec b(kInputSize);
  GreyVec out(kInputSize);

  // generate random input data
  srand(0);
  for (size_t i = 0; i < kInputSize; i++) {
    r[i] = static_cast<RGBType>(rand() % 256);
    g[i] = static_cast<RGBType>(rand() % 256);
    b[i] = static_cast<RGBType>(rand() % 256);
  }

  RunKernel(r, g, b, out);

  // validate results
  for (size_t i = 0; i < kInputSize; i++) {
    GreyType golden = Compute(r[i], g[i], b[i]);
    if (std::fabs(out[i] - golden) > 1e-4) {
      std::cout << "FAILED: result mismatch\n"
                << "out[" << i << "] = " << out[i] << "; golden = " << golden
                << '\n';
      return 1;
    }
  }

  std::cout << "PASSED\n";
  return 0;
}
