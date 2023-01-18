//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "exception_handler.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

class Kernel;

#if defined(FPGA_SIMULATOR)
constexpr int kInputSize = 10;
#else
constexpr int kInputSize = 1000;
#endif

typedef int RGBType;
typedef std::vector<RGBType> RGBVec;
typedef float GreyType;
typedef std::vector<GreyType> GreyVec;

// Return the execution time of the event, in seconds
double GetExecutionTime(const sycl::event &e) {
  double start_k =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  double end_k =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
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
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // create the SYCL device queue
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    sycl::device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    sycl::buffer r_buf(r);
    sycl::buffer g_buf(g);
    sycl::buffer b_buf(b);
    sycl::buffer out_buf(out);

    // submit the kernel
    auto e = q.submit([&](sycl::handler &h) {
      sycl::accessor r_acc(r_buf, h, sycl::read_only);
      sycl::accessor g_acc(g_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor out_acc(out_buf, h, sycl::write_only, sycl::no_init);

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

  bool passed = true;

  // validate results
  for (size_t i = 0; i < kInputSize; i++) {
    GreyType golden = Compute(r[i], g[i], b[i]);
    if (std::fabs(out[i] - golden) > 1e-4) {
      std::cout << "Result mismatch:\n"
                << "out[" << i << "] = " << out[i] << "; golden = " << golden
                << '\n';
      passed = false;
    }
  }

  if (passed) {
    std::cout << "PASSED: all kernel results are correct\n";
  } else {
    std::cout << "FAILED\n";
  }
  return passed ? 0 : 1;
}
