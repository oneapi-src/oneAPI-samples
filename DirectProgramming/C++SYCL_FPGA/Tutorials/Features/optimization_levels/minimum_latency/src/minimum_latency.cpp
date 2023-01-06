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

constexpr int kLoopSize = 50;
constexpr int kInputSize = 1000;

typedef long WorkType;
typedef std::vector<WorkType> WorkVec;

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

WorkType Compute(WorkType a, WorkType b) {
#pragma unroll
  for (size_t j = 0; j < kLoopSize; j++) {
    if (j & 1)
      a -= b;
    else
      a *= b;
  }
  return a;
}

void RunKernel(const WorkVec &a, const WorkVec &b, WorkVec &out) {
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

    buffer a_buf(a);
    buffer b_buf(b);
    buffer out_buf(out);

    // submit the kernel
    auto e = q.submit([&](handler &h) {
      accessor a_acc(a_buf, h, read_only);
      accessor b_acc(b_buf, h, read_only);
      accessor out_acc(out_buf, h, write_only, no_init);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < kInputSize; i += 2) {
          out_acc[i] = Compute(a_acc[i], b_acc[i]);
          out_acc[i + 1] = Compute(b_acc[i], a_acc[i]);
        }
      });
    });

    double exec_time = GetExecutionTime(e);
    double inputMB = (kInputSize * sizeof(WorkType)) / (double)(1024 * 1024);

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
  std::vector<WorkType> a(kInputSize);
  std::vector<WorkType> b(kInputSize);
  std::vector<WorkType> out(kInputSize);
  std::vector<WorkType> golden(kInputSize);

  // generate random input data
  srand(0);
  for (size_t i = 0; i < kInputSize; i += 2) {
    a[i] = static_cast<WorkType>(rand());
    b[i] = static_cast<WorkType>(rand());
    golden[i] = Compute(a[i], b[i]);
    golden[i + 1] = Compute(b[i], a[i]);
  }

  RunKernel(a, b, out);

  // validate results
  for (size_t i = 0; i < kInputSize; i++) {
    if (out[i] != golden[i]) {
      std::cout << "FAILED: result mismatch\n"
                << "out[" << i << "] = " << out[i] << "; golden[" << i
                << "] = " << golden[i] << '\n';
      return 1;
    }
  }

  std::cout << "PASSED\n";
  return 0;
}
