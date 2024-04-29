//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <numeric>
#include <vector>

#include "exception_handler.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class IDKernelArgsRestrict_Lambda;
class IDConservative_Lambda;
class IDKernelArgsRestrict_Functor;
class IDConservative_Functor;

template <class InputAcc, class OutputAcc>
struct ArgsRestrictFunctor {
  InputAcc in;
  OutputAcc out;
  size_t len;

  [[intel::kernel_args_restrict]]  // NO-FORMAT: Attribute
  void operator()() const {
    for (int idx = 0; idx < len; ++idx) {
      out[idx] = in[idx];
    }
  }
};

template <class InputAcc, class OutputAcc>
struct ConservativeFunctor {
  InputAcc in;
  OutputAcc out;
  size_t len;

  void operator()() const {
    for (int idx = 0; idx < len; ++idx) {
      out[idx] = in[idx];
    }
  }
};

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

void RunKernels(size_t size, std::vector<int> &in, std::vector<int> &conservative_lambda_out,
                std::vector<int> &restrict_lambda_out, std::vector<int> &conservative_functor_out,
                std::vector<int> &restrict_functor_out) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // create the SYCL device queue
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});
    
    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    buffer in_buf(in);

    // Below, we submit two SYCL kernels defined in lambda coding style.
    // submit the task that DOES NOT apply the kernel_args_restrict attribute
    buffer conservative_lambda_out_buf(conservative_lambda_out);
    auto e_conservative_lambda = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(conservative_lambda_out_buf, h, write_only, no_init);

      h.single_task<IDConservative_Lambda>([=]() {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // submit the task that DOES apply the kernel_args_restrict attribute
    buffer restrict_lambda_out_buf(restrict_lambda_out);
    auto e_restrict_lambda = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(restrict_lambda_out_buf, h, write_only, no_init);

      h.single_task<IDKernelArgsRestrict_Lambda>([=  // NO-FORMAT: Attribute
      ]() [[intel::kernel_args_restrict]] {          // NO-FORMAT: Attribute
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // Below, we submit two SYCL kernels defined in functor coding style.
    // submit the task that DOES NOT apply the kernel_args_restrict attribute
    buffer conservative_functor_out_buf(conservative_functor_out);
    auto e_conservative_functor = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(conservative_functor_out_buf, h, write_only, no_init);

      h.single_task<IDConservative_Functor>(
        ConservativeFunctor<decltype(in_acc), decltype(out_acc)>{in_acc, out_acc, size}
      );
    });

    // submit the task that DOES apply the kernel_args_restrict attribute
    buffer restrict_functor_out_buf(restrict_functor_out);
    auto e_restrict_functor = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(restrict_functor_out_buf, h, write_only, no_init);

      h.single_task<IDKernelArgsRestrict_Functor>(
        ArgsRestrictFunctor<decltype(in_acc), decltype(out_acc)>{in_acc, out_acc, size}
      );
    });

    // measure the execution time of each kernel
    double size_mb = (size * sizeof(int)) / (1024 * 1024);
    double conservative_lambda_time = GetExecutionTime(e_conservative_lambda);
    double restrict_lambda_time = GetExecutionTime(e_restrict_lambda);
    double conservative_functor_time = GetExecutionTime(e_conservative_functor);
    double restrict_functor_time = GetExecutionTime(e_restrict_functor);

    std::cout << "Size of vector: " << size << " elements\n";
    std::cout << "Lambda kernel throughput without attribute: " << (size_mb / conservative_lambda_time)
              << " MB/s\n";
    std::cout << "Lambda kernel throughput with attribute: " << (size_mb / restrict_lambda_time)
              << " MB/s\n";
    std::cout << "Functor kernel throughput without attribute: " << (size_mb / conservative_functor_time)
              << " MB/s\n";
    std::cout << "Functor kernel throughput with attribute: " << (size_mb / restrict_functor_time)
              << " MB/s\n";

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

  // Exiting the 'try' scope above, where the buffers were declared, will cause 
  // the buffer destructors to be called which will wait until the kernels that
  // use them to finish and copy the data back to the host (if the buffer was
  // written to).
  // Therefore, at this point in the code, we know that the kernels have
  // finished and the data has been transferred back to the host (in the
  // 'nr_out' and 'r_out' vectors).
}

// Checks if the content of two vectors are equal up to a specified size.
// The function prints a message indicating the index at which the mismatch occurred, 
// and the name of the kernel of which the result is checked, specified by `kernel_name`.
bool IsEqual(std::vector<int> &in, std::vector<int> &out, std::string &&kernel_name, size_t size) {
  bool equal {true};
  for (size_t i = 0; i < size; i++) {
    if (in[i] != out[i]) {
      std::cout << "FAILED: mismatch at entry " << i
                << " of " << kernel_name
                << " kernel output\n";
      std::cout << " (" << in[i] << " != " << out[i]
                << ", in[i] != out[i])\n";
      equal = false;
      break;
    }
  }
  return equal;
}

int main(int argc, char* argv[]) {
  // size of vectors to copy, allow user to change it from the command line
#if defined(FPGA_SIMULATOR)
  size_t size = 5000; // smaller size to keep the default runtime reasonable
#else
  size_t size = 5000000;
#endif

  if (argc > 1) size = atoi(argv[1]);

  // input/output data
  std::vector<int> in(size);
  std::vector<int> conservative_lambda_out(size), restrict_lambda_out(size);
  std::vector<int> conservative_functor_out(size), restrict_functor_out(size);

  // generate some input data
  std::iota(in.begin(), in.end(), 0);

  // clear the output data
  std::fill(conservative_lambda_out.begin(), conservative_lambda_out.end(), -1);
  std::fill(restrict_lambda_out.begin(), restrict_lambda_out.end(), -1);
  std::fill(conservative_functor_out.begin(), conservative_functor_out.end(), -1);
  std::fill(restrict_functor_out.begin(), restrict_functor_out.end(), -1);

  // Run the kernels
  RunKernels(size, in, conservative_lambda_out, restrict_lambda_out, conservative_functor_out, restrict_functor_out);

  // validate the results
  bool passed = true;

  passed &= IsEqual(in, conservative_lambda_out, "IDConservative_Lambda", size);
  passed &= IsEqual(in, restrict_lambda_out, "IDKernelArgsRestrict_Lambda", size);
  passed &= IsEqual(in, conservative_functor_out, "IDConservative_Functor", size);
  passed &= IsEqual(in, restrict_functor_out, "IDKernelArgsRestrict_Functor", size);

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 0;
  }
}
