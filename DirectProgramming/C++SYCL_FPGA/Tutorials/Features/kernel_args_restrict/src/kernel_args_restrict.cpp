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
class KernelArgsRestrict_Lambda;
class KernelArgsNoRestrict_Lambda;
class KernelArgsRestrict_Functor;
class KernelArgsNoRestrict_Functor;

template <class InputAcc, class OutputAcc>
struct ArgsRestrictFunctor {
  InputAcc in;
  OutputAcc out;
  size_t len;

  [[intel::kernel_args_restrict]] 
  void operator()() const {
    for (int idx = 0; idx < len; ++idx) {
      out[idx] = in[idx];
    }
  };
};

template <class InputAcc, class OutputAcc>
struct ArgsNoRestrictFunctor {
  InputAcc in;
  OutputAcc out;
  size_t len;

  void operator()() const {
    for (int idx = 0; idx < len; ++idx) {
      out[idx] = in[idx];
    }
  };
};

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

void RunKernels(size_t size, std::vector<int> &in, std::vector<int> &nr_out_lambda,
                std::vector<int> &r_out_lambda, std::vector<int> &nr_out_functor,
                std::vector<int> &r_out_functor) {
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
    buffer nr_out_buf_lambda(nr_out_lambda);
    buffer r_out_buf_lambda(r_out_lambda);
    buffer nr_out_buf_functor(nr_out_functor);
    buffer r_out_buf_functor(r_out_functor);

    // submit the task that DOES NOT apply the kernel_args_restrict attribute
    auto e_nr_lambda = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(nr_out_buf_lambda, h, write_only, no_init);

      h.single_task<KernelArgsNoRestrict_Lambda>([=]() {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // submit the task that DOES apply the kernel_args_restrict attribute
    auto e_r_lambda = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(r_out_buf_lambda, h, write_only, no_init);

      h.single_task<KernelArgsRestrict_Lambda>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    auto e_nr_functor = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(nr_out_buf_functor, h, write_only, no_init);

      h.single_task<KernelArgsNoRestrict_Functor>(
        ArgsNoRestrictFunctor<decltype(in_acc), decltype(out_acc)>{in_acc, out_acc, size}
      );
    });

    auto e_r_functor = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(r_out_buf_functor, h, write_only, no_init);

      h.single_task<KernelArgsRestrict_Functor>(
        ArgsRestrictFunctor<decltype(in_acc), decltype(out_acc)>{in_acc, out_acc, size}
      );
    });

    // measure the execution time of each kernel
    double size_mb = (size * sizeof(int)) / (1024 * 1024);
    double nr_time_lambda = GetExecutionTime(e_nr_lambda);
    double r_time_lambda = GetExecutionTime(e_r_lambda);
    double nr_time_functor = GetExecutionTime(e_nr_functor);
    double r_time_functor = GetExecutionTime(e_r_functor);

    std::cout << "Size of vector: " << size << " elements\n";
    std::cout << "Lambda kernel throughput without attribute: " << (size_mb / nr_time_lambda)
              << " MB/s\n";
    std::cout << "Lambda kernel throughput with attribute: " << (size_mb / r_time_lambda)
              << " MB/s\n";
    std::cout << "Functor kernel throughput without attribute: " << (size_mb / nr_time_functor)
              << " MB/s\n";
    std::cout << "Functor kernel throughput with attribute: " << (size_mb / r_time_functor)
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
bool isEqual(std::vector<int> &in, std::vector<int> &out, std::string &&kernel_name, size_t size) {
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
  std::vector<int> nr_out_lambda(size), r_out_lambda(size);
  std::vector<int> nr_out_functor(size), r_out_functor(size);

  // generate some input data
  std::iota(in.begin(), in.end(), 0);

  // clear the output data
  std::fill(nr_out_lambda.begin(), nr_out_lambda.end(), -1);
  std::fill(r_out_lambda.begin(), r_out_lambda.end(), -1);
  std::fill(nr_out_functor.begin(), nr_out_functor.end(), -1);
  std::fill(r_out_functor.begin(), r_out_functor.end(), -1);

  // Run the kernels
  RunKernels(size, in, nr_out_lambda, r_out_lambda, nr_out_functor, r_out_functor);

  // validate the results
  bool passed = true;

  passed &= isEqual(in, nr_out_lambda, "KernelArgsNoRestrict_Lambda", size);
  passed &= isEqual(in, r_out_lambda, "KernelArgsRestrict_Lambda", size);
  passed &= isEqual(in, nr_out_functor, "KernelArgsNoRestrict_Functor", size);
  passed &= isEqual(in, r_out_functor, "KernelArgsRestrict_Functor", size);

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 0;
  }
}
