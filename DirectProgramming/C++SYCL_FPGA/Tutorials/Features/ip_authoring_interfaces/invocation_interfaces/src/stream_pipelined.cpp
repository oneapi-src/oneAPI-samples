// oneAPI headers
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class StreamPipelined;

struct StreamPipelinedIP {
  // Kernel arguments will be passed as conduits since the invocation interface
  // is configured to be 'streaming', and no annotated_arg wrapper is used.
  int *input;
  int *output;

  // Kernel properties method to configure the kernel to be a kernel with
  // streaming pipelined invocation interface.
  // The property `sycl::ext::intel::experimental::pipelined` takes an optional
  // template parameter that controls whether to pipeline the kernel. Valid
  // parameters are: -1: Pipeline the kernel, and automatically infer lowest
  // possible II at target fMAX. 0: Do not pipeline the kernel. N (N> 0):
  // Pipeline the kernel, and force the II of the kernel to be N. If a parameter
  // is not specified, the default behaviour of -1 will be inferred.
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::streaming_interface<>,
        sycl::ext::intel::experimental::pipelined<>};
  }

  void operator()() const {
    int val = *input;
    *output = (int)(val * (val + 1));
  }
};

int main(int argc, char *argv[]) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  bool passed = true;
  int count = 16;
  if (argc > 1) count = atoi(argv[1]);

  if (count <= 0) {
    std::cerr << "ERROR: 'count' must be positive" << std::endl;
    return 1;
  }

  try {
    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    // make sure the device supports USM host allocations
    sycl::device d = q.get_device();

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    if (!d.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations" << std::endl;
      return 1;
    }

    int *input = sycl::malloc_host<int>(count, q);
    int *functor_streaming_pipelined_out = sycl::malloc_host<int>(count, q);
    int *golden_out = sycl::malloc_host<int>(count, q);

    // test that mallocs did not return nullptr
    assert(input);
    assert(functor_streaming_pipelined_out);
    assert(golden_out);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      input[i] = rand() % 77;
      golden_out[i] = (int)(input[i] * (input[i] + 1));
      functor_streaming_pipelined_out[i] = 0;
    }

    // validation lambda
    auto validate = [](auto *golden_out, auto *functor_streaming_pipelined_out,
                       int count) {
      for (int i = 0; i < count; i++) {
        if (functor_streaming_pipelined_out[i] != golden_out[i]) {
          std::cout << "functor_streaming_pipelined_out[" << i
                    << "] != golden_out[" << i << "]"
                    << " (" << functor_streaming_pipelined_out[i]
                    << " != " << golden_out[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    std::cout << "Launching streaming pipelined kernels consecutively"
              << std::endl;
    for (int i = 0; i < count; i++) {
      q.single_task<StreamPipelined>(
          StreamPipelinedIP{&input[i], &functor_streaming_pipelined_out[i]});
    }
    q.wait();
    std::cout << "\t Done" << std::endl;

    passed &= validate(golden_out, functor_streaming_pipelined_out, count);
    std::cout << std::endl;

    sycl::free(input, q);
    sycl::free(functor_streaming_pipelined_out, q);
    sycl::free(golden_out, q);
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}