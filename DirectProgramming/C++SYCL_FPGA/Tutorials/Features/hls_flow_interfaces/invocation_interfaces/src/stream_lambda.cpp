// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LambdaStream;

/////////////////////////////////////////

void LambdaStreamKernel(sycl::queue &q, int *input, int *output, int n) {
  // Create a properties object containing the kernel invocation interface
  // property 'streaming_interface_remove_downstream_stall'.
  sycl::ext::oneapi::experimental::properties kernel_properties{
      sycl::ext::intel::experimental::
          streaming_interface_remove_downstream_stall};

  // In the Lambda programming model, pass a properties object argument to
  // configure the kernel invocation interface. All kernel arguments will have
  // the same interface as the kernel invocation interface.
  q.single_task<LambdaStream>(kernel_properties, [=] {
     for (int i = 0; i < n; i++) {
       output[i] = input[i] * (input[i] + 1);
     }
   }).wait();

  std::cout << "\t Done" << std::endl;
}

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
    int *lambda_streaming_out = sycl::malloc_host<int>(count, q);
    int *golden_out = sycl::malloc_host<int>(count, q);

    // test that mallocs did not return nullptr
    assert(input);
    assert(lambda_streaming_out);
    assert(golden_out);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      input[i] = rand() % 77;
      golden_out[i] = (int)(input[i] * (input[i] + 1));
      lambda_streaming_out[i] = 0;
    }

    // validation lambda
    auto validate = [](int *golden_out, int *lambda_streaming_out, int count) {
      for (int i = 0; i < count; i++) {
        if (lambda_streaming_out[i] != golden_out[i]) {
          std::cout << "lambda_streaming_out[" << i << "] != golden_out[" << i
                    << "]"
                    << " (" << lambda_streaming_out[i]
                    << " != " << golden_out[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with a streaming invocation interface implemented in
    // the lambda programming model
    std::cout << "Running the kernel with streaming invocation interface "
                 "implemented in the "
                 "lambda programming model"
              << std::endl;
    LambdaStreamKernel(q, input, lambda_streaming_out, count);
    passed &= validate(golden_out, lambda_streaming_out, count);
    std::cout << std::endl;

    sycl::free(input, q);
    sycl::free(lambda_streaming_out, q);
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
