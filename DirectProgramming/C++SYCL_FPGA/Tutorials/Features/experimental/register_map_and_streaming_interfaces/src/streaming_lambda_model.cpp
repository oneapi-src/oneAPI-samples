#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using ValueT = int;
// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LambdaStreamingIP;

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * (val + 1)); }

/////////////////////////////////////////

void TestLambdaStreamingKernel(sycl::queue &q, ValueT *in, ValueT *out,
                               size_t count) {
  // In the Lambda programming model, all kernel arguments will have the same
  // interface as the kernel invocation interface.
  q.single_task<LambdaStreamingIP>([=] streaming_interface {
     for (int i = 0; i < count; i++) {
       out[i] = SomethingComplicated(in[i]);
     }
   })
      .wait();

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

  size_t count = 16;
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

    ValueT *in = sycl::malloc_host<ValueT>(count, q);
    ValueT *lambda_streaming_out = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      in[i] = rand() % 77;
      golden[i] = SomethingComplicated(in[i]);
      lambda_streaming_out[i] = 0;
    }

    // validation lambda
    auto validate = [](auto &in, auto &out, size_t size) {
      for (int i = 0; i < size; i++) {
        if (out[i] != in[i]) {
          std::cout << "out[" << i << "] != in[" << i << "]"
                    << " (" << out[i] << " != " << in[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with a streaming invocation interface implemented in
    // the lambda programming model
    std::cout << "Running kernel with a streaming invocation interface "
                 "implemented in the "
                 "lambda programming model"
              << std::endl;
    TestLambdaStreamingKernel(q, in, lambda_streaming_out, count);
    passed &= validate(golden, lambda_streaming_out, count);
    std::cout << std::endl;

    sycl::free(in, q);
    sycl::free(lambda_streaming_out, q);
    sycl::free(golden, q);
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
