// oneAPI headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include "exception_handler.hpp"

using ValueT = int;
using MyUInt5 = ac_int<5, false>;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LambdaRegisterMapIP;

// Create a properties object containing the
// kernel invocation interface property 'register_map'
sycl::ext::oneapi::experimental::properties kernel_properties{
  sycl::ext::intel::experimental::register_map
};

/////////////////////////////////////////

void TestLambdaRegisterMapKernel(sycl::queue &q, ValueT *input, ValueT *output,
                                 MyUInt5 n) {
  // In the Lambda programming model, all kernel arguments will have the same
  // interface as the kernel invocation interface.
  q.single_task<LambdaRegisterMapIP>(kernel_properties, [=] {
     for (MyUInt5 i = 0; i < n; i++) {
       output[i] = (ValueT)(input[i] * (input[i] + 1));
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

  MyUInt5 count = 16;
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

    ValueT *input = sycl::malloc_host<ValueT>(count, q);
    ValueT *lambda_register_map_out = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden_out = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (MyUInt5 i = 0; i < count; i++) {
      input[i] = rand() % 77;
      golden_out[i] = (ValueT)(input[i] * (input[i] + 1));
      lambda_register_map_out[i] = 0;
    }

    // validation lambda
    auto validate = [](ValueT *golden_out, ValueT *lambda_register_map_out, MyUInt5 count) {
      for (MyUInt5 i = 0; i < count; i++) {
        if (lambda_register_map_out[i] != golden_out[i]) {
          std::cout << "lambda_register_map_out[" << i << "] != golden_out[" << i << "]"
                    << " (" << lambda_register_map_out[i] << " != " << golden_out[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with a register map invocation interface implemented in
    // the lambda programming model
    std::cout << "Running kernel with a register map invocation interface "
                 "implemented in the "
                 "lambda programming model"
              << std::endl;
    TestLambdaRegisterMapKernel(q, input, lambda_register_map_out, count);
    passed &= validate(golden_out, lambda_register_map_out, count);
    std::cout << std::endl;

    sycl::free(input, q);
    sycl::free(lambda_register_map_out, q);
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
