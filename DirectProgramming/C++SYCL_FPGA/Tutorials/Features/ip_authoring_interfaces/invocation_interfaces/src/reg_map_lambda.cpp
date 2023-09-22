// oneAPI headers
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using MyUInt5 = ac_int<5, false>;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LambdaRegMap;

/////////////////////////////////////////

void LambdaRegMapKernel(sycl::queue &q, int *input, int *output, MyUInt5 n) {
  // A kernel with a register map invocation interface can also independently
  // have streaming kernel arguments, when annotated by 'conduit' property.
  sycl::ext::oneapi::experimental::annotated_arg<
      MyUInt5, decltype(sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::conduit})>
      n_annotated = n;

  // Without passing a properties object argument, the compiler will infer a
  // register-mapped invocation interface.
  q.single_task<LambdaRegMap>([=] {
     // For annotated_arg of ac_int type, explicitly cast away the annotated_arg
     // to prevent compiler error when using methods or accessing members.
     for (MyUInt5 i = 0; i < ((MyUInt5)n_annotated).slc<5>(0); i++) {
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

    int *input = sycl::malloc_host<int>(count, q);
    int *lambda_register_map_out = sycl::malloc_host<int>(count, q);
    int *golden_out = sycl::malloc_host<int>(count, q);

    // test that mallocs did not return nullptr
    assert(input);
    assert(lambda_register_map_out);
    assert(golden_out);

    // create input and golden output data
    for (MyUInt5 i = 0; i < count; i++) {
      input[i] = rand() % 77;
      golden_out[i] = (int)(input[i] * (input[i] + 1));
      lambda_register_map_out[i] = 0;
    }

    // validation lambda
    auto validate = [](int *golden_out, int *lambda_register_map_out,
                       MyUInt5 count) {
      for (MyUInt5 i = 0; i < count; i++) {
        if (lambda_register_map_out[i] != golden_out[i]) {
          std::cout << "lambda_register_map_out[" << i << "] != golden_out["
                    << i << "]"
                    << " (" << lambda_register_map_out[i]
                    << " != " << golden_out[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with a register map invocation interface implemented in
    // the lambda programming model
    std::cout << "Running the kernel with register map invocation interface "
                 "implemented in the lambda programming model"
              << std::endl;
    LambdaRegMapKernel(q, input, lambda_register_map_out, count);
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
