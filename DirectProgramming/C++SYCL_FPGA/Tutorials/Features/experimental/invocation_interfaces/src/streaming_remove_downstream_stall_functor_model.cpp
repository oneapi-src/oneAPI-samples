// oneAPI headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include "exception_handler.hpp"

using ValueT = int;
using MyUInt5 = ac_int<5, false>;

/////////////////////////////////////////

struct FunctorStreamingRmDownstreamStallIP {
  // Annotate kernel argument with 'conduit' property 
  // to specify it to be a streaming kernel argument.
  sycl::ext::oneapi::experimental::annotated_arg<
      ValueT *, decltype(sycl::ext::oneapi::experimental::properties{
                    sycl::ext::intel::experimental::conduit})>                    
      input;

  // A kernel with a streaming invocation interface can also independently
  // have register map kernel arguments, when annotated by 'register_map' property.
  sycl::ext::oneapi::experimental::annotated_arg<
      ValueT *, decltype(sycl::ext::oneapi::experimental::properties{
                    sycl::ext::intel::experimental::register_map})>                    
      output;

  // Without the annotation, kernel argument will be inferred to be streaming
  // kernel arguments if the kernel invocation interface is streaming, and
  // vice-versa.
  MyUInt5 n;

  // Kernel properties method to configure the kernel to be a kernel with 
  // streaming pipelined invocation interface without downstream 'ready_in' interface
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::streaming_interface_remove_downstream_stall,
        sycl::ext::intel::experimental::pipelined<>};
  }

  void operator()() const {
    for (MyUInt5 i = 0; i < n; i++) {
      output[i] = (ValueT)(input[i] * (input[i] + 1));
    }
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
    ValueT *functor_streaming_rm_downstream_stall_out = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden_out = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      input[i] = rand() % 77;
      golden_out[i] = (ValueT)(input[i] * (input[i] + 1));
      functor_streaming_rm_downstream_stall_out[i] = 0;
    }

    // validation lambda
    auto validate = [](auto *golden_out, auto *functor_streaming_rm_downstream_stall_out, MyUInt5 count) {
      for (int i = 0; i < count; i++) {
        if (functor_streaming_rm_downstream_stall_out[i] != golden_out[i]) {
          std::cout << "functor_streaming_rm_downstream_stall_out[" << i << "] != golden_out[" << i << "]"
                    << " (" << functor_streaming_rm_downstream_stall_out[i] << " != " << golden_out[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with a streaming invocation interface implemented in
    // the functor programming model
    std::cout << "Running the kernel with a streaming invocation interface "
                 "implemented in the "
                 "functor programming model"
              << std::endl;
    q.single_task(FunctorStreamingRmDownstreamStallIP{input, functor_streaming_rm_downstream_stall_out, count})
        .wait();
    std::cout << "\t Done" << std::endl;

    passed &= validate(golden_out, functor_streaming_rm_downstream_stall_out, count);
    std::cout << std::endl;

    sycl::free(input, q);
    sycl::free(functor_streaming_rm_downstream_stall_out, q);
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
