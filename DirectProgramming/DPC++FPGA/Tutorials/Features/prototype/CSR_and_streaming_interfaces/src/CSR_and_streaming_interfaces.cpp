#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

//
// Selects a SYCL device using a string. This is typically used to select
// the FPGA simulator device
//
class select_by_string : public sycl::default_selector {
public:
  select_by_string(std::string s) : target_name(s) {}
  virtual int operator()(const sycl::device& device) const {
    std::string name = device.get_info<sycl::info::device::name>();
    if (name.find(target_name) != std::string::npos) {
      // The returned value represents a priority, this number is chosen to be
      // large to ensure high priority
      return 10000;
    }
    return -1;
  }

private:
  std::string target_name;
};

using ValueT = int;
// forward declare the test functions
template <typename KernelType>
void KernelTest(sycl::queue&, ValueT*, ValueT*, size_t);

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * (val + 1)); }

/////////////////////////////////////////

struct StreamingControlIP {
  // Use the 'conduit' annotation on a kernel argument to specify it to be
  // a streaming kernel argument.
  conduit ValueT *input;
  conduit ValueT *output;
  // Without the annotations, kernel arguments will be inferred to be streaming
  // kernel arguments if the kernel control interface is streaming, and vise-versa.
  size_t n;
  StreamingControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  // Use the 'streaming_interface' annotation on a kernel to specify it to be
  // a kernel with streaming kernel control signals.
  streaming_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};

struct CSRAgentControlIP {
  // Use the 'register_map' annotation on a kernel argument to specify it to be
  // a CSR Agent kernel argument.
  register_map ValueT *input;
  // Without the annotations, kernel arguments will be inferred to be CSR Agent 
  // kernel arguments if the kernel control interface is CSR Agent, and vise-versa.
  ValueT *output;
  // A kernel with CSR Agent control can also independently have streaming kernel 
  // arguments, when annotated by 'conduit'.
  conduit size_t n;
  CSRAgentControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  register_map_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};

int main(int argc, char* argv[]) {
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector selector;
#elif defined(FPGA_SIMULATOR)
  std::string simulator_device_string =
      "SimulatorDevice : Multi-process Simulator (aclmsim0)";
  select_by_string selector = select_by_string{simulator_device_string};
#else
  sycl::ext::intel::fpga_selector selector;
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
    sycl::queue q(selector, dpc_common::exception_handler);

    // make sure the device supports USM device allocations
    sycl::device d = q.get_device();
    if (!d.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations" << std::endl;
      return 1;
    }

    ValueT *in = sycl::malloc_host<ValueT>(count, q);
    ValueT *streamingOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *CSRAgentOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      in[i] = rand() % 77;
      golden[i] = SomethingComplicated(in[i]);
      streamingOut[i] = 0;
      CSRAgentOut[i] = 0;
    }

    // validation lambda
    auto validate = [](auto& in, auto& out, size_t size) {
      for (int i = 0; i < size; i++) {
        if (out[i] != in[i]) {
          std::cout << "out[" << i << "] != in[" << i << "]"
                    << " (" << out[i] << " != " << in[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with streaming control
    std::cout << "Running kernel with streaming control" << std::endl;
    KernelTest<StreamingControlIP>(q, in, streamingOut, count);
    passed &= validate(golden, streamingOut, count);
    std::cout << std::endl;

    // Launch the kernel with CSR Agnet control
    std::cout << "Running kernel with CSR Agnet control" << std::endl;
    KernelTest<CSRAgentControlIP>(q, in, CSRAgentOut, count);
    passed &= validate(golden, CSRAgentOut, count);
    std::cout << std::endl;

    sycl::free(in, q);
    sycl::free(streamingOut, q);
    sycl::free(CSRAgentOut, q);
    sycl::free(golden, q);
  } catch (sycl::exception const& e) {
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

template <typename KernelType>
void KernelTest(sycl::queue& q, ValueT* in, ValueT* out, size_t count) {
  q.single_task(KernelType{in, out, count}).wait();

  std::cout << "\t Done" << std::endl;
}
