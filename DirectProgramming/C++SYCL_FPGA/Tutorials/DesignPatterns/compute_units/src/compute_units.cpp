#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>

#include "exception_handler.hpp"
#include "compute_units.hpp"
#include "pipe_utils.hpp" // Included from DirectProgramming/C++SYCL_FPGA/include/


using namespace sycl;

constexpr float kTestData = 555;
constexpr size_t kEngines = 5;

using Pipes = fpga_tools::PipeArray<class MyPipe, float, 1, kEngines + 1>;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class Source;
class Sink;
template <std::size_t ID> class ChainComputeUnit;

// Write the data into the chain
void SourceKernel(queue &q, float data) {
  q.single_task<Source>([=] { Pipes::PipeAt<0>::write(data); });
}

// Get the data out of the chain and return it to the host
void SinkKernel(queue &q, float &out_data) {

  // The verbose buffer syntax is necessary here,
  // since out_data is just a single scalar value
  // and its size can not be inferred automatically
  buffer<float, 1> out_buf(&out_data, 1);

  q.submit([&](handler &h) {
    accessor out_accessor(out_buf, h, write_only, no_init);
    h.single_task<Sink>([=] {
      out_accessor[0] = Pipes::PipeAt<kEngines>::read();
    });
  });
}

int main() {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  float out_data = 0;

  try {
    queue q(selector, fpga_tools::exception_handler);

    sycl::device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Enqueue the Source kernel
    SourceKernel(q, kTestData);

    // Enqueue the chain of kEngines compute units
    // Compute unit must take a single argument, its ID
    SubmitComputeUnits<kEngines, ChainComputeUnit>(q, [=](auto ID) {
      auto f = Pipes::PipeAt<ID>::read();
      // Pass the data to the next compute unit in the chain
      // The compute unit with ID k reads from pipe k and writes to pipe
      // k + 1
      Pipes::PipeAt<ID + 1>::write(f);
    });

    // Enqueue the Sink kernel
    SinkKernel(q, out_data);

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

  // Verify result
  if (out_data != kTestData) {
    std::cout << "FAILED: The results are incorrect\n";
    std::cout << "Expected: " << kTestData << " Got: " << out_data << "\n";
    return 1;
  }

  std::cout << "PASSED: The results are correct\n";
  return 0;
}
