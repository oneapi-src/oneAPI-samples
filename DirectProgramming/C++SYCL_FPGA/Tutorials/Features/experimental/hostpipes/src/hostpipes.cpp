#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "exception_handler.hpp"

// forward declare kernel and pipe names to reduce name mangling
class LoopBackKernelID;
class H2DPipeID;
class D2HPipeID;

// the host pipes
using ValueT = int;
constexpr size_t kPipeMinCapacity = 8;

using H2DPipe = sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    H2DPipeID,         // An identifier for the pipe
    ValueT,            // The type of data in the pipe
    kPipeMinCapacity   // The capacity of the pipe
    >;

using D2HPipe = sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    D2HPipeID,         // An identifier for the pipe
    ValueT,            // The type of data in the pipe
    kPipeMinCapacity   // The capacity of the pipe
    >;

// forward declare the test functions
void AlternatingTest(sycl::queue&, ValueT*, ValueT*, size_t, size_t);
void LaunchCollectTest(sycl::queue&, ValueT*, ValueT*, size_t, size_t);

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * sqrt(val)); }

/////////////////////////////////////////

int main(int argc, char* argv[]) {
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
  if (count < kPipeMinCapacity) {
    std::cerr
        << "ERROR: 'count' must be greater than the minimum pipe capacity ("
        << kPipeMinCapacity << ")" << std::endl;
    return 1;
  }

  try {
    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;


    // create input and golden output data
    std::vector<ValueT> in(count), out(count), golden(count);
    std::generate(in.begin(), in.end(), [] { return ValueT(rand() % 77); });
    for (int i = 0; i < count; i++) {
      golden[i] = SomethingComplicated(in[i]);
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

    // Alternating write-and-read
    std::cout << "Running Alternating write-and-read" << std::endl;
    std::fill(out.begin(), out.end(), 0);
    AlternatingTest(q, in.data(), out.data(), count, 3);
    passed &= validate(golden, out, count);
    std::cout << std::endl;

    // Launch and Collect
    std::cout << "Running Launch and Collect" << std::endl;
    std::fill(out.begin(), out.end(), 0);
    LaunchCollectTest(q, in.data(), out.data(), kPipeMinCapacity, 3);
    passed &= validate(out, golden, kPipeMinCapacity);
    std::cout << std::endl;

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

// This kernel reads a data element from InHostPipe, processes it,
// and writes the result to OutHostPipe
template <typename KernelId,    // type identifier for kernel
          typename InHostPipe,  // host-to-device pipe
          typename OutHostPipe  // device-to-host pipe
          >
sycl::event SubmitLoopBackKernel(sycl::queue& q, size_t count) {
  return q.single_task<KernelId>([=] {
    for (size_t i = 0; i < count; i++) {
      auto d = InHostPipe::read();
      auto r = SomethingComplicated(d);
      OutHostPipe::write(r);
    }
  });
}

// This test launches SubmitLoopBackKernel, then alternates writes
// and reads to and from the H2DPipe and D2HPipe hostpipes respectively
void AlternatingTest(sycl::queue& q, ValueT* in, ValueT* out, size_t count,
                     size_t repeats) {
  std::cout << "\t Run Loopback Kernel on FPGA" << std::endl;
  auto e = SubmitLoopBackKernel<LoopBackKernelID, H2DPipe, D2HPipe>(
      q, count * repeats);

  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": "
              << "Doing " << count << " writes & reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      // write data in host-to-device hostpipe
      H2DPipe::write(q, in[i]);
      // read data from device-to-host hostpipe
      out[i] = D2HPipe::read(q);
    }
  }

  // No need to wait on kernel to finish as the pipe reads are blocking

  std::cout << "\t Done" << std::endl;
}

// This test launches SubmitLoopBackKernel, writes 'count'
// elements to H2DPipe, and then reads 'count' elements from
// D2HPipe
void LaunchCollectTest(sycl::queue& q, ValueT* in, ValueT* out, size_t count,
                       size_t repeats) {
  std::cout << "\t Run Loopback Kernel on FPGA" << std::endl;

  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": "
              << "Doing " << count << " writes" << std::endl;
    for (size_t i = 0; i < count; i++) {
      // write data in host-to-device hostpipe
      H2DPipe::write(q, in[i]);
    }
  }

  auto e = SubmitLoopBackKernel<LoopBackKernelID, H2DPipe, D2HPipe>(
      q, count * repeats);

  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": "
              << "Doing " << count << " reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      // read data from device-to-host hostpipe
      out[i] = D2HPipe::read(q);
    }
  }

  // No need to wait on kernel to finish as the pipe reads are blocking

  std::cout << "\t Done" << std::endl;
}
