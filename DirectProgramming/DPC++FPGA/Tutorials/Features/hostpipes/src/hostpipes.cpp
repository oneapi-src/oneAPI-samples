include <CL/sycl.hpp>
#include <algorithm>
#include <host_pipes.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std::chrono;

// forward declare kernel and pipe names to reduce name mangling
class LoopBackKernelID;
class H2DPipeID;
class D2HPipeID;

// the host pipes
using ValueT = int;
constexpr size_t kPipeMinCapacity = 8;
constexpr size_t kReadyLatency = 0;
constexpr size_t kBitsPerSymbol = 1;

using H2DPipe = cl::sycl::ext::intel::prototype::pipe<
    // Usual pipe parameters
    H2DPipeID,         // An identifier for the pipe
    ValueT,            // The type of data in the pipe
    kPipeMinCapacity,  // The capacity of the pipe
    // Additional host pipe parameters
    kReadyLatency,   // Latency for ready signal deassert
    kBitsPerSymbol,  // Symbol size on data bus
    true,            // Exposes a valid on the pipe interface
    false,           // First symbol in high order bits
    cl::sycl::ext::intel::prototype::internal::protocol_name::
        AVALON_STREAMING  // Protocol
    >;

using D2HPipe = cl::sycl::ext::intel::prototype::pipe<
    // Usual pipe parameters
    D2HPipeID,         // An identifier for the pipe
    ValueT,            // The type of data in the pipe
    kPipeMinCapacity,  // The capacity of the pipe
    // Additional host pipe parameters
    kReadyLatency,   // Latency for ready signal deassert
    kBitsPerSymbol,  // Symbol size on data bus
    true,            // Exposes a valid on the pipe interface
    false,           // First symbol in high order bits
    cl::sycl::ext::intel::prototype::internal::protocol_name::
        AVALON_STREAMING  // Protocol
    >;

// forward declare the test functions
void AlternatingTest(queue&, ValueT*, ValueT*, size_t, size_t);
void LaunchCollectTest(queue&, ValueT*, ValueT*, size_t, size_t);

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * sqrt(val)); }

/////////////////////////////////////////

int main(int argc, char* argv[]) {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
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
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    // make sure the device supports USM device allocations
    device d = q.get_device();
    if (!d.has(aspect::usm_host_allocations)) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations" << std::endl;
      return 1;
    }

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

  } catch (exception const& e) {
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
event SubmitLoopBackKernel(queue& q, size_t count) {
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
void AlternatingTest(queue& q, ValueT* in, ValueT* out, size_t count,
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

  std::cout << "\t Waiting on kernel to finish" << std::endl;
  e.wait();
  std::cout << "\t Done" << std::endl;
}

// This test launches SubmitLoopBackKernel, writes 'count'
// elements to H2DPipe, and then reads 'count' elements from
// D2HPipe
void LaunchCollectTest(queue& q, ValueT* in, ValueT* out, size_t count,
                       size_t repeats) {
  std::cout << "\t Run Loopback Kernel on FPGA" << std::endl;
  auto e = SubmitLoopBackKernel<LoopBackKernelID, H2DPipe, D2HPipe>(
      q, count * repeats);

  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": "
              << "Doing " << count << " writes" << std::endl;
    for (size_t i = 0; i < count; i++) {
      // write data in host-to-device hostpipe
      H2DPipe::write(q, in[i]);
    }

    std::cout << "\t " << r << ": "
              << "Doing " << count << " reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      // read data from device-to-host hostpipe
      out[i] = D2HPipe::read(q);
    }
  }

  std::cout << "\t Waiting on kernel to finish" << std::endl;
  e.wait();
  std::cout << "\t Done" << std::endl;
}
