#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

constexpr int kVectSize = 128;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class IDNaive;

using ValueT = int;

// Minimum capacity of a pipe.
// Set to 0 to let compiler decides on the pipe capacity.
constexpr size_t kPipeMinCapacity = 0;
constexpr size_t kPipeMinCapacity = 0;

// Pipes
class PipeIn0_ID;
using PipeIn0 = sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    PipeIn0_ID,       // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class PipeIn1_ID;
using PipeIn1 = sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    PipeIn1_ID,       // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class PipeOut_ID;
using PipeOut = sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    PipeOut_ID,       // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

///////////////////////////////////////

struct NaiveKernel {
  int len;

  void operator()() const {
    int arrAB[kVectSize];
    int arrBC[kVectSize];
    int arrCD[kVectSize];
    int arrAD[kVectSize];

    // loopA
    [[intel::initiation_interval(1)]]  
    for (size_t i = 0; i < len; i++) {
      int in0 = PipeIn0::read();
      int in1 = PipeIn1::read();
      arrAB[i] = in0;
      arrAD[i] = in1;
    }

    // loopB
    [[intel::initiation_interval(1)]]  
    for (size_t i = 0; i < len; i++) {
      int tmp = arrAB[i];
      tmp += i;
      arrBC[i] = tmp;
    }

    // loopC
    [[intel::initiation_interval(1)]]  
    for (size_t i = 0; i < len; i++) {
      int tmp = arrBC[i];
      tmp += i;
      arrCD[i] = tmp;
    }

    // loopD
    [[intel::initiation_interval(1)]]  
    for (size_t i = 0; i < len; i++) {
      int tmp0 = arrCD[i];
      int tmp1 = arrAD[i];
      int out = tmp0 + tmp1;
      PipeOut::write(out);  
    }
  }
};

int main() {
  bool passed = false;

  try {
    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int *a = new int[kVectSize];
    int *b = new int[kVectSize];

    // Generate input data
    for (int i = 0; i < kVectSize; i++) {
      a[i] = i;                
      b[i] = (kVectSize - i); 

      PipeIn0::write(q, i);
      PipeIn1::write(q, kVectSize - i);
    }

    // Call the kernel
    auto e = q.single_task<IDNaive>(NaiveKernel{kVectSize});
    e.wait();

    // verify that output is correct
    passed = true;
    for (int i = 0; i < kVectSize; i++) {
      int expected = a[i] * 3 + b[i];
      int result = PipeOut::read(q);
      if (result != expected) {
        std::cout << "idx=" << i << ": naive result " << result
                  << ", expected (" << expected << ") ." << std::endl;
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    delete[] a;
    delete[] b;

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code.
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

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
