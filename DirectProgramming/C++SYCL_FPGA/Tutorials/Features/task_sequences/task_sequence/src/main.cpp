#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// forward declare kernel and pipe names to reduce name mangling
class IDOptimized;

constexpr int kVectSize = 128;

// Host pipes
using ValueT = int;
constexpr size_t kPipeMinCapacity = 0;

// Pipes for task_sequences
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

class PipeAB_ID;
using PipeAB = sycl::ext::intel::pipe<
    // Usual pipe parameters
    PipeAB_ID,        // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class PipeBC_ID;
using PipeBC = sycl::ext::intel::pipe<
    // Usual pipe parameters
    PipeBC_ID,        // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class PipeCD_ID;
using PipeCD = sycl::ext::intel::pipe<
    // Usual pipe parameters
    PipeCD_ID,        // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class PipeAD_ID;
using PipeAD = sycl::ext::intel::pipe<
    // Usual pipe parameters
    PipeAD_ID,        // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // CThe capacity of the pipe
    >;

class PipeOut_ID;
using PipeOut = sycl::ext::intel::experimental::pipe<
    // Usual pipe parameters
    PipeOut_ID,       // An identifier for the pipe
    ValueT,           // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

[[intel::use_stall_enable_clusters]] //
void loopA(int len) {
  [[intel::initiation_interval(1)]]  //
  for (size_t i = 0; i < len; i++) {
    int in0 = PipeIn0::read();
    int in1 = PipeIn1::read();

    PipeAB::write(in0);
    PipeAD::write(in1);
  }
}

[[intel::use_stall_enable_clusters]] //
void loopB(int len) {
  [[intel::initiation_interval(1)]]  //
  for (size_t i = 0; i < len; i++) {
    int tmp = PipeAB::read();
    tmp += i;
    PipeBC::write(tmp);  
  }
}

[[intel::use_stall_enable_clusters]] //
void loopC(int len) {
  [[intel::initiation_interval(1)]]  //
  for (size_t i = 0; i < len; i++) {
    int tmp = PipeBC::read();
    tmp += i;
    PipeCD::write(tmp);  
  }
}

[[intel::use_stall_enable_clusters]] //
void loopD(int len) {
  [[intel::initiation_interval(1)]]  //
  for (size_t i = 0; i < len; i++) {
    int tmp0 = PipeCD::read();
    int tmp1 = PipeAD::read();
    int out = tmp0 + tmp1;
    PipeOut::write(out);  
  }
}

struct OptimizedKernel {
  int len;

  void operator()() const {
    sycl::ext::intel::experimental::task_sequence<loopA> taskA;
    sycl::ext::intel::experimental::task_sequence<loopB> taskB;
    sycl::ext::intel::experimental::task_sequence<loopC> taskC;
    sycl::ext::intel::experimental::task_sequence<loopD> taskD;

    taskA.async(len);
    taskB.async(len);
    taskC.async(len);
    taskD.async(len);
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

    // declare arrays and fill them
    // allocate in shared memory so the kernel can see them
    for (int i = 0; i < kVectSize; i++) {
      a[i] = i;                
      b[i] = (kVectSize - i);  

      PipeIn0::write(q, i);
      PipeIn1::write(q, kVectSize - i);
    }

    auto e = q.single_task<IDOptimized>(OptimizedKernel{kVectSize});
    e.wait();

    // verify that output is correct
    passed = true;
    for (int i = 0; i < kVectSize; i++) {
      int expected = a[i] * 3 + b[i];
      int result = PipeOut::read(q);
      if (result != expected) {
        std::cout << "idx=" << i << ": task_sequences result " << result
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