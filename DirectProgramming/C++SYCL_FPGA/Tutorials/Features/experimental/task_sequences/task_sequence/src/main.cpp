#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>

#include "exception_handler.hpp"

constexpr int kVectSize = 128;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class IDOptimized;

// Minimum capacity of a pipe.
// Set to 0 to allow the compiler to save area if possible.
constexpr size_t kPipeMinCapacity = 0;

// Pipes 
class IDPipeIn0;
using PipeIn0 = sycl::ext::intel::experimental::pipe<
    IDPipeIn0,        // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;   
    
class IDPipeIn1;
using PipeIn1 = sycl::ext::intel::experimental::pipe<
    IDPipeIn1,        // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class IDPipeAB;
using PipeAB = sycl::ext::intel::pipe<
    IDPipeAB,         // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class IDPipeBC;
using PipeBC = sycl::ext::intel::pipe<
    IDPipeBC,         // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class IDPipeCD;
using PipeCD = sycl::ext::intel::pipe<
    IDPipeCD,         // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class IDPipeAD;
using PipeAD = sycl::ext::intel::pipe<
    IDPipeAD,         // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

class IDPipeOut;
using PipeOut = sycl::ext::intel::experimental::pipe<
    IDPipeOut,        // An identifier for the pipe
    int,              // The type of data in the pipe
    kPipeMinCapacity  // The capacity of the pipe
    >;

// [[intel::use_stall_enable_clusters]] is required to 
// work around a compiler bug that hurts performance
[[intel::use_stall_enable_clusters]] 
void LoopA(int len) {
  [[intel::initiation_interval(1)]]  
  for (size_t i = 0; i < len; i++) {
    int in0 = PipeIn0::read();
    int in1 = PipeIn1::read();

    PipeAB::write(in0);
    PipeAD::write(in1);
  }
}

// [[intel::use_stall_enable_clusters]] is required to 
// work around a compiler bug that hurts performance
[[intel::use_stall_enable_clusters]] 
void LoopB(int len) {
  [[intel::initiation_interval(1)]]  
  for (size_t i = 0; i < len; i++) {
    int tmp = PipeAB::read();
    tmp += i;
    PipeBC::write(tmp);  
  }
}

// [[intel::use_stall_enable_clusters]] is required to 
// work around a compiler bug that hurts performance
[[intel::use_stall_enable_clusters]] 
void LoopC(int len) {
  [[intel::initiation_interval(1)]]  
  for (size_t i = 0; i < len; i++) {
    int tmp = PipeBC::read();
    tmp += i;
    PipeCD::write(tmp);  
  }
}

// [[intel::use_stall_enable_clusters]] is required to 
// work around a compiler bug that hurts performance
[[intel::use_stall_enable_clusters]] 
void LoopD(int len) {
  [[intel::initiation_interval(1)]]  
  for (size_t i = 0; i < len; i++) {
    int tmp0 = PipeCD::read();
    int tmp1 = PipeAD::read();
    int out = tmp0 + tmp1;
    PipeOut::write(out);  
  }
}

///////////////////////////////////////

struct OptimizedKernel {
  int len;

  void operator()() const {
    sycl::ext::intel::experimental::task_sequence<LoopA> task_a;
    sycl::ext::intel::experimental::task_sequence<LoopB> task_b;
    sycl::ext::intel::experimental::task_sequence<LoopC> task_c;
    sycl::ext::intel::experimental::task_sequence<LoopD> task_d;

    task_a.async(len);
    task_b.async(len);
    task_c.async(len);
    task_d.async(len);
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
    auto e = q.single_task<IDOptimized>(OptimizedKernel{kVectSize});

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

    // Wait for kernel to exit
    e.wait();
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