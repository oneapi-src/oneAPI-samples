//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/ext/intel/ac_types/ac_int.hpp>
// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "lib_rtl.hpp"

// RTL Library will use the Verilog model during hardware generation, and the
// c++ model during emulation.
#include "exception_handler.hpp"
#include <stdint.h>
// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization report.
class KernelCompute;
class KernelComputeRTL;

// Using host pipes to stream data in and out of kernal
// IDPipeA and IDPipeB will be written to by the host, and then read by the kernel (device)
// IDPipeC will be written to by the kernel (device), and then read by the host
class IDPipeA;
using InputPipeA = sycl::ext::intel::experimental::pipe<IDPipeA, uint32_t>;
class IDPipeB;
using InputPipeB = sycl::ext::intel::experimental::pipe<IDPipeB, uint32_t>;
class IDPipeC;
using OutputPipeC = sycl::ext::intel::experimental::pipe<IDPipeC, uint64_t>;

class IDPipeD;
using InputPipeD = sycl::ext::intel::experimental::pipe<IDPipeD, uint32_t>;
class IDPipeE;
using InputPipeE = sycl::ext::intel::experimental::pipe<IDPipeE, uint32_t>;
class IDPipeF;
using OutputPipeF = sycl::ext::intel::experimental::pipe<IDPipeF, uint64_t>;

// This kernel computes multiplier result by using the C++ '*' operator
template <typename PipeIn1, typename PipeIn2, typename PipeOut>
struct NativeMult27x27 {

  // use a streaming pipelined invocation interface to minimize hardware
  // overhead
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::streaming_interface_accept_downstream_stall, 
        sycl::ext::intel::experimental::pipelined<1>};
  }
  
   void operator()() const {
    MyInt27 a_val = PipeIn1::read();
    MyInt27 b_val = PipeIn2::read();
    MyInt54 res =(MyInt54)a_val * b_val;
    PipeOut::write(res);
  }
};

// This kernel computes multiplier result by calling RTL function RtlDSPm27x27u
template <typename PipeIn1, typename PipeIn2, typename PipeOut>
struct RtlMult27x27 {

  // use a streaming pipelined invocation interface to minimize hardware
  // overhead
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::streaming_interface_accept_downstream_stall, 
        sycl::ext::intel::experimental::pipelined<1>};
  }
  
  void operator()() const {
    MyInt27 a_val = PipeIn1::read();
    MyInt27 b_val = PipeIn2::read();
    MyInt54 res = RtlDSPm27x27u(a_val, b_val);
    PipeOut::write(res);
  }
};

int main() {
  uint64_t result_rtl = 0;
  uint64_t result_native = 0;
  uint32_t kA = 134217727; // 0x7FFFFFF is the largest possible ac_int<27, false>.
  uint32_t kB = 100;

  // Select the FPGA emulator (CPU), FPGA simulator, or FPGA device
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    sycl::queue q(selector, fpga_tools::exception_handler);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;
    {
      // write data to host-to-device pipes
      InputPipeA::write(q, kA);
      InputPipeB::write(q, kB);
      // launch kernel that infers a multiplier automatically
      q.single_task<KernelCompute>(NativeMult27x27<InputPipeA,InputPipeB,OutputPipeC>{}).wait();
      // read data from device-to-host pipe
      result_native = OutputPipeC::read(q);
    }
    {
      // write data to host-to-device pipes
      InputPipeD::write(q, kA);
      InputPipeE::write(q, kB);
      // launch a kernel to that uses a multiplier defined in RTL
      q.single_task<KernelComputeRTL>(RtlMult27x27<InputPipeD,InputPipeE,OutputPipeF>{}).wait();
      // read data from device-to-host pipe
      result_rtl = OutputPipeF::read(q);
    }

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

  // Check the results
  uint64_t expected_result = (uint64_t ) kA * kB;
  if (result_native != expected_result) {
    std::cout << "FAILED: result native (" << result_native << ") is incorrect! Expected " << expected_result << "\n";
    return -1;
  }
  if (result_rtl != expected_result) {
    std::cout << "FAILED: result RTL (" << result_rtl << ") is incorrect! Expected " << expected_result << "\n";
    return -1;
  }
  std::cout << "PASSED: result is correct!\n";
  return 0;
}

