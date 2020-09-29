//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include "lib.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// Values used as input to the kernel
constexpr float kA = 2.0f;
constexpr float kB = 3.0f;

// Forward declaration of the kernel name
// (This will become unnecessary in a future compiler version.)
class KernelCompute;

int main() {
  unsigned result = 0;

  // Select either the FPGA emulator (CPU) or FPGA device
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  try {
    queue q(device_selector, dpc_common::exception_handler);

    // The scalar inputs are passed to the kernel using the lambda capture,
    // but a SYCL buffer must be used to return a scalar from the kernel.
    buffer<unsigned, 1> buffer_c(&result, 1);

    q.submit([&](handler &h) {

      // Accessor to the scalar result
      auto accessor_c = buffer_c.get_access<access::mode::discard_write>(h);

      // Kernel
      h.single_task<class KernelCompute>([=]() {

        // OclSquare is an OpenCL function, defined in lib_ocl.cl.
        float a_sq = OclSquare(kA);

        // HlsSqrtf is an Intel HLS component, defined in lib_hls.cpp.
        // (Intel HLS is a C++ based High Level Synthesis language for FPGA.)
        float a_sq_sqrt = HlsSqrtf(a_sq);

        // SyclSquare is a SYCL library function, defined in lib_sycl.cpp.
        float b_sq = SyclSquare(kB);

        // RtlByteswap is an RTL library.
        //  - When compiled for FPGA, Verilog module byteswap_uint in lib_rtl.v
        //    is instantiated in the datapath by the compiler.
        //  - When compiled for FPGA emulator (CPU), the C model of RtlByteSwap
        //    in lib_rtl_model.cpp is used instead.
        accessor_c[0] = RtlByteswap((unsigned)(a_sq_sqrt + b_sq));
      });
    });
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // Compute the expected "golden" result
  unsigned gold = sqrt(kA * kA) + (kB * kB);
  gold = gold << 16 | gold >> 16;

  // Check the results
  if (result != gold) {
    std::cout << "FAILED: result is incorrect!\n";
    return -1;
  }
  std::cout << "PASSED: result is correct!\n";
  return 0;
}
