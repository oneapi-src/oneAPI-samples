#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <chrono>
#include <thread>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

// The type that will stream through the IO pipe. When using real IO pipes,
// make sure the width of this datatype matches the width of the IO pipe, which
// you can find in the BSP XML file.
using IOPipeType = int;

#include "LoopbackTest.hpp"
#include "SideChannelTest.hpp"

using namespace sycl;

// check is USM host allocations are enabled
#if defined(USM_HOST_ALLOCATIONS)
constexpr bool kUseUSMHostAllocation = true;
#else
constexpr bool kUseUSMHostAllocation = false;
#endif

int main() {
  bool passed = true;

#if defined(FPGA_EMULATOR)
  size_t count = 1 << 12;
#else
  size_t count = 1 << 24;
#endif

  try {
    // device selector
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // queue properties to enable SYCL profiling of kernels
    auto prop_list = property_list{property::queue::enable_profiling()};

    // create the device queue
    queue q(selector, dpc_common::exception_handler, prop_list);

    // run the loopback example system
    // see 'LoopbackTest.hpp'
    std::cout << "Running loopback test\n";
    passed &= 
      RunLoopbackSystem<IOPipeType, kUseUSMHostAllocation>(q, count);

    // run the side channel example system
    // see 'SideChannelTest.hpp'
    std::cout << "Running side channel test\n";
    passed &= 
      RunSideChannelsSystem<IOPipeType, kUseUSMHostAllocation>(q, count);

  } catch (exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
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

