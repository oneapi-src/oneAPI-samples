//==- fpga_sim_device_selector.hpp - SYCL FPGA sim device selector shortcut ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/fpga_device_selector.hpp>

#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
    namespace sycl {
    class platform;
    namespace ext {
    namespace intel {

    class fpga_simulator_selector : public fpga_selector {
      public:
        fpga_simulator_selector() {
            // Tell the runtime to use a simulator device rather than hardware
#ifdef _WIN32
            _putenv_s("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "1");
#else
            setenv("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "1", 0);
#endif
        }
    };

    } // namespace intel
    } // namespace ext

    } // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

cl::sycl::ext::intel::platform_selector chooseSelector() {
#if FPGA_SIMULATOR
    std::cout << "using FPGA Simulator." << std::endl;
    return sycl::ext::intel::fpga_simulator_selector{};
#elif FPGA_HARDWARE
    std::cout << "using FPGA Hardware." << std::endl;
    return sycl::ext::intel::fpga_selector{};
#else // #if FPGA_EMULATOR
    std::cout << "using FPGA Emulator." << std::endl;
    return sycl::ext::intel::fpga_emulator_selector{};
#endif
}
