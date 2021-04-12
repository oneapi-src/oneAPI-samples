//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
constexpr double EPSILON = 1e-6;

// Forward declare the kernel name in the global scope.
class ContractOffKernel;
class ReassociateOffKernel;
class ContractFastKernel;
class ReassociateOnKernel;

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// baseline performance without the fp contract pragma.
void disable_contract(const device_selector &selector, const double a,
                      const double b, const double c, const double d,
                      double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorRes(bufferE, h, write_only, noinit);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<ContractOffKernel>([=]() [[intel::kernel_args_restrict]] {
        double temp1 = 0.0, temp2 = 0.0;
        temp1 = accessorA[0] + accessorB[0];
        temp2 = accessorC[0] + accessorD[0];
        accessorRes[0] = temp1 * temp2;
      });
    });
  } catch (sycl::exception const &e) {
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
}

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// baseline performance without the fp reassociate pragma.
void disable_reassociate(const device_selector &selector, const double a,
                         const double b, const double c, const double d,
                         double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorRes(bufferE, h, write_only, noinit);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<ReassociateOffKernel>([=
      ]() [[intel::kernel_args_restrict]] {
        accessorRes[0] =
            accessorA[0] + accessorB[0] + accessorC[0] + accessorD[0];
      });
    });
  } catch (sycl::exception const &e) {
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
}

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// performance impact of the fp contract pragma.
void enable_contract(const device_selector &selector, const double a,
                     const double b, const double c, const double d,
                     double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorRes(bufferE, h, write_only, noinit);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<ContractFastKernel>([=]() [[intel::kernel_args_restrict]] {
#pragma clang fp contract(fast)
        double temp1 = 0.0, temp2 = 0.0;
        temp1 = accessorA[0] + accessorB[0];
        temp2 = accessorC[0] + accessorD[0];
        accessorRes[0] = temp1 * temp2;
      });
    });
  } catch (sycl::exception const &e) {
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
}

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// performance impact of the fp reassociate pragma.
void enable_reassociate(const device_selector &selector, const double a,
                        const double b, const double c, const double d,
                        double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorRes(bufferE, h, write_only, noinit);

      // FPGA-optimized kernel
      // Using kernel_args_restrict tells the compiler that the input
      // and output buffers won't alias.
      h.single_task<ReassociateOnKernel>([=]() [[intel::kernel_args_restrict]] {
#pragma clang fp reassociate(on)
        accessorRes[0] =
            accessorA[0] + accessorB[0] + accessorC[0] + accessorD[0];
      });
    });
  } catch (sycl::exception const &e) {
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
}

int main() {
  const double a = 1.0, b = 2.0, c = 3.0, d = 4.0;
  // Result variables
  double res_contract_off, res_reassociate_off, res_contract_fast,
      res_reassociate_on;
  const double golden_res_contract = 21.0;
  const double golden_res_reassociate = 10.0;

  // Run the kernel on either the FPGA emulator, or FPGA
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // Test fp contract pragma
  disable_contract(selector, a, b, c, d, res_contract_off);
  enable_contract(selector, a, b, c, d, res_contract_fast);
  // Test fp reassociate pragma
  disable_reassociate(selector, a, b, c, d, res_reassociate_off);
  enable_reassociate(selector, a, b, c, d, res_reassociate_on);

  if (fabs(res_contract_off - golden_res_contract) <= EPSILON &&
      fabs(res_reassociate_off - golden_res_reassociate) <= EPSILON &&
      fabs(res_contract_fast - golden_res_contract) <= EPSILON &&
      fabs(res_reassociate_on - golden_res_reassociate) <= EPSILON) {
    printf("PASSED: The results are correct\n");
  } else {
    printf(
        "FAILED: The results are incorrect. Result for contract off: %f, "
        "contract fast: %f, reassociate off: %f, reassociate on: %f\n",
        res_contract_off, res_contract_fast, res_reassociate_off,
        res_reassociate_on);
  }
  return 0;
}
