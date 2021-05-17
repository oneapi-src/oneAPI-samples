//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

//==============================================================
// The purpose of this tutorial is to examine the drawbacks
// of not using fp pragmas
//==============================================================

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

// Unify the baseline of the kernels by setting fp pragmas to on (by default) in global
// scope.
constexpr double EPSILON = 1e-6;

// Forward declare the kernel name in the global scope.
class ContractOffKernel;
class ReassociateOffKernel;
class ContractFastKernel;
class ReassociateOnKernel;

// Launch a kernel on the device specified by selector.
// The kernel's functionality is designed to show the
// baseline performance without the fp contract pragma.
void enable_contract(const device_selector &selector, const double a,
                      const double b, const double c, const double d,
                      const double *arr, const size_t size,
                      double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&res, 1);
    buffer<double, 1> bufferArray(arr, size);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorRes(bufferE, h, write_only);
      accessor accessorArray(bufferArray, h, read_only);

      // The "kernel_args_restrict" tells the compiler that a, b, and r
      // do not alias. For a full explanation, see:
      //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
      h.single_task<ContractFastKernel>([=]() [[intel::kernel_args_restrict]] {
        accessorRes[0] = 0.0;

        for (size_t i = 0; i < size; i++) {
          double temp1 = 0.0, temp2 = 0.0;
          temp1 = accessorA[0] + accessorB[0];
          temp2 = accessorC[0] + accessorArray[i];
          accessorRes[0] += temp1 * temp2 + accessorD[0];
        }
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
void disable_contract(const device_selector &selector, const double a,
                      const double b, const double c, const double d,
                      const double *arr, const size_t size,
                      double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&res, 1);
    buffer<double, 1> bufferArray(arr, size);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorRes(bufferE, h, write_only);
      accessor accessorArray(bufferArray, h, read_only);

      // The "kernel_args_restrict" tells the compiler that a, b, and r
      // do not alias. For a full explanation, see:
      //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
      h.single_task<ContractOffKernel>([=]() [[intel::kernel_args_restrict]] {
#pragma clang fp contract(off)

        accessorRes[0] = 0.0;

        for (size_t i = 0; i < size; i++) {
          double temp1 = 0.0, temp2 = 0.0;
          temp1 = accessorA[0] + accessorB[0];
          temp2 = accessorC[0] + accessorArray[i];
          accessorRes[0] += temp1 * temp2 + accessorD[0];
        }
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
                      const double e, const double f,
                      double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&e, 1);
    buffer<double, 1> bufferF(&f, 1);
    buffer<double, 1> bufferRes(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorE(bufferE, h, read_only);
      accessor accessorF(bufferF, h, read_only);
      accessor accessorRes(bufferRes, h, write_only, noinit);

      // The "kernel_args_restrict" tells the compiler that a, b, and r
      // do not alias. For a full explanation, see:
      //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
      h.single_task<ReassociateOnKernel>([=]() [[intel::kernel_args_restrict]] {
        accessorRes[0] += accessorA[0] + accessorB[0]
                          + accessorC[0] + accessorD[0]
                          + accessorE[0] + accessorF[0];
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
                      const double e, const double f,
                      double &res) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<double, 1> bufferA(&a, 1);
    buffer<double, 1> bufferB(&b, 1);
    buffer<double, 1> bufferC(&c, 1);
    buffer<double, 1> bufferD(&d, 1);
    buffer<double, 1> bufferE(&e, 1);
    buffer<double, 1> bufferF(&f, 1);
    buffer<double, 1> bufferRes(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessorA(bufferA, h, read_only);
      accessor accessorB(bufferB, h, read_only);
      accessor accessorC(bufferC, h, read_only);
      accessor accessorD(bufferD, h, read_only);
      accessor accessorE(bufferE, h, read_only);
      accessor accessorF(bufferF, h, read_only);
      accessor accessorRes(bufferRes, h, write_only, noinit);

      // The "kernel_args_restrict" tells the compiler that a, b, and r
      // do not alias. For a full explanation, see:
      //    DPC++FPGA/Tutorials/Features/kernel_args_restrict
      h.single_task<ReassociateOffKernel>([=]() [[intel::kernel_args_restrict]] {
#pragma clang fp reassociate(off)

        accessorRes[0] += accessorA[0] + accessorB[0]
                          + accessorC[0] + accessorD[0]
                          + accessorE[0] + accessorF[0];
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
  const double a = 4235.3245, b = 2346.733, c = 343.20537195, d = 623.7643257;
  const double e = 65465.65737, f = 6576.65787189;
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
  double arr[] = {
    43.25534, 234.5223562, 6523423.16241, 42.2364325, 340651.23567881
  };


  // Test performance loss of turning fp contract pragma off
  enable_contract(selector, a, b, c, d, arr, 5, res_contract_fast);
  disable_contract(selector, a, b, c, d, arr, 5, res_contract_off);
  // Test performance loss of turning fp reassociate pragma off
  enable_reassociate(selector, a, b, c, d, e, f, res_reassociate_on);
  disable_reassociate(selector, a, b, c, d, e, f, res_reassociate_off);

  // The fp contract and reassociate pragmas can save area and improve
  // performance, but because the operations can be reordered we may get
  // slightly different results from those computed by the CPU. The EPISILON
  // constant allows us to define what we consider to be 'close enough' to an
  // exact match.
  if (fabs(res_contract_off - res_contract_fast) <= EPSILON
      && fabs(res_reassociate_off - res_reassociate_on) <= EPSILON) {
    printf(
        "Passed: The results are correct. Result for contract off: %f, "
        "contract fast: %f, reassociate off: %f, reassociate on: %f\n",
        res_contract_off, res_contract_fast, res_reassociate_off,
        res_reassociate_on);
  } else {
    printf(
        "FAILED: The results are incorrect. Result for contract off: %f, "
        "contract fast: %f, reassociate off: %f, reassociate on: %f\n",
        res_contract_off, res_contract_fast, res_reassociate_off,
        res_reassociate_on);
  }
  return 0;
}
