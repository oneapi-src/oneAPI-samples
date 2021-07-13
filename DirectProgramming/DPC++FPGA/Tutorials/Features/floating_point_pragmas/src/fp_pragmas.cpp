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
#include <algorithm>
#include <random>
#include <chrono>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

// The constant double representing allowed error
constexpr double kEpsilon = 1e-6;

// Fp pragmas are turned on by default

// Forward declare the kernel name in the global scope.
class ContractOffKernel;
class ReassociateOffKernel;
class ContractFastKernel;
class ReassociateOnKernel;

// Unified error handler for sycl kernels
void syclErrorHandler(sycl::exception const &e) {
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

// Launch a kernel to perform multiplication and addition for each double
// element within a, b, c, and store the result into res
//
// Template variable 'contractEnabled' is used to control whether contract is
// turned on or off
template <bool contractEnabled>
void testContract(queue &q, const vector<double> &a,
                  const vector<double> &b, const vector<double> &c,
                  double &res) {
  // Turn reassociate off while testing contract
#pragma clang fp reassociate(off)

  // Prepare buffer
  assert(a.size() == b.size());
  assert(b.size() == c.size());
  size_t size = a.size();

  buffer<double> bufA(a);
  buffer<double> bufB(b);
  buffer<double> bufC(c);
  buffer<double, 1> bufRes(&res, 1);

  event e = q.submit([&](handler &h) {
    accessor accA(bufA, h, read_only);
    accessor accB(bufB, h, read_only);
    accessor accC(bufC, h, read_only);
    accessor accRes(bufRes, h, write_only, noinit);

    if constexpr (contractEnabled) {
      // Turn on fp contract if boolean template variable is true
#pragma clang fp contract(fast)
      h.single_task<ContractFastKernel>([=]() [[intel::kernel_args_restrict]] {
        accRes[0] = 0.0;
        for (size_t i = 0; i < size; i++)
          accRes[0] += accA[i] * accB[i] + accC[i];
      });
    } else {
#pragma clang fp contract(off)
      // Turn off fp contract if boolean template variable is false
      h.single_task<ContractOffKernel>([=]() [[intel::kernel_args_restrict]] {
        accRes[0] = 0.0;
        for (size_t i = 0; i < size; i++)
          accRes[0] += accA[i] * accB[i] + accC[i];
      });
    }
  });
}

// Launch a kernel to perform multiplication and addition for each double
// element within a, b, c, and store the result into res in step of 4
//
// Template variable 'reassociateEnabled' is used to control whether
// reassociation is turned on or off
template <bool reassociateEnabled>
void testReassociate(queue &q, const vector<double> &a,
                      const vector<double> &b, const vector<double> &c,
                      double &res) {
  // Turn contract off while testing contract
#pragma clang fp contract(off)

  // Prepare buffer
  assert(a.size() == b.size());
  assert(b.size() == c.size());
  assert(a.size() % 4 == 0);
  size_t size = a.size();

  buffer<double> bufA(a);
  buffer<double> bufB(b);
  buffer<double> bufC(c);
  buffer<double, 1> bufRes(&res, 1);

  event e = q.submit([&](handler &h) {
    accessor accA(bufA, h, read_only);
    accessor accB(bufB, h, read_only);
    accessor accC(bufC, h, read_only);
    accessor accRes(bufRes, h, write_only, noinit);

    if constexpr (reassociateEnabled) {
#pragma clang fp reassociate(on)
      // Turn on fp reassociate if boolean template variable is true
      h.single_task<ReassociateOnKernel>([=]() [[intel::kernel_args_restrict]] {
        accRes[0] = 0.0;
        for (size_t i = 0; i < size; i += 4)
          accRes[0] += accA[i] * accB[i] + accC[i]
                       + accA[i + 1] * accB[i + 1] + accC[i + 1]
                       + accA[i + 2] * accB[i + 2] + accC[i + 2]
                       + accA[i + 3] * accB[i + 3] + accC[i + 3];
      });
    } else {
#pragma clang fp reassociate(off)
      // Turn off fp reassociate if boolean template variable is false
      h.single_task<ReassociateOffKernel>([=]() [[intel::kernel_args_restrict]] {
        accRes[0] = 0.0;
        for (size_t i = 0; i < size; i += 4)
          accRes[0] += accA[i] * accB[i] + accC[i]
                       + accA[i + 1] * accB[i + 1] + accC[i + 1]
                       + accA[i + 2] * accB[i + 2] + accC[i + 2]
                       + accA[i + 3] * accB[i + 3] + accC[i + 3];
      });
    }
  });
}

// Given the output of the kernel and the input, test
// whether the output matches the input
//
// Template variable 'contractEnabled' is used to test the
// correctness of the data while contract is on or off
template <bool contractEnabled>
bool verifyContract(const vector<double> &a, const vector<double> &b,
                    const vector<double> &c, double res) {
// Turn off pragmas for CPU code
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  assert(a.size() == b.size());
  assert(b.size() == c.size());
  size_t size = a.size();

  double acc = 0.0;
  for (size_t i = 0; i < size; i++)
    acc += a[i] * b[i] + c[i];
  
  std::cout << (contractEnabled ? "Contract enabled: " : "Contract disabled: ")
            << std::endl;

  std::cout << "    kernel result: " << res
            << ", correct result: " << acc
            << std::endl;

  double delta = fabs(res - acc);

  std::cout << "    delta: " << delta
            << ", expected delta: " << (contractEnabled ? ("less than " + to_string(kEpsilon)) : "0")
            << std::endl;

  // delta should be 0 if contract is off, and should less than expected
  // delta when contract is on
  return contractEnabled ? delta < kEpsilon : delta == 0.0;
}

// Given the output of the kernel and the input, test
// whether the output matches the input
//
// Template variable 'reassociateEnabled' is used to test the
// correctness of the data while reassociate is on or off
template <bool reassociateEnabled>
bool verifyReassociate(const vector<double> &a, const vector<double> &b,
                       const vector<double> &c, double res) {
// Turn off pragmas for CPU code
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  assert(a.size() == b.size());
  assert(b.size() == c.size());
  assert(a.size() % 4 == 0);
  size_t size = a.size();

  double acc = 0.0;
  for (size_t i = 0; i < size; i += 4)
    acc += a[i] * b[i] + c[i]
           + a[i + 1] * b[i + 1] + c[i + 1]
           + a[i + 2] * b[i + 2] + c[i + 2]
           + a[i + 3] * b[i + 3] + c[i + 3];
  
  std::cout << "Reassociate "
            << (reassociateEnabled ? "enabled: " : "disabled: ")
            << std::endl;

  std::cout << "    kernel result: " << res
            << ", correct result: " << acc
            << std::endl;

  double delta = fabs(res - acc);

  std::cout << "    delta: " << delta
            << ", expected delta: "
            << (reassociateEnabled
                ? ("less than " + to_string(kEpsilon))
                : "0")
            << std::endl;

  // delta should be 0 if reassociate is off, and should less than expected
  // delta when reassociate is on
  return reassociateEnabled ? delta < kEpsilon : delta == 0.0;
}

// Generate randomized doubles for vec
void randomizeVector(vector<double> &vec) {
  constexpr double lb = -10000;
  constexpr double ub = 10000;
  static std::uniform_real_distribution<double> ud(lb, ub);
  static std::default_random_engine re;

  for (auto &it : vec)
    it = ud(re);
}

int main() {
  // Prepare data
  constexpr size_t size = 1 << 8;

  vector<double> a(size), b(size), c(size);
  randomizeVector(a);
  randomizeVector(b);
  randomizeVector(c);
  double res;

  try {
    queue q(default_selector{}, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    bool success = true;

    testContract<true>(q, a, b, c, res);
    success &= verifyContract<true>(a, b, c, res);
    testContract<false>(q, a, b, c, res);
    success &= verifyContract<false>(a, b, c, res);

    testReassociate<true>(q, a, b, c, res);
    success &= verifyReassociate<true>(a, b, c, res);
    testReassociate<false>(q, a, b, c, res);
    success &= verifyReassociate<false>(a, b, c, res);

    std::cout << (success ? "Test Passed" : "Test Failed") << std::endl;
  } catch (sycl::exception const &e) {
    syclErrorHandler(e);
  }

  return 0;
}
