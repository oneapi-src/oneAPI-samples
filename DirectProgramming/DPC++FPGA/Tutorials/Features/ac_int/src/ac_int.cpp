//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <bitset>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class BasicOpsInt;
class BasicOpsAcInt;
class ShiftOp;
class EfficientShiftOp;
class BitOps;

using my_uint2 = ac_int<2, false>;
using my_int14 = ac_int<14, true>;
using my_int15 = ac_int<15, true>;
using my_int28 = ac_int<28, true>;

void TestBasicOpsInt(queue &q, const int &a, const int &b, int &c, int &d,
                     int &e) {
  buffer<int, 1> a_buf(&a, 1);
  buffer<int, 1> b_buf(&b, 1);
  buffer<int, 1> c_buf(&c, 1);
  buffer<int, 1> d_buf(&d, 1);
  buffer<int, 1> e_buf(&e, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor b_acc(b_buf, h, read_only);
    accessor c_acc(c_buf, h, write_only, no_init);
    accessor d_acc(d_buf, h, write_only, no_init);
    accessor e_acc(e_buf, h, write_only, no_init);
    h.single_task<BasicOpsInt>([=]() [[intel::kernel_args_restrict]] {
      c_acc[0] = a_acc[0] + b_acc[0];
      d_acc[0] = a_acc[0] * b_acc[0];
      e_acc[0] = a_acc[0] / b_acc[0];
    });
  });
}

void TestBasicOpsAcInt(queue &q, const my_int14 &a, const my_int14 &b,
                       my_int15 &c, my_int28 &d, my_int15 &e) {
  buffer<my_int14, 1> a_buf(&a, 1);
  buffer<my_int14, 1> b_buf(&b, 1);
  buffer<my_int15, 1> c_buf(&c, 1);
  buffer<my_int28, 1> d_buf(&d, 1);
  buffer<my_int15, 1> e_buf(&e, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor b_acc(b_buf, h, read_only);
    accessor c_acc(c_buf, h, write_only, no_init);
    accessor d_acc(d_buf, h, write_only, no_init);
    accessor e_acc(e_buf, h, write_only, no_init);
    h.single_task<BasicOpsAcInt>([=]() [[intel::kernel_args_restrict]] {
      c_acc[0] = a_acc[0] + b_acc[0];
      d_acc[0] = a_acc[0] * b_acc[0];
      e_acc[0] = a_acc[0] / b_acc[0];
    });
  });
}

void TestShiftOp(queue &q, const my_int14 &a, const my_int14 &b, my_int14 &c) {
  buffer<my_int14, 1> a_buf(&a, 1);
  buffer<my_int14, 1> b_buf(&b, 1);
  buffer<my_int14, 1> c_buf(&c, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor b_acc(b_buf, h, read_only);
    accessor c_acc(c_buf, h, write_only, no_init);
    h.single_task<ShiftOp>([=]() [[intel::kernel_args_restrict]] {
      c_acc[0] = a_acc[0] << b_acc[0];
    });
  });
}

void TestEfficientShiftOp(queue &q, const my_int14 &a, const my_uint2 &b,
                          my_int14 &c) {
  buffer<my_int14, 1> a_buf(&a, 1);
  buffer<my_uint2, 1> b_buf(&b, 1);
  buffer<my_int14, 1> c_buf(&c, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor b_acc(b_buf, h, read_only);
    accessor c_acc(c_buf, h, write_only, no_init);
    h.single_task<EfficientShiftOp>([=]() [[intel::kernel_args_restrict]] {
      c_acc[0] = a_acc[0] << b_acc[0];
    });
  });
}

my_int14 TestBitOps(queue &q, const my_int14 &a) {
  my_int14 res;
  buffer<my_int14, 1> a_buf(&a, 1);
  buffer<my_int14, 1> res_buf(&res, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor res_acc(res_buf, h, write_only, no_init);
    h.single_task<BitOps>([=]() [[intel::kernel_args_restrict]] {
      my_int14 temp = a_acc[0].slc<14>(0);

      res_acc[0] = 0;  // Must be initialized before bit operations, otherwise,
                       // it will be undefined behavior

      // 0b0 -> 0b1111101000 (decimal:1000)
      res_acc[0].set_slc(0, temp);

      // 0b1111101000 -> 0b1111101111
      res_acc[0][2] = 1;
      res_acc[0][1] = 1;
      res_acc[0][0] = 1;
    });
  });
  return res;
}

int main() {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  bool passed = true;

  try {
    queue q(device_selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    const int t1 = 1000, t2 = 2;

    // Kernel `BasicOpsInt` contains native `int` type addition, multiplication,
    // and division operations, while kernel `BasicOpsAcInt` contains `ac_int`
    // type addition, multiplication, and division operations. By comparing
    // these two kernels, you will find reduced width `ac_int` generates more
    // efficient hardware than native `int`.
    {
      my_int14 a = t1, b = t2;
      my_int15 c;
      my_int28 d;
      my_int15 e;
      TestBasicOpsAcInt(q, a, b, c, d, e);

      int _c, _d, _e;
      TestBasicOpsInt(q, a, b, _c, _d, _e);

      if (c != _c || d != _d || e != _e) {
        std::cout << "Result mismatch!\n"
                  << "Kernel BasicOpsInt:   addition = " << _c
                  << ", multiplication = " << _d << ", division = " << _e
                  << "\n"
                  << "Kernel BasicOpsAcInt: addition = " << c
                  << ", multiplication = " << d << ", division = " << e
                  << "\n\n";
        passed = false;
      }
    }

    // Kernel `ShiftOp` contains an `ac_int` left shifter and the data type of
    // the shift amount is a large width signed `ac_int`. On contrast, kernel
    // `EfficientShiftOp` also contains an `ac_int` left shifter but the data
    // type of the shift amount is a reduced width unsigned `ac_int`. By
    // comparing these two kernels, you will find shift operations of `ac_int`
    // can generate more efficient hardware if the amount to shift by is stored
    // in a minimally sized unsigned `ac_int`.
    {
      my_int14 a = t1, b = t2;
      my_uint2 _b = t2;
      my_int14 c, _c;
      TestShiftOp(q, a, b, c);
      TestEfficientShiftOp(q, a, _b, _c);

      if (c != _c) {
        std::cout << "Result mismatch!\n"
                  << "Kernel ShiftOp: result = " << c << "\n"
                  << "Kernel EfficientShiftOp: result = " << _c << "\n\n";
        passed = false;
      }
    }

    // Kernel `BitOps` demonstrates bit operations with bit select operator `[]`
    // and bit slice write operation `set_slc`. Note: An `ac_int` must be
    // initialized before being access by bit select operator `[]` and bit slice
    // operations `slc` and `set_slc`, otherwise it is undefined behavior and
    // will give you unexpected results.
    {
      my_int14 a = t1;
      my_int14 result = TestBitOps(q, a);

      int golden = 0b001111101111;

      if (result != golden) {
        std::cout << "Kernel BitOps result mismatch!\n"
                  << "result = 0b" << std::bitset<14>(result) << "\n"
                  << "golden = 0b" << std::bitset<14>(golden) << "\n\n";
        passed = false;
      }
    }
  } catch (exception const &e) {
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

  if (passed) {
    std::cout << "PASSED: all kernel results are correct.\n";
  } else {
    std::cout << "FAILED\n";
  }
  return passed ? 0 : 1;
}
