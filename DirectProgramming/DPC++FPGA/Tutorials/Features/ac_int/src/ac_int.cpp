#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <bitset>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class BasicOpsInt;
class BasicOpsAcInt;
class ShiftOps;
class EfficientShiftOps;
class BitAccess;

using MyUInt2 = ac_int<2, false>;
using MyInt7 = ac_int<7, true>;
using MyInt14 = ac_int<14, true>;
using MyInt15 = ac_int<15, true>;
using MyInt28 = ac_int<28, true>;

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

// Kernel `BasicOpsAcInt` consumes fewer resources than kernel `BasicOpsInt`.
void TestBasicOpsAcInt(queue &q, const MyInt14 &a, const MyInt14 &b, MyInt15 &c,
                       MyInt28 &d, MyInt15 &e) {
  buffer<MyInt14, 1> a_buf(&a, 1);
  buffer<MyInt14, 1> b_buf(&b, 1);
  buffer<MyInt15, 1> c_buf(&c, 1);
  buffer<MyInt28, 1> d_buf(&d, 1);
  buffer<MyInt15, 1> e_buf(&e, 1);

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

void TestShiftOps(queue &q, const MyInt14 &a, const MyInt14 &b, MyInt14 &c) {
  buffer<MyInt14, 1> a_buf(&a, 1);
  buffer<MyInt14, 1> b_buf(&b, 1);
  buffer<MyInt14, 1> c_buf(&c, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor b_acc(b_buf, h, read_only);
    accessor c_acc(c_buf, h, write_only, no_init);
    h.single_task<ShiftOps>([=]() [[intel::kernel_args_restrict]] {
      MyInt14 temp = a_acc[0] << b_acc[0];
      c_acc[0] = temp >> b_acc[0];
    });
  });
}

// Kernel `EfficientShiftOps` consumes fewer resources than kernel `ShiftOps`.
void TestEfficientShiftOps(queue &q, const MyInt14 &a, const MyUInt2 &b,
                           MyInt14 &c) {
  buffer<MyInt14, 1> a_buf(&a, 1);
  buffer<MyUInt2, 1> b_buf(&b, 1);
  buffer<MyInt14, 1> c_buf(&c, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor b_acc(b_buf, h, read_only);
    accessor c_acc(c_buf, h, write_only, no_init);
    h.single_task<EfficientShiftOps>([=]() [[intel::kernel_args_restrict]] {
      MyInt14 temp = a_acc[0] << b_acc[0];
      c_acc[0] = temp >> b_acc[0];
    });
  });
}

MyInt14 TestBitAccess(queue &q, const MyInt14 &a) {
  MyInt14 res;
  buffer<MyInt14, 1> a_buf(&a, 1);
  buffer<MyInt14, 1> res_buf(&res, 1);

  q.submit([&](handler &h) {
    accessor a_acc(a_buf, h, read_only);
    accessor res_acc(res_buf, h, write_only, no_init);
    h.single_task<BitAccess>([=]() [[intel::kernel_args_restrict]] {
      // 0b1111101
      MyInt7 temp = a_acc[0].slc<7>(3);

      res_acc[0] = 0; // Must be initialized before being accessed by the bit
                      // select operator `[]`. Using the `[]` operator on an
                      // uninitialized `ac_int` variable is undefined behavior
                      // and can give you unexpected results.

      // 0 -> 0b1111101000
      res_acc[0].set_slc(3, temp);

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
    queue q(device_selector, dpc_common::exception_handler);

    constexpr int kVal1 = 1000, kVal2 = 2;

    {
      MyInt14 input_a = kVal1, input_b = kVal2;
      MyInt15 output_c;
      MyInt28 output_d;
      MyInt15 output_e;
      TestBasicOpsAcInt(q, input_a, input_b, output_c, output_d, output_e);

      int golden_c, golden_d, golden_e;
      TestBasicOpsInt(q, input_a, input_b, golden_c, golden_d, golden_e);

      if (output_c != golden_c || output_d != golden_d ||
          output_e != golden_e) {
        std::cout << "Result mismatch!\n"
                  << "Kernel BasicOpsInt:   addition = " << golden_c
                  << ", multiplication = " << golden_d
                  << ", division = " << golden_e << "\n"
                  << "Kernel BasicOpsAcInt: addition = " << output_c
                  << ", multiplication = " << output_d
                  << ", division = " << output_e << "\n\n";
        passed = false;
      }
    }

    {
      MyInt14 input_a = kVal1, input_b = kVal2;
      MyUInt2 input_efficient_b = kVal2;
      MyInt14 output_c, output_efficient_c;
      TestShiftOps(q, input_a, input_b, output_c);
      TestEfficientShiftOps(q, input_a, input_efficient_b, output_efficient_c);

      if (output_c != output_efficient_c) {
        std::cout << "Result mismatch!\n"
                  << "Kernel ShiftOps: result = " << output_c << "\n"
                  << "Kernel EfficientShiftOps: result = " << output_efficient_c
                  << "\n\n";
        passed = false;
      }
    }

    {
      MyInt14 input = kVal1;
      MyInt14 output = TestBitAccess(q, input);

      constexpr int kGolden = 0b001111101111;

      if (output != kGolden) {
        std::cout << "Kernel BitAccess result mismatch!\n"
                  << "result = 0b" << std::bitset<14>(output) << "\n"
                  << "golden = 0b" << std::bitset<14>(kGolden) << "\n\n";
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