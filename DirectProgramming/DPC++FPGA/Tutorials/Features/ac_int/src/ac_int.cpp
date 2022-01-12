// clang-format off
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
// clang-format on

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel name in the global scope.
// This is a FPGA best practice that reduces name mangling in the optimization
// reports.
class Add;
class Div;
class Mult;
class ShiftLeft;
class EfficientShiftLeft;
class GetBitSlice;
class SetBitSlice;

using ac_int4 = ac_intN::int4;
using ac_int14 = ac_intN::int14;
using ac_int15 = ac_intN::int15;
using ac_int28 = ac_intN::int28;
using ac_uint4 = ac_intN::uint4;

constexpr int thirteen_bits_mask = (1 << 13) - 1;
constexpr int fourteen_bits_mask = (1 << 14) - 1;

void TestAdd(queue &q, const ac_int14 &a, const ac_int14 &b, ac_int15 &c) {
  buffer<ac_int14, 1> inp1(&a, 1);
  buffer<ac_int14, 1> inp2(&b, 1);
  buffer<ac_int15, 1> result(&c, 1);

  q.submit([&](handler &h) {
    accessor x{inp1, h, read_only};
    accessor y{inp2, h, read_only};
    accessor res{result, h, write_only, no_init};
    h.single_task<Add>([=] { res[0] = x[0] + y[0]; });
  });
}

void TestDiv(queue &q, const ac_int14 &a, const ac_int14 &b, ac_int15 &c) {
  buffer<ac_int14, 1> inp1(&a, 1);
  buffer<ac_int14, 1> inp2(&b, 1);
  buffer<ac_int15, 1> result(&c, 1);

  q.submit([&](handler &h) {
    accessor x{inp1, h, read_only};
    accessor y{inp2, h, read_only};
    accessor res{result, h, write_only, no_init};
    h.single_task<Div>([=] { res[0] = x[0] / y[0]; });
  });
}

void TestMult(queue &q, const ac_int14 &a, const ac_int14 &b, ac_int28 &c) {
  buffer<ac_int14, 1> inp1(&a, 1);
  buffer<ac_int14, 1> inp2(&b, 1);
  buffer<ac_int28, 1> result(&c, 1);

  q.submit([&](handler &h) {
    accessor x{inp1, h, read_only};
    accessor y{inp2, h, read_only};
    accessor res{result, h, write_only, no_init};
    h.single_task<Mult>([=] { res[0] = x[0] * y[0]; });
  });
}

void TestShiftLeft(queue &q, const ac_int14 &a, const ac_int14 &b,
                   ac_int14 &c) {
  buffer<ac_int14, 1> inp1(&a, 1);
  buffer<ac_int14, 1> inp2(&b, 1);
  buffer<ac_int14, 1> result(&c, 1);

  q.submit([&](handler &h) {
    accessor x{inp1, h, read_only};
    accessor y{inp2, h, read_only};
    accessor res{result, h, write_only, no_init};
    h.single_task<ShiftLeft>([=] { res[0] = x[0] << y[0]; });
  });
}

// Note how the shift amount is specified with a smaller ac_int than in
// TestShiftLeft above
void TestEfficientShiftLeft(queue &q, const ac_int14 &a, const ac_uint4 &b,
                            ac_int14 &c) {
  buffer<ac_int14, 1> inp1(&a, 1);
  buffer<ac_uint4, 1> inp2(&b, 1);
  buffer<ac_int14, 1> result(&c, 1);

  q.submit([&](handler &h) {
    accessor x{inp1, h, read_only};
    accessor y{inp2, h, read_only};
    accessor res{result, h, write_only, no_init};
    h.single_task<EfficientShiftLeft>([=] { res[0] = x[0] << y[0]; });
  });
}

// The method
//      x = y.slc<M>(n)
// is equivalent to the VHDL behavior of
//      x := y((M+n-1) downto n);
// Note that only static bit widths are supported
void TestGetBitSlice(queue &q, const ac_int14 &a, ac_uint4 &b, const int lsb) {
  buffer<ac_int14, 1> inp1(&a, 1);
  buffer<int, 1> inp2(&lsb, 1);
  buffer<ac_uint4, 1> result(&b, 1);

  q.submit([&](handler &h) {
    accessor x{inp1, h, read_only};
    accessor y{inp2, h, read_only};
    accessor res{result, h, write_only, no_init};
    h.single_task<GetBitSlice>([=] { res[0] = x[0].slc<4>(y[0]); });
  });
}

// There is a set_slc(int lsb, const ac_int<W, S> &slc) which allows the user to
// set a bit slice as shown in the example below.
void TestSetBitSlice(queue &q, ac_int14 &a, const ac_int4 &b, const int lsb) {
  buffer<ac_int14, 1> buff_a(&a, 1);
  buffer<ac_int4, 1> buff_b(&b, 1);
  buffer<int, 1> buff_lsb(&lsb, 1);

  q.submit([&](handler &h) {
    accessor x{buff_a, h, write_only, no_init};
    accessor y{buff_b, h, read_only};
    accessor lsb_accessor{buff_lsb, h, read_only};

    h.single_task<SetBitSlice>([=] {
      // The set_slc method does not need to have a width specified as a
      // template argument since the width is inferred from the width of the
      // argument x
      x[0].set_slc(lsb_accessor[0], y[0]);

      // Bits can also be individually set as follows:
      x[0][3] = 0;
      x[0][2] = 0;
      x[0][1] = 0;
      x[0][0] = 0;
    });
  });
}

int main() {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif
  bool passed = true;

  try {
    // create the SYCL device queue
    queue q(selector, dpc_common::exception_handler);

    // Use a fixed initial seed
    srand(123);

    // Initialize two random ints
    int t1 = rand();
    int t2 = rand();
    // Truncate each of the two ints to 13 bits. The 14th bit is the sign bit.
    // In this testbench, even though the datatypes are signed, we will be
    // storing unsigned values to make the testing process simpler.
    t1 &= thirteen_bits_mask;
    t2 &= thirteen_bits_mask;

    std::cout << "Arithmetic Operations:\n";
    // Test adder
    {
      // ac_int offers type casting from and to native datatypes
      ac_int14 a = t1;
      ac_int14 b = t2;
      ac_int15 c;
      TestAdd(q, a, b, c);
      int c_golden = t1 + t2;
      // We can check the result of the ac_int addition with the native C int
      // addition
      if (c != c_golden) {
        passed = false;
        std::cerr << "Addition failed\n";
      }
      std::cout << "ac_int: " << a << " + " << b << " = " << c << "\n";
      std::cout << "int:    " << t1 << " + " << t2 << " = " << c_golden << "\n";
    }

    // Test multiplier
    {
      t1 = rand() & thirteen_bits_mask;
      ac_int14 a = t1;
      ac_int14 b = t2;
      ac_int28 c;
      TestMult(q, a, b, c);
      int c_golden = t1 * t2;
      // We can check the result of the ac_int multiplication with the native C
      // int multiplication
      if (c != c_golden) {
        passed = false;
        std::cerr << "Multiplier failed\n";
      }
      std::cout << "ac_int: " << a << " * " << b << " = " << c << "\n";
      std::cout << "int:    " << t1 << " * " << t2 << " = " << c_golden << "\n";
    }

    // Test divider
    {
      t1 = rand() & thirteen_bits_mask;
      t2 = rand() % 50;  // Use a small value for the divisor so that the result
                         // is not 0 or 1
      ac_int14 a = t1;
      ac_int14 b = t2;
      ac_int15 c;
      TestDiv(q, a, b, c);
      int c_golden = t1 / t2;
      // We can check the result of the ac_int division with the native C
      // int division
      if (c != c_golden) {
        passed = false;
        std::cerr << "divider failed\n";
      }
      std::cout << "ac_int: " << a << " / " << b << " = " << c << "\n";
      std::cout << "int:    " << t1 << " / " << t2 << " = " << c_golden << "\n";
    }

    std::cout << "\nBitwise Operations:\n";
    // Shift operator
    {
      t1 = rand() & thirteen_bits_mask;
      t2 = rand() % 8;  // Use a small value for the shift
      ac_int14 a = t1;
      ac_int14 b = t2;
      ac_int14 c;
      // b can be positive or negative. If (b > 0), a will be shifted to the
      // left. Else, a will be shifted to the right
      TestShiftLeft(q, a, b, c);

      // Note that the left shift in ac_int is logical so to check the result,
      // we need to do a little bit manipulation to get the correct signed value
      int c_golden = (t1 << t2) & fourteen_bits_mask;
      if ((t1 << t2) & (1 << 13)) c_golden |= (~fourteen_bits_mask);
      if (c != c_golden) {
        passed = false;
        std::cerr << "left_shift failed\n";
      }
      std::cout << "ac_int: " << a << " << " << b << " = " << c << "\n";
      std::cout << "int:    " << t1 << " << " << t2 << " = " << c_golden
                << "\n";
    }

    // Efficient left shift operator
    {
      t1 = rand() & thirteen_bits_mask;
      t2 = rand() % 14;  // Use a small value for the shift
      ac_int14 a = t1;
      ac_uint4 b = t2;
      ac_int14 c;
      // b is always positive in this case. This shift is more efficient in HW
      // than the one above. If the direction of the shift is known at compile
      // time, this is recommended. Note that the two datatypes need not be the
      // same. Note that the datatype of b here is just 4 bits since 4 bits can
      // completely contain the value of the full width of a. This will generate
      // a more efficient datapath.
      TestEfficientShiftLeft(q, a, b, c);

      // Note that the left shift in ac_int is logical so to check the result,
      // we need to do a little bit manipulation to get the correct signed value
      int c_golden = (t1 << t2) & fourteen_bits_mask;
      if ((t1 << t2) & (1 << 13)) c_golden |= (~fourteen_bits_mask);
      if (c != c_golden) {
        passed = false;
        std::cerr << "efficient_left_shift failed\n";
      }
      std::cout << "ac_int: " << a << " << " << b << " = " << c << "\n";
      std::cout << "int:    " << t1 << " << " << t2 << " = " << c_golden
                << "\n";
    }

    // Slice operations
    {
      t1 = rand() & thirteen_bits_mask;
      ac_int14 a = t1;
      ac_uint4 b;
      TestGetBitSlice(q, a, b, 5);

      // Replicate the same operation using bitwise operation
      t2 = (t1 >> 5) & 0xF;
      // Compare the CPU result with the FPGA result
      if (b != t2) {
        passed = false;
        std::cerr << "GetBitSlice failed\n";
      }
      std::cout << "(" << a << ").slc<4>(5) = " << b << "\n";

      int t2 = 10;
      ac_uint4 d = t2;
      ac_int14 c = a;
      // Sets the bits (3, 2, 1, 0) as 0 and sets bits (9, 8, 7, 6) with d
      TestSetBitSlice(q, c, d, 6);

      // Replicate the same operation using bitwise operation
      int mask = -1;
      mask ^= (0xF << 6) | (0xF);
      int mask2 = (t2 << 6);
      int t3 = (t1 & mask) | mask2;
      // Compare the CPU result with the FPGA result
      if (c != t3) {
        passed = false;
        std::cerr << "SetBitSlice failed\n";
      }
      std::cout << "Running these two ops on " << a << "\n";
      std::cout << "\t(" << a << ").set_slc(6, " << d << ") = " << c << "\n";
      std::cout << "\ta[3] = 0; a[2] = 0; a[1] = 0; a[0] = 0;\n";
      std::cout << "\tResult = " << c << "\n";
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

  if (passed) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    return 1;
  }

  return 0;
}
