// clang-format off
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed_math.hpp>
// clang-format on

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

using fixed_10_3_t = ac_fixed<10, 3, true>;
using fixed_9_2_t = ac_fixed<9, 2, true>;

// Forward declare the kernel name in the global scope.
// This is a FPGA best practice that reduces name mangling in the reports.
class ConstructFromFloat;
class ConstructFromACFixed;
class CalculateWithFloat;
class CalculateWithACFixed;

// Not recommended Usage example:
// Convert dynamic float value inside the kernel
void TestConstructFromFloat(queue &q, const float &a,
                            ac_fixed<20, 10, true, AC_RND, AC_SAT> &b) {
  buffer<float, 1> inp_buffer(&a, 1);
  buffer<ac_fixed<20, 10, true, AC_RND, AC_SAT>, 1> ret_buffer(&b, 1);

  q.submit([&](handler &h) {
    accessor in_acc{inp_buffer, h, read_only};
    accessor out_acc{ret_buffer, h, write_only, no_init};

    h.single_task<ConstructFromFloat>([=] {
      ac_fixed<20, 10, true, AC_RND, AC_SAT> t(in_acc[0]);
      ac_fixed<20, 10, true, AC_RND, AC_SAT> some_offset(0.5f);
      out_acc[0] = t + some_offset;
    });
  });
}

// Recommended Usage example:
// Convert dynamic float value outside the kernel
void TestConstructFromACFixed(queue &q,
                              const ac_fixed<20, 10, true, AC_RND, AC_SAT> &a,
                              ac_fixed<20, 10, true, AC_RND, AC_SAT> &ret) {
  buffer<ac_fixed<20, 10, true, AC_RND, AC_SAT>, 1> inp_buffer(&a, 1);
  buffer<ac_fixed<20, 10, true, AC_RND, AC_SAT>, 1> ret_buffer(&ret, 1);

  q.submit([&](handler &h) {
    accessor in_acc{inp_buffer, h, read_only};
    accessor out_acc{ret_buffer, h, write_only};

    h.single_task<ConstructFromACFixed>([=] {
      ac_fixed<20, 10, true, AC_RND, AC_SAT> t(in_acc[0]);
      ac_fixed<20, 10, true, AC_RND, AC_SAT> some_offset(0.5f);
      out_acc[0] = t + some_offset;
    });
  });
}

void TestCalculateWithFloat(queue &q, const float &x, float &ret) {
  buffer<float, 1> inp_buffer(&x, 1);
  buffer<float, 1> ret_buffer(&ret, 1);

  q.submit([&](handler &h) {
    accessor x{inp_buffer, h, read_only};
    accessor res{ret_buffer, h, write_only, no_init};

    h.single_task<CalculateWithFloat>([=] {
      float sin_x = sinf(x[0]);
      float cos_x = cosf(x[0]);
      res[0] = sqrtf(sin_x * sin_x + cos_x * cos_x);
    });
  });
}

// clang-format off
// Please refer to ac_fixed_math.hpp header file for fixed point math
// functions' type deduction rule. In this case, following those rules:
// I, W, S are input type template parameter (ac_fixed<W, I, S>)
// rI, rW, rS are output type template parameter (ac_fixed<rW, rI, rS>)
//* Function Name         Type Propagation Rule
//* sqrt_fixed             rI = I, rW = W, rS = S
//* sin_fixed              For signed (S == true), rI == 2, rW =  W - I + 2;
//*                        For unsigned (S == false), I == 1, rW =  W - I + 1
//* cos_fixed              For signed (S == true), rI == 2, rW =  W - I + 2;
//*                        For unsigned (S == false), I == 1, rW =  W - I + 1
// clang-format on
void TestCalculateWithACFixed(queue &q, const fixed_10_3_t &x,
                              fixed_9_2_t &ret) {
  buffer<fixed_10_3_t, 1> inp_buffer(&x, 1);
  buffer<fixed_9_2_t, 1> ret_buffer(&ret, 1);

  q.submit([&](handler &h) {
    accessor x{inp_buffer, h, read_only};
    accessor res{ret_buffer, h, write_only, no_init};

    h.single_task<CalculateWithACFixed>([=] {
      fixed_9_2_t sin_x = sin_fixed(x[0]);
      fixed_9_2_t cos_x = cos_fixed(x[0]);
      // The RHS expression evaluates to a larger `ac_fixed`, which gets
      // truncated to fit in res[0].
      res[0] = sqrt_fixed(sin_x * sin_x + cos_x * cos_x);
    });
  });
}

int main() {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  try {
    // Create the SYCL device queue
    queue q(selector, dpc_common::exception_handler);

    // I. Constructing `ac_fixed` Numbers
    // Set quantization mode `AC_RND` to round towards plus infinity
    // Set overflow mode `AC_SAT` to saturate to max and min when overflow happens
    ac_fixed<20, 10, true, AC_RND, AC_SAT> a;
    TestConstructFromFloat(q, 3.1415f, a);
    std::cout << "Constructed from float:\t\t" << a << "\n";

    ac_fixed<20, 10, true, AC_RND, AC_SAT> b = 3.1415f;
    ac_fixed<20, 10, true, AC_RND, AC_SAT> c;
    TestConstructFromACFixed(q, b, c);
    std::cout << "Constructed from ac_fixed:\t" << c << "\n\n";

    // II. Using `ac_fixed` Math Functions
    constexpr int kSize = 5;
    constexpr float inputs[kSize] = {-0.807991899423f, -2.09982907558f,
                                     -0.742066235466f, -2.33217071676f,
                                     1.14324158042f};

    // quantum: the minimum positive value this type can represent
    // Quantum is 1 / 2 ^ (W - I), where W and I are the total width and the
    // integer width of the ac_fixed number
    constexpr fixed_9_2_t quantum = 0.0078125f;

    // for fixed point, the error should be less than 1 quantum of data type
    // (1 / 2^(W - I))
    constexpr fixed_9_2_t epsilon_fixed_9_2 = quantum;
    constexpr float epsilon_float = 1.0f / (1.0f * float(1 << 20));

    std::cout << "MAX DIFF for ac_fixed<10, 3, true>:  "
              << epsilon_fixed_9_2.to_double() << "\n";
    std::cout << "MAX DIFF for float:                  " << epsilon_float
              << "\n\n";

    bool pass = true;
    for (int i = 0; i < kSize; i++) {
      fixed_10_3_t fixed_type_input = inputs[i];
      float float_type_input = inputs[i];

      // declare output and diff variable
      fixed_9_2_t fixed_type_result;
      TestCalculateWithACFixed(q, fixed_type_input, fixed_type_result);
      float float_type_result;
      TestCalculateWithFloat(q, float_type_input, float_type_result);

      std::cout << "result(fixed point): " << fixed_type_result.to_double()
                << "\n";
      std::cout << "result(float):       " << float_type_result << "\n\n";

      // expected result is 1.0 = sqrt(sin^2(x) + cos^2(x))
      fixed_9_2_t diff1 = fabs(fixed_type_result.to_double() - 1.0);
      float diff2 = fabs(float_type_result - 1.0);

      if (diff1 > epsilon_fixed_9_2 || diff2 > epsilon_float) {
        pass = false;
      }
    }

    if (pass) {
      std::cout << "PASSED: all kernel results are correct.\n";
    } else {
      std::cout << "ERROR\n";
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

  return 0;
}
