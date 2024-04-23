#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iomanip>  // for std::setprecision

#include "exception_handler.hpp"

using namespace sycl;

// Type aliases for ac_fixed types
using fixed_10_3_t = ac_fixed<10, 3, true>;
using fixed_9_2_t = ac_fixed<9, 2, true>;

// Quantization mode `AC_RND`: round towards plus infinity
// Overflow mode `AC_SAT`: saturate to max and min when overflow happens
// For the other quantization and overflow modes, refer to section 2.1 of
// "Algorithmic C (AC) Datatypes" document.
using fixed_20_10_t = ac_fixed<20, 10, true, AC_RND, AC_SAT>;

// Forward declare the kernel name in the global scope.
// This is a FPGA best practice that reduces name mangling in the reports.
class ConstructFromFloat;
class ConstructFromACFixed;
class CalculateWithFloat;
class CalculateWithACFixed;

// Not recommended Usage example:
// Convert dynamic float value inside the kernel
void TestConstructFromFloat(queue &q, const float &x, fixed_20_10_t &ret) {
  buffer<float, 1> inp_buffer(&x, 1);
  buffer<fixed_20_10_t, 1> ret_buffer(&ret, 1);

  q.submit([&](handler &h) {
    accessor in_acc{inp_buffer, h, read_only};
    accessor out_acc{ret_buffer, h, write_only, no_init};

    h.single_task<ConstructFromFloat>([=] {
      fixed_20_10_t t(in_acc[0]);
      fixed_20_10_t some_offset(0.5f);
      out_acc[0] = t + some_offset;
    });
  });
}

// Recommended Usage example:
// Convert dynamic float value outside the kernel
void TestConstructFromACFixed(queue &q, const fixed_20_10_t &x,
                              fixed_20_10_t &ret) {
  buffer<fixed_20_10_t, 1> inp_buffer(&x, 1);
  buffer<fixed_20_10_t, 1> ret_buffer(&ret, 1);

  q.submit([&](handler &h) {
    accessor in_acc{inp_buffer, h, read_only};
    accessor out_acc{ret_buffer, h, write_only};

    h.single_task<ConstructFromACFixed>([=] {
      fixed_20_10_t t(in_acc[0]);
      fixed_20_10_t some_offset(0.5f);
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
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // Create the SYCL device queue
    queue q(selector, fpga_tools::exception_handler);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // I. Constructing `ac_fixed` Numbers
    std::cout << "1. Testing Constructing ac_fixed from float or ac_fixed:\n";

    fixed_20_10_t a;
    TestConstructFromFloat(q, 3.1415f, a);
    std::cout << "Constructed from float:\t\t" << a << "\n";

    fixed_20_10_t b = 3.1415f;
    fixed_20_10_t c;
    TestConstructFromACFixed(q, b, c);
    std::cout << "Constructed from ac_fixed:\t" << c << "\n\n";

    // II. Using `ac_fixed` Math Functions
    std::cout
        << "2. Testing calculation with float or ac_fixed math functions:\n";

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

    std::cout << "MAX DIFF (quantum) for ac_fixed<10, 3, true>:\t"
              << epsilon_fixed_9_2.to_double()
              << "\nMAX DIFF for float:\t\t\t\t" << epsilon_float << "\n\n";

    bool pass = true;
    for (int i = 0; i < kSize; i++) {
      fixed_10_3_t fixed_type_input = inputs[i];
      float float_type_input = inputs[i];

      // declare output and diff variable
      fixed_9_2_t fixed_type_result;
      TestCalculateWithACFixed(q, fixed_type_input, fixed_type_result);
      float float_type_result;
      TestCalculateWithFloat(q, float_type_input, float_type_result);

      // expected result is 1.0 = sqrt(sin^2(x) + cos^2(x))
      double fixed_diff = abs(fixed_type_result.to_double() - 1.0);
      double float_diff = abs(float_type_result - 1.0);

      std::cout << std::setprecision(8);
      std::cout << "Input " << i << ":\t\t\t" << inputs[i]
                << "\nresult(fixed point):\t\t" << fixed_type_result.to_double()
                << "\ndifference(fixed point):\t" << fixed_diff
                << "\nresult(float):\t\t\t" << float_type_result
                << "\ndifference(float):\t\t" << float_diff << "\n\n";

      // check differences
      if (fixed_diff > epsilon_fixed_9_2 || float_diff > epsilon_float) {
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
