#ifndef __CONSTEXPR_MATH__
#define __CONSTEXPR_MATH__

//
// This file contains various helper C++ metaprogramming math functions that
// are useful across various designs.
//

#include <limits>

namespace fpga_tools {

// returns the absolute value of 'x'
template <typename T>
constexpr T Abs(T x) { return (x < 0) ? -x : x; }

// returns the minimum of 'a' and 'b'.
// The type, 'T', must have an operator<
template <typename T>
constexpr T Min(T a, T b) { return (a < b) ? a : b; }

// returns the maximum of 'a' and 'b'.
// The type, 'T', must have an operator>
template <typename T>
constexpr T Max(T a, T b) { return (a > b) ? a : b; }

// rounds up to the nearest multiple of of 'multiple'
// only works for positive numbers
template <typename T>
constexpr T RoundUpToMultiple(T num, T multiple) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  if (multiple == 0) {
    return num;
  }

  int remainder = num % multiple;
  if (remainder == 0) {
    return num;
  } else {
    return num + multiple - remainder;
  }
}

// returns n^2
template <typename T>
constexpr T Pow2(T n) {
  static_assert(std::is_integral_v<T>);
  return (n < 0) ? (T(1) << (-n)) : (T(1) << n);
}

// returns whether abs(n) is a power of 2
template <typename T>
constexpr bool IsPow2(T n) {
  static_assert(std::is_integral_v<T>);
  if (n < 0) n = -n;
  return (n != 0) && ((n & (n - 1)) == 0);
}

// returns log2(n) rounding down
template <typename T>
constexpr T Log2(T n) {
  static_assert(std::is_integral_v<T>);
  if (n < 2) {
    return T(0);
  } else {
    T ret = 0;
    while (n >= 2) {
      ret++;
      n /= 2;
    }
    return ret;
  }
}

// returns log(2) rounded up
template <typename T>
static constexpr T CeilLog2(T n) {
  return ((n == 1) ? T(0) : Log2(n - 1) + T(1));
}


// returns the number of bits required to encode all the values between 0 and N
template <unsigned int n>
static constexpr unsigned int BitsForMaxValue() {
  return CeilLog2(n + 1);
}


// return 'n' rounded up to the nearest power of 2
template <typename T>
constexpr T RoundUpPow2(T n) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  if (n == 0) {
    return 2;
  } else if (IsPow2(n)) {
    return n;
  } else {
    return T(1) << (Log2(n) + 1);
  }
}

// computes x^y where y must be an integer (positive or negative)
constexpr double Pow(double x, int y) {
  if (y == 0) {
    // x^0 = 1
    return 1.0;
  } else {
    // handle both y < 0 and y > 0 by changing loop bound and multiply value
    bool y_is_negative = (y < 0);
    double mult_val = y_is_negative ? (1/x) : x;
    int loop_bound = y_is_negative ? -y : y;

    double ret = 1.0;
    for (int i = 0; i < loop_bound; i++) {
      ret *= mult_val;
    }
    return ret;
  }
}

// estimates e^(x) for x >= 0 using a taylor series expansion
// https://en.wikipedia.org/wiki/Taylor_series
constexpr double Exp(double x, unsigned taylor_terms=32) {
  double factorial = 1.0;
  double power = 1.0;
  double answer = 1.0;

  for(int i = 1; i < taylor_terms-1; i++) {
    power *= x;
    factorial *= i;
    answer += power / factorial;
  }
  return answer;
}

// Scale significand using floating-point base exponent
// see: http://www.cplusplus.com/reference/cmath/scalbn/
constexpr float Scalbn(float value, int exponent) {
  if (exponent == 0) {
    return value;
  } else {
    float ret = value;
    while(exponent != 0) {
      if (exponent > 0) {
        ret *= 2;
        exponent--;
      } else {
        ret /= 2;
        exponent++;
      }
    }
    return ret;
  }
}

// extract the exponent from a 32-bit float
constexpr int FP32ExtractExponent(float x) {
  if (x == 0) {
    return 0;
  } else {
    float ret = 0;
    float abs_x = Abs(x);
    while (abs_x >= 2 || abs_x < 1) {
      bool abs_x_gte_2 = (abs_x >= 2);
      ret += (abs_x_gte_2 ? 1 : -1);
      x = (abs_x_gte_2 ? (x/2) : (x*2));
      abs_x = Abs(x);
    }
    return ret;
  }
}

// extract the mantissa from a 32-bit float
constexpr int FP32ExtractMantissa(float x) {
  // remove hidden 1 and bias the exponent to get integer
  //#pragma clang fp contract(off)
  //return (Abs(x) < std::numeric_limits<float>::infinity()) ?
  //        Scalbn(Scalbn(Abs(x),-FP32ExtractExponent(x))-1,23) : 0;
  return Scalbn(Scalbn(Abs(x),-FP32ExtractExponent(x))-1,23);
}

}  // namespace fpga_tools

#endif /* __CONSTEXPR_MATH__ */
