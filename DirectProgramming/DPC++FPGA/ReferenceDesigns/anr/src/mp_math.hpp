#ifndef __MP_MATH__
#define __MP_MATH__

namespace hldutils {

template <typename T>
constexpr T Min(T a, T b) {
  return (a < b) ? a : b;
}
template <typename T>
constexpr T Max(T a, T b) {
  return (a > b) ? a : b;
}

// returns n^2
template <typename T>
constexpr T Pow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
  return T(1) << n;
}

// returns whether 'n' is a power of 2
template <typename T>
constexpr bool IsPow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
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

// return 'n' rounded up to the nearest power of 2
template <typename T>
constexpr T RoundUpPow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
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

}  // namespace hldutils

#endif /* __MP_MATH__ */