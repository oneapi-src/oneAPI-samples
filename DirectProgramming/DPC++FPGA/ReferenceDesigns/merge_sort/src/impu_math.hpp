#ifndef __IMPU_MATH__
#define __IMPU_MATH__

namespace impu {
namespace math {

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
}  // namespace math
}  // namespace impu

#endif /* __IMPU_MATH__ */