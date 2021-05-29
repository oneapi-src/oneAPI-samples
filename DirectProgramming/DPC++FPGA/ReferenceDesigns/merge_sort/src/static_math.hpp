#ifndef __STATICMATH_HPP__
#define __STATICMATH_HPP__

#define SWAP(a, b)  \
  do {              \
    auto tmp = (a); \
    (a) = (b);      \
    (b) = tmp;      \
  } while (0)

template <typename T>
static constexpr T Pow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
  return T(1) << n;
}

template <typename T>
constexpr bool IsPow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
  return (n != 0) && ((n & (n - 1)) == 0);
}

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

#endif /* __STATICMATH_HPP__ */