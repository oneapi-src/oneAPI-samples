#ifndef __MISC_HPP__
#define __MISC_HPP__

template <typename T>
static constexpr T Pow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
  return T(1) << n;
}

template<typename T>
constexpr bool IsPow2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
  return (n != 0) && ((n & (n - 1)) == 0);
}

template <typename T>
constexpr T Log2(T n) {
  static_assert(std::is_integral<T>::value);
  static_assert(std::is_unsigned<T>::value);
  return ((n < 2) ? T(0) : T(1) + Log2(n / 2));
}

// return 'n' rounded up to the nearest power of 2
template<typename T>
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

#endif /* __MISC_HPP__ */