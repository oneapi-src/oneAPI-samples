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

///////////////////////////////////////////////////////////
// Simple 1D and 2D pipe arrays
template <class Id, typename T, size_t depth=0>
struct PipeArray {
  PipeArray() = delete;

  template <size_t idx>
  struct PipeId;

  template <size_t idx>
  using pipe = sycl::INTEL::pipe<PipeId<idx>, T, depth>;
};

template <class Id, typename T, size_t depth=0>
struct PipeArray2D {
  PipeArray2D() = delete;

  template <size_t x, size_t y>
  struct PipeId;

  template <size_t x, size_t y>
  using pipe = sycl::INTEL::pipe<PipeId<x,y>, T, depth>;
};
///////////////////////////////////////////////////////////

#endif /* __MISC_HPP__ */