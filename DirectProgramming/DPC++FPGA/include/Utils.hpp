#pragma once

/*
  Static implementation of the base 2 logarithm function
*/
template <typename T>
static constexpr T Log2(T n) {
  T ret = T(0);
  T val = n;
  while (val > T(1)) {
    val >>= 1;
    ret++;
  }
  return ret;
}

/*
  Static implementation of the CEIL base 2 logarithm function
*/
template <unsigned int N, uint8_t remains = 0>
static constexpr inline unsigned int CeilLog2() {
  return (N <= 1) ? remains : 1 + CeilLog2<(N >> 1), remains | (N % 2)>();
}

/*
  Static implementation of the base 2 power function
*/
template <typename T>
static constexpr T Pow2(T n) {
  return T(1) << n;
}

/*
  Return the number of bits required to encode all the values between 0 and N
*/
template <unsigned int N>
static constexpr inline unsigned int BitsForMaxValue() {
  return CeilLog2<N + 1>();
}

/*
  A structure that holds a table a of count elements of type T.
*/
template <unsigned count, typename T>
struct pipeTable {
  T elem[count];

  template <int idx>
  T get() {
    return elem[idx];
  }

  template <int idx>
  void set(T &in) {
    elem[idx] = in;
  }
};