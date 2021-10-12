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
template<unsigned int N, uint8_t remains=0>
static constexpr inline unsigned int CeilLog2()
{
  return (N <= 1) ? remains : 1 + CeilLog2<(N>>1), remains | (N%2)>();
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
template<unsigned int N>
static constexpr inline unsigned int BitsForMaxValue()
{
  return CeilLog2<N+1>();
}

/*
  A structure that hold a column a of matrix of type T.
*/
template<unsigned rows, typename T>
struct column{
  T d[rows];
};

/*
  A structure that hold a row a of matrix of type T.
*/
template<unsigned columns, typename T>
struct row{
  // [[intel::fpga_memory("BLOCK_RAM")]] // NO-FORMAT: Attribute
  T d[columns];
};