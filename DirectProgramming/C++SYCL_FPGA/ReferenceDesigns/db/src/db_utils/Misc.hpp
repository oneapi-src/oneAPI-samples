#ifndef __MISC_HPP__
#define __MISC_HPP__
#pragma once

#include <type_traits>

//// Some useful math functions

//
// computes 2^n where 'n' is a compile time constant
//
template <typename T>
static constexpr T Pow2(T n) {
  return T(1) << n;
}

//
// base-2 logarithm
//
template <typename T>
static constexpr T Log2(T n) {
  return ((n < 2) ? T(0) : T(1) + Log2(n / 2));
}

//
// round up Log2
//
template <typename T>
static constexpr T CeilLog2(T n) {
  return ((n == 1) ? T(0) : Log2(n - 1) + T(1));
}

//
// Count the number of bits set to '1'
//
template <typename T>
constexpr T CountOnes(T val) {
  T count = T(0);

  while (val != 0) {
    count += val & 1;
    val >>= 1;
  }

  return count;
};

//
// Find the position of the Nth '1' (starting at 1)
// E.g. (11 = 4'b1011)
//  PositionOfNthOne(1, 11) = 1
//  PositionOfNthOne(2, 11) = 2
//  PositionOfNthOne(3, 11) = 4
//
template <typename T>
constexpr T PositionOfNthOne(T n, T bits) {
  T set = T(0);
  T pos = T(0);
  while (set < n) {
    set += bits & 1;
    bits = bits >> 1;

    pos++;
  }
  return pos;
};

#endif /* __MISC_HPP__ */
