#ifndef __SHANNONITERATOR_HPP__
#define __SHANNONITERATOR_HPP__
#pragma once

#include <array>
#include <type_traits>

#include "Tuple.hpp"
#include "Unroller.hpp"

//
// This class is intended to improve iterator performance for the FPGA.
// It optimizes the critical by precomputing the iterator change and comparison
// and stores the results in a shift register.
// This process is called 'Shannonization'.
//
// Templated on the following variables:
//    CounterType         - the datatype to use for the counter
//                          default = size_t
//    shift_register_size - the size of the shift register to use
//    step_size           - the amount to change the counters on each step (e.g.
//                          i += step_size)
//                        default = 1
//    increment           - boolean where true=increment, false=decrement
//                        default = true
//    inclusive           - whether the end of the range is include or not (e.g.
//                        i < 100 vs i <= 100)
//                        default = false
//
template <typename CounterType, int shift_register_size,
          CounterType step_size = 1, bool increment = true,
          bool inclusive = false>
class ShannonIterator {
  // static asserts
  static_assert(std::is_integral<CounterType>::value,
                "CounterType must be an integral type");
  static_assert(shift_register_size > 1,
                "shift_register_size must great than"
                "one (otherwise just use a normal iterator!)");
  static_assert(step_size > 0,
                "step_size must great than 0"
                "step_size==0: does no work,"
                "step_size<0: use the increment template instead");

 public:
  //
  // Constructor
  // Counts from start...end, inclusive/exclusive depends on inclusive template
  //
  ShannonIterator(CounterType start, CounterType end) : end_(end) {
    // initialize the counter shift register
    UnrolledLoop<0, kCountShiftRegisterSize>([&](auto i) {
      if (increment) {
        counter_.template get<i>() = start + (i * step_size);
      } else {
        counter_.template get<i>() = start - (i * step_size);
      }
    });

    // initialize the in range boolean shift register
    UnrolledLoop<0, kInRangeShiftRegisterSize>([&](auto i) {
      CounterType val = counter_.template get<i>();

      // depends on increment and inclusive template parameters,
      // which are determined at compile time)
      if (increment) {
        inrange_.template get<i>() = inclusive ? (val <= end_) : (val < end_);
      } else {
        inrange_.template get<i>() = inclusive ? (val >= end_) : (val > end_);
      }
    });
  }

  //
  // move the iterator, return whether the counter is in range
  //
  bool Step() {
    // shift the in range checks
    UnrolledLoop<0, kInRangeShiftRegisterSize - 1>([&](auto i) {
      inrange_.template get<i>() = inrange_.template get<i + 1>();
    });

    // compute the new in range check
    if (increment) {
      inrange_.last() =
          inclusive ? (counter_.last() <= end_) : (counter_.last() < end_);
    } else {
      inrange_.last() =
          inclusive ? (counter_.last() >= end_) : (counter_.last() > end_);
    }

    // shift the counters
    UnrolledLoop<0, shift_register_size - 1>([&](auto i) {
      counter_.template get<i>() = counter_.template get<i + 1>();
    });

    // compute the new counter
    if (increment) {
      counter_.last() += step_size;
    } else {
      counter_.last() -= step_size;
    }

    // return whether the index is currently in range (post-increment)
    return InRange();
  }

  //
  // returns whether the given counter is in range
  // defaults to the head counter
  //
  template <int i = 0>
  bool InRange() {
    static_assert(i < kInRangeShiftRegisterSize, "i out of range");
    return inrange_.template get<i>();
  }

  //
  // returns the given counter value
  // defaults to the head counter
  //
  template <int i = 0>
  CounterType Index() {
    static_assert(i < kCountShiftRegisterSize, "i out of range");
    return counter_.template get<i>();
  }

 private:
  static constexpr int kCountShiftRegisterSize = shift_register_size;
  static constexpr int kInRangeShiftRegisterSize = shift_register_size - 1;

  // the shift register of counters
  NTuple<shift_register_size, CounterType> counter_;

  // the shift register of in range checks
  NTuple<kInRangeShiftRegisterSize, bool> inrange_;

  // the ending value
  CounterType end_;
};

#endif /* __SHANNONITERATOR_HPP__ */
