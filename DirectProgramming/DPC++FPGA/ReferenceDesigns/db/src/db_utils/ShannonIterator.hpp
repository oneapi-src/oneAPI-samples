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
//    CounterType       - the datatype to use for the counter
//                        default = size_t
//    ShiftRegisterSize - the size of the shift register to use
//    StepSize          - the amount to change the counters on each step (e.g.
//                        i += StepSize)
//                        default = 1
//    Increment         - boolean where true=increment, false=decrement
//                        default = true
//    Inclusive         - whether the end of the range is include or not (e.g.
//                        i < 100 vs i <= 100)
//                        default = false
//
template <typename CounterType, int ShiftRegisterSize,
          CounterType StepSize = 1, bool Increment = true,
          bool Inclusive = false>
class ShannonIterator {
  // static asserts
  static_assert(std::is_integral<CounterType>::value,
                "CounterType must be an integral type");
  static_assert(ShiftRegisterSize > 1,
                "ShiftRegisterSize must great than"
                "one (otherwise just use a normal iterator!)");
  static_assert(StepSize > 0,
                "StepSize must great than 0"
                "StepSize==0: does no work,"
                "StepSize<0: use the Increment template instead");

 public:
  //
  // Constructor
  // Counts from start...end, inclusive/exclusive depends on Inclusive template
  //
  ShannonIterator(CounterType start, CounterType end) : end_(end) {
    // initialize the counter shift register
    UnrolledLoop<0, kCountShiftRegisterSize>([&](auto i) {
      if (Increment) {
        counter_.template get<i>() = start + (i * StepSize);
      } else {
        counter_.template get<i>() = start - (i * StepSize);
      }
    });

    // initialize the in range boolean shift register
    UnrolledLoop<0, kInRangeShiftRegisterSize>([&](auto i) {
      CounterType val = counter_.template get<i>();

      // depends on Increment and Inclusive template parameters,
      // which are determined at compile time)
      if (Increment) {
        inrange_.template get<i>() = Inclusive ? (val <= end_) : (val < end_);
      } else {
        inrange_.template get<i>() = Inclusive ? (val >= end_) : (val > end_);
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
    if (Increment) {
      inrange_.last() =
          Inclusive ? (counter_.last() <= end_) : (counter_.last() < end_);
    } else {
      inrange_.last() =
          Inclusive ? (counter_.last() >= end_) : (counter_.last() > end_);
    }

    // shift the counters
    UnrolledLoop<0, ShiftRegisterSize - 1>([&](auto i) {
      counter_.template get<i>() = counter_.template get<i + 1>();
    });

    // compute the new counter
    if (Increment) {
      counter_.last() += StepSize;
    } else {
      counter_.last() -= StepSize;
    }

    // return whether the index is currently in range (post-increment)
    return InRange();
  }

  //
  // returns whether the given counter is in range
  // defaults to the head counter
  //
  template <int Index = 0>
  bool InRange() {
    static_assert(Index < kInRangeShiftRegisterSize, "Index out of range");
    return inrange_.template get<Index>();
  }

  //
  // returns the given counter value
  // defaults to the head counter
  //
  template <int Index = 0>
  CounterType Index() {
    static_assert(Index < kCountShiftRegisterSize, "Index out of range");
    return counter_.template get<Index>();
  }

 private:
  static constexpr int kCountShiftRegisterSize = ShiftRegisterSize;
  static constexpr int kInRangeShiftRegisterSize = ShiftRegisterSize - 1;

  // the shift register of counters
  NTuple<ShiftRegisterSize, CounterType> counter_;

  // the shift register of in range checks
  NTuple<kInRangeShiftRegisterSize, bool> inrange_;

  // the ending value
  CounterType end_;
};

#endif /* __SHANNONITERATOR_HPP__ */
