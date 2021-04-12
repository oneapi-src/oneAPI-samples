#ifndef __PARALLEL_COPY_ARRAY_HPP__
#define __PARALLEL_COPY_ARRAY_HPP__

#include "UnrolledLoop.hpp"

// ParallelCopyArray
// Defines a struct with a single element data, which is an array of type T.
// Defies the copy and = operators to do an unrolled (parallel) assignment of
// all elements in the array.  Defines the [] operator so the struct can be
// accessed like a normal array.
template <typename T,         // type of elements in the array
          std::size_t k_size  // number of T elements in the array
          >
struct ParallelCopyArray {
  // constructor
  ParallelCopyArray() {}

  // copy constructor - do a parallel copy
  ParallelCopyArray(const ParallelCopyArray& source) {
    UnrolledLoop<k_size>([&](auto k) { data[k] = source[k]; });
  }

  // assignment operator - do a parallel copy
  ParallelCopyArray& operator=(const ParallelCopyArray& source) {
    UnrolledLoop<k_size>([&](auto k) { data[k] = source[k]; });
    return *this;
  }

  // data accessors
  T& operator[](std::size_t index) { return data[index]; }
  const T& operator[](std::size_t index) const { return data[index]; }

 private:
  T data[k_size];

};  // end of struct ParallelCopyArray

#endif  // ifndef __PARALLEL_COPY_ARRAY_HPP__
