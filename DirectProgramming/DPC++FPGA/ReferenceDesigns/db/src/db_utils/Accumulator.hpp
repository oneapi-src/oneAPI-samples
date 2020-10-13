#ifndef __ACCUMULATOR_HPP__
#define __ACCUMULATOR_HPP__
#pragma once

#include <limits>
#include <type_traits>

#include "Tuple.hpp"
#include "Unroller.hpp"

///////////////////////////////////////////////////////
//
// Register-based accumulator
// Accumulates into Size bins of type StorageType
//
template <typename StorageType, int Size, typename IndexType = unsigned int>
class RegisterAccumulator {
  // static asserts
  static_assert(std::is_arithmetic<StorageType>::value,
                "StorageType must be arithmetic to support accumulation");
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integral type");
  static_assert(std::numeric_limits<IndexType>::max() >= (Size - 1),
                "IndexType must be large enough to index the entire array");

 public:
  // initialize the memory to 0
  void Init() {
    UnrolledLoop<0, Size>([&](auto i) { 
      registers.template get<i>() = 0;
    });
  }

  // accumulate 'value' into register 'index' (i.e. registers[index] += value)
  void Accumulate(IndexType index, StorageType value) {
    UnrolledLoop<0, Size>([&](auto i) {
      registers.template get<i>() += (i == index) ? value : 0;
    });
  }

  // template version of accumulate
  template <IndexType INDEX>
  void Accumulate(StorageType value) {
    static_assert(INDEX < Size, "INDEX is out of range");
    registers.template get<INDEX>() += value;
  }

  // get the value of memory at 'index'
  StorageType Get(IndexType index) {
    StorageType ret;
    UnrolledLoop<0, Size>([&](auto i) {
      if (i == index) {
        ret = registers.template get<i>();
      }
    });

    return ret;
  }

  // template version of get
  template <IndexType INDEX>
  StorageType Get() {
    static_assert(INDEX < Size, "INDEX is out of range");
    return registers.template get<INDEX>();
  }

  // register storage using tuples
  NTuple<Size, StorageType> registers;
};
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
//
// BRAM-based accumulator
// Accumulates into Size bins of type StorageType
//
template <typename StorageType, int Size, int CacheSize,
          typename IndexType = unsigned int>
class BRAMAccumulator {
  // static asserts
  static_assert(std::is_arithmetic<StorageType>::value,
                "StorageType must be arithmetic to support accumulation");
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integral type");
  static_assert(std::numeric_limits<IndexType>::max() >= (Size - 1),
                "IndexType must be large enough to index the entire array");

 public:
  // initialize the memory entries
  void Init() {
    // initialize the memory entries
    for (IndexType i = 0; i < Size; i++) {
      mem[i] = 0;
    }

// initialize the cache
#pragma unroll
    for (IndexType i = 0; i < CacheSize + 1; i++) {
      cache_value[i] = 0;
      cache_tag[i] = 0;
    }
  }

  // accumulate 'value' into register 'index' (i.e. registers[index] += value)
  void Accumulate(IndexType index, StorageType value) {
    // get value from memory
    StorageType currVal = mem[index];

// check if value is in cache
#pragma unroll
    for (IndexType i = 0; i < CacheSize + 1; i++) {
      if (cache_tag[i] == index) {
        currVal = cache_value[i];
      }
    }

    // write the new value to both the shift register cache and the local mem
    const StorageType newVal = currVal + value;
    mem[index] = cache_value[CacheSize] = newVal;
    cache_tag[CacheSize] = index;

// Cache is just a shift register, so shift it
// pushing into back of the shift register done above
#pragma unroll
    for (IndexType i = 0; i < CacheSize; i++) {
      cache_value[i] = cache_value[i + 1];
      cache_tag[i] = cache_tag[i + 1];
    }
  }

  // get the value of memory at 'index'
  StorageType Get(IndexType index) {
    return mem[index];
  }

  // internal storage
  StorageType mem[Size];

  // internal cache for hiding write latency
  [[intelfpga::register]]
  StorageType cache_value[CacheSize + 1];

  [[intelfpga::register]]
  IndexType cache_tag[CacheSize + 1];
};
///////////////////////////////////////////////////////

#endif /* __ACCUMULATOR_HPP__ */
