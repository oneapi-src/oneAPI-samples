#ifndef __CACHED_LOCAL_MEMORY_HPP__
#define __CACHED_LOCAL_MEMORY_HPP__

#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "constexpr_math.hpp"   // DirectProgramming/DPC++FPGA/include

namespace fpga_tools {

template <typename T,             // type to store in the memory
          size_t k_mem_depth,     // depth of the memory
          size_t k_cache_depth    // number of elements in the cache
         >
class CachedLocalMemory {
 public:
  static constexpr int kNumAddrBits = fpga_tools::CeilLog2(k_mem_depth);
  using addr_t = ac_int<kNumAddrBits, false>;

  CachedLocalMemory() {}
  CachedLocalMemory(T init_val) { 
    for (int i = 0; i < k_mem_depth; i++) {
      write(i, init_val);
    }
  }
  
  CachedLocalMemory(const CachedLocalMemory&) = delete;
  CachedLocalMemory& operator=(const CachedLocalMemory&) = delete;

  // explicitly communicate to developers that we don't want to support a
  // square bracket operator that returns a reference, as it would allow
  // modification of the memory without updating the cache
  template <typename I>
  T& operator[](I addr) = delete;

  // we can support the square bracket operator that returns a const ref
  const T& operator[](addr_t addr) const { return read(addr); }

  void write(addr_t addr, T val) {
    // Shift the values in the cache
    #pragma unroll
    for (int i = 0; i < k_cache_depth-1; i++) {
      cache_val_[i] = cache_val_[i + 1];
      cache_addr_[i] = cache_addr_[i + 1];
    }

    // write the new value into the array and the cache
    data_[addr] = val;
    cache_val_[k_cache_depth-1] = val;
    cache_addr_[k_cache_depth-1] = addr;
  }

  T read(addr_t addr) { 
    // Get the value from the local memory
    auto val = data_[addr];

    // If this address was recently written, take the value from the cache
    #pragma unroll
    for (int i = 0; i < k_cache_depth; i++) {
      if (cache_addr_[i] == addr) val = cache_val_[i];
    }

    return val;
  }

  // data_ has to be public so we can use it in an ivdep attribute
  // User code must NEVER modify data_ directly, it must be treated as private
  T data_[k_mem_depth];

 private:
  // the cache must be implemented in registers to allow simultaneous access
  // to all elements
  [[intel::fpga_register]] T cache_val_[k_cache_depth];
  [[intel::fpga_register]] addr_t cache_addr_[k_cache_depth];
};

// specialization for cache size 0 (no cache)
template <typename T,             // type to store in the memory
          size_t k_mem_depth      // depth of the memory
         >
class CachedLocalMemory<T, k_mem_depth, 0> {
 public:
  static constexpr int kNumAddrBits = fpga_tools::CeilLog2(k_mem_depth);
  using addr_t = ac_int<kNumAddrBits, false>;

  CachedLocalMemory() {}
  CachedLocalMemory(T init_val) { 
    for (int i = 0; i < k_mem_depth; i++) {
      write(i, init_val);
    }
  }
  
  CachedLocalMemory(const CachedLocalMemory&) = delete;
  CachedLocalMemory& operator=(const CachedLocalMemory&) = delete;

  template <typename I>
  T& operator[](I addr) = delete;

  const T& operator[](addr_t addr) const { return read(addr); }

  void write(addr_t addr, T val) {
    data_[addr] = val;
  }

  T read(addr_t addr) { 
    auto val = data_[addr];
    return val;
  }

  // data_ has to be public so we can use it in an ivdep attribute
  // User code must NEVER modify data_ directly, it must be treated as private
  T data_[k_mem_depth];
};








}   // namespace fpga_tools

#endif  // __CACHED_LOCAL_MEMORY_HPP__