#ifndef __LOCAL_MEMORY_CACHE_HPP__
#define __LOCAL_MEMORY_CACHE_HPP__

#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "constexpr_math.hpp"   // DirectProgramming/DPC++FPGA/include

namespace fpga_tools {

template <typename T,             // type to store in the memory
          size_t k_mem_depth,     // depth of the memory
          size_t k_cache_depth    // number of elements in the cache
         >
class LocalMemoryCache {
 public:
  static constexpr int kNumAddrBits = fpga_tools::CeilLog2(k_mem_depth);
  using addr_t = ac_int<kNumAddrBits, false>;

  LocalMemoryCache() {
    for (int i = 0; i < k_cache_depth; i++) {
      cache_valid_[i] = false;
    }
  }
  
  LocalMemoryCache(const LocalMemoryCache&) = delete;
  LocalMemoryCache& operator=(const LocalMemoryCache&) = delete;

  // add a value into the cache
  void AddToCache(addr_t addr, T val, bool valid = true) {
    // Shift the values in the cache
    #pragma unroll
    for (int i = 0; i < k_cache_depth-1; i++) {
      if (cache_addr_[i+1] == addr && valid) {
        cache_valid_[i] = false;    // invalidate old cache entry at same addr
      } else {
        cache_valid_[i] = cache_valid_[i + 1];
      }
      cache_val_[i] = cache_val_[i + 1];
      cache_addr_[i] = cache_addr_[i + 1];
    }

    // write the new value into the cache
    cache_val_[k_cache_depth-1] = val;
    cache_addr_[k_cache_depth-1] = addr;
    cache_valid_[k_cache_depth-1] = valid;
  }

  // check if a given address is in the cache, return true on a cache hit
  // val is updated only on a cache hit
  bool CheckCache(addr_t addr, T &val) { 
    
    bool return_val = false;

    // check the cache, newest value will take precedence
    #pragma unroll
    for (int i = 0; i < k_cache_depth; i++) {
      if ((cache_addr_[i] == addr) && (cache_valid_[i])) {
        val = cache_val_[i];
        return_val = true;
      } 
    }

    return return_val;
  }

  // use a template parameter here for index to avoid generating a mux
  template <size_t index> bool GetCacheVal( addr_t &addr, T &val) {
    val = cache_val_[index];
    addr = cache_addr_[index];
    return cache_valid_[index];
  }

 private:
  // the cache must be implemented in registers to allow simultaneous access
  // to all elements
  T cache_val_[k_cache_depth];
  addr_t cache_addr_[k_cache_depth];
  bool cache_valid_[k_cache_depth];
};

// specialization for cache size 0 (no cache)
template <typename T,             // type to store in the memory
          size_t k_mem_depth      // depth of the memory
         >
class LocalMemoryCache<T, k_mem_depth, 0> {
 public:
  static constexpr int kNumAddrBits = fpga_tools::CeilLog2(k_mem_depth);
  using addr_t = ac_int<kNumAddrBits, false>;

  LocalMemoryCache() {}

  LocalMemoryCache(const LocalMemoryCache&) = delete;
  LocalMemoryCache& operator=(const LocalMemoryCache&) = delete;

  void AddToCache(addr_t addr, T val, bool valid = true) {}
  bool CheckCache(addr_t addr, T &val) { return false; }
  template <size_t index> bool GetCacheVal( addr_t &addr, T &val) { 
    return false;
  }
};

}   // namespace fpga_tools

#endif  // __LOCAL_MEMORY_CACHE_HPP__