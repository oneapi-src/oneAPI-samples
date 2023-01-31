#ifndef __ONCHIP_MEMORY_WITH_CACHE_HPP__
#define __ONCHIP_MEMORY_WITH_CACHE_HPP__

#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "constexpr_math.hpp"  // DirectProgramming/C++SYCL_FPGA/include
#include "unrolled_loop.hpp"   // DirectProgramming/C++SYCL_FPGA/include

namespace fpga_tools {

template <typename T,           // type to store in the memory
          size_t k_mem_depth,   // depth of the memory
          size_t k_cache_depth  // number of elements in the cache
          >
class OnchipMemoryWithCache {
 public:
  static constexpr int kNumAddrBits = fpga_tools::CeilLog2(k_mem_depth);
  using addr_t = ac_int<kNumAddrBits, false>;

  OnchipMemoryWithCache() {
    UnrolledLoop<k_cache_depth>([&](auto i) {
      cache_valid_[i] = false;
    });
  }
  OnchipMemoryWithCache(T init_val) { init(init_val); }

  void init(T init_val) {
    for (int i = 0; i < k_mem_depth; i++) {
      data_[i] = init_val;
    }
    UnrolledLoop<k_cache_depth>([&](auto i) {
      cache_valid_[i] = false;
    });
  }

  OnchipMemoryWithCache(const OnchipMemoryWithCache&) = delete;
  OnchipMemoryWithCache& operator=(const OnchipMemoryWithCache&) = delete;

  // explicitly communicate to developers that we don't want to support a
  // square bracket operator that returns a reference, as it would allow
  // modification of the memory without updating the cache
  template <typename I>
  T& operator[](I addr) = delete;

  // we can support the square bracket operator that returns a const ref
  const T& operator[](addr_t addr) const { return read(addr); }

  void write(addr_t addr, T val) {
    // write the value from the end of the cache into the memory
    if (cache_valid_[0]) {
      data_[cache_addr_[0]] = cache_val_[0];
    }

    // Shift the values in the cache
    UnrolledLoop<k_cache_depth - 1>([&](auto i) {
      if (cache_addr_[i + 1] == addr) {
        cache_valid_[i] = false;  // invalidate old cache entry at same addr
      } else {
        cache_valid_[i] = cache_valid_[i + 1];
      }
      cache_val_[i] = cache_val_[i + 1];
      cache_addr_[i] = cache_addr_[i + 1];
    });

    // write the new value into the cache
    cache_val_[k_cache_depth - 1] = val;
    cache_addr_[k_cache_depth - 1] = addr;
    cache_valid_[k_cache_depth - 1] = true;
  }

  T read(addr_t addr) {
    T return_val;
    bool in_cache = false;

    // check the cache, newest value will take precedence
    UnrolledLoop<k_cache_depth>([&](auto i) {
      if ((cache_addr_[i] == addr) && (cache_valid_[i])) {
        return_val = cache_val_[i];
        in_cache = true;
      }
    });

    // if not in the cache, fetch from memory
    if (!in_cache) {
      return_val = data_[addr];
    }

    return return_val;
  }

 private:
  T data_[k_mem_depth];
  T cache_val_[k_cache_depth];
  addr_t cache_addr_[k_cache_depth];
  bool cache_valid_[k_cache_depth];
};  // class OnchipMemoryWithCache

// specialization for cache size 0 (no cache)
template <typename T,         // type to store in the memory
          size_t k_mem_depth  // depth of the memory
          >
class OnchipMemoryWithCache<T, k_mem_depth, 0> {
 public:
  static constexpr int kNumAddrBits = fpga_tools::CeilLog2(k_mem_depth);
  using addr_t = ac_int<kNumAddrBits, false>;

  OnchipMemoryWithCache() {}
  OnchipMemoryWithCache(T init_val) {
    for (int i = 0; i < k_mem_depth; i++) {
      data_[i] = init_val;
    }
  }
  OnchipMemoryWithCache(const OnchipMemoryWithCache&) = delete;
  OnchipMemoryWithCache& operator=(const OnchipMemoryWithCache&) = delete;
  template <typename I>
  T& operator[](I addr) = delete;
  const T& operator[](addr_t addr) const { return read(addr); }
  void write(addr_t addr, T val) { data_[addr] = val; }
  T read(addr_t addr) { return data_[addr]; }

 private:
  T data_[k_mem_depth];
};  // class OnchipMemoryWithCache<T, k_mem_depth, 0>

}  // namespace fpga_tools

#endif  // __ONCHIP_MEMORY_WITH_CACHE_HPP__
