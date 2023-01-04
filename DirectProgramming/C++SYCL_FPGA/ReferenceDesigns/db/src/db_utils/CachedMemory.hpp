#ifndef __CACHED_MEMORY_HPP__
#define __CACHED_MEMORY_HPP__

template <typename StorageType, int n, int cache_n,
          typename IndexType = int>
class CachedMemory {
  // static asserts
  static_assert(n > 0);
  static_assert(cache_n >= 0);
  static_assert(std::is_arithmetic<StorageType>::value,
                "StorageType must be arithmetic to support accumulation");
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integral type");
  static_assert(std::numeric_limits<IndexType>::max() >= (n - 1),
                "IndexType must be large enough to index the entire array");

public:
  CachedMemory() {}
  
  void Init(StorageType init_val = 0) {
    for (int i = 0; i < n; i++) {
      mem[i] = init_val;
    }
    #pragma unroll
    for (int i = 0; i < cache_n + 1; i++) {
      cache_value[i] = init_val;
      cache_tag[i] = 0;
    }
  }

  auto Get(IndexType idx) {
    // grab the value from memory
    StorageType ret = mem[idx];

    // check for this value in the cache as well
    #pragma unroll
    for (int i = 0; i < cache_n + 1; i++) {
      if (cache_tag[i] == idx) {
        ret = cache_value[i];
      }
    }

    return ret;
  }

  void Set(IndexType idx, StorageType val) {
    // store the new value in the actual memory, and the start of the shift
    // register cache
    mem[idx] = val;
    cache_value[cache_n] = val;
    cache_tag[cache_n] = idx;

    // shift the shift register cache
    #pragma unroll
    for (int i = 0; i < cache_n; i++) {
      cache_value[i] = cache_value[i + 1];
      cache_tag[i] = cache_tag[i + 1];
    }
  }

private:
  // internal storage
  StorageType mem[n];

  // internal cache for hiding write latency
  [[intel::fpga_register]]
  StorageType cache_value[cache_n + 1];

  [[intel::fpga_register]]
  int cache_tag[cache_n + 1];
};

#endif /* __CACHED_MEMORY_HPP__ */