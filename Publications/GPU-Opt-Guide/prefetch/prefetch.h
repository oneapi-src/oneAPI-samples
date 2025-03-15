// Snippet begin
#include <sycl/sycl.hpp>

enum LSC_LDCC {
  LSC_LDCC_DEFAULT,
  LSC_LDCC_L1UC_L3UC, // 1 // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C,  // 2 // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC,  // 3 // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C,   // 4 // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC,  // 5 // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C,   // 6 // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C, // 7 // Override to L1 invalidate-after-read, and L3
                      // cached
};

extern "C" {
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uchar(const __attribute__((opencl_global))
                                       uint8_t *base,
                                       int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_ushort(const __attribute__((opencl_global))
                                        uint16_t *base,
                                        int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uint(const __attribute__((opencl_global))
                                      uint32_t *base,
                                      int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uint2(const __attribute__((opencl_global))
                                       uint32_t *base,
                                       int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uint3(const __attribute__((opencl_global))
                                       uint32_t *base,
                                       int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uint4(const __attribute__((opencl_global))
                                       uint32_t *base,
                                       int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_uint8(const __attribute__((opencl_global))
                                       uint32_t *base,
                                       int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_ulong(const __attribute__((opencl_global))
                                       uint64_t *base,
                                       int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_ulong2(const __attribute__((opencl_global))
                                        uint64_t *base,
                                        int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_ulong3(const __attribute__((opencl_global))
                                        uint64_t *base,
                                        int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_ulong4(const __attribute__((opencl_global))
                                        uint64_t *base,
                                        int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void
__builtin_IB_lsc_prefetch_global_ulong8(const __attribute__((opencl_global))
                                        uint64_t *base,
                                        int immElemOff, enum LSC_LDCC cacheOpt);
}
// Snippet end
