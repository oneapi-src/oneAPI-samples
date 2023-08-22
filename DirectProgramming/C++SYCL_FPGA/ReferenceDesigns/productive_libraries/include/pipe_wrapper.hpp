#pragma once
#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/ext/intel/fpga_extensions.hpp>

// The SYCL 1.2.1 device_selector class is deprecated in SYCL 2020.
// Use the callable selector object instead.
#if SYCL_LANGUAGE_VERSION >= 202001
using device_selector_t = int(*)(const sycl::device&);
#else
using device_selector_t = const sycl::device_selector &;
#endif

template <typename name, typename data_type, int32_t min_capacity, int32_t... dims>
struct pipe_wrapper {
  template <int32_t...> struct unique_id;
  template <int32_t... idxs>
  static data_type read() {
    static_assert(((idxs >= 0) && ...), "Negative index");
    static_assert(((idxs < dims) && ...), "Index out of bounds");
    return sycl::ext::intel::pipe<unique_id<idxs...>, data_type, min_capacity>::read();
  }
  template <int32_t... idxs>
  static void write(const data_type &t) {
    static_assert(((idxs >= 0) && ...), "Negative index");
    static_assert(((idxs < dims) && ...), "Index out of bounds");
    sycl::ext::intel::pipe<unique_id<idxs...>, data_type, min_capacity>::write(t);
  }
};
