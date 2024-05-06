//==----------- basic.hpp ----------------------------*-C++-*-------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===--------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_SER_BASIC_HPP__
#define __DPCT_CODEPIN_SER_BASIC_HPP__

#if defined(__linux__)
#include <cxxabi.h>
#endif
#include <iostream>
#include <sstream>
#include <string>
#ifdef __NVCC__
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

namespace dpct {
namespace experimental {

#ifdef __NVCC__
typedef cudaStream_t StreamType;
#else
typedef dpct::queue_ptr StreamType;
#endif

namespace detail {

template <typename T> void demangle_name(std::ostream &ss) {
#if defined(__linux__)
  int s;
  auto mangle_name = typeid(T).name();
  auto demangle_name = abi::__cxa_demangle(mangle_name, NULL, NULL, &s);
  if (s != 0) {
    ss << "CODEPIN:ERROR:0: Unable to demangle symbol " << mangle_name << ".";
  } else {
    ss << demangle_name;
    std::free(demangle_name);
  }
#else
  ss << typeid(T).name();
#endif
}

template <class T, class T2 = void> class DataSer {
public:
  static void dump(std::ostream &ss, T value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"";
    demangle_name<T>(ss);
    ss << "\",\"Data\":[";
    ss << "CODEPIN:ERROR:1: Unable to find the corresponding serialization "
          "function.";
    ss << "]}";
  }
};

template <class T>
class DataSer<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
  static void dump(std::ostream &ss, const T &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"";
    demangle_name<T>(ss);
    ss << "\",\"Data\":[" << value << "]}";
  }
};

#ifdef __NVCC__
template <> class DataSer<int3> {
public:
  static void dump(std::ostream &ss, const int3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"int3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x, stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y, stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z, stream);
    ss << "}";
    ss << "]}";
  }
};

template <> class DataSer<float3> {
public:
  static void dump(std::ostream &ss, const float3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"float3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.x, stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.y, stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.z, stream);
    ss << "}";
    ss << "]}";
  }
};

#else
template <> class DataSer<sycl::int3> {
public:
  static void dump(std::ostream &ss, const sycl::int3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"sycl::int3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.x(), stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.y(), stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<int>::dump(ss, value.z(), stream);
    ss << "}";
    ss << "]}";
  }
};

template <> class DataSer<sycl::float3> {
public:
  static void dump(std::ostream &ss, const sycl::float3 &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"sycl::float3\",\"Data\":[";
    ss << "{\"x\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.x(), stream);
    ss << "},";
    ss << "{\"y\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.y(), stream);
    ss << "},";
    ss << "{\"z\":";
    dpct::experimental::detail::DataSer<float>::dump(ss, value.z(), stream);
    ss << "}";
    ss << "]}";
  }
};
#endif

template <> class DataSer<char *> {
public:
  static void dump(std::ostream &ss, const char *value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"char *\",\"Data\":[";
    ss << std::string(value);
    ss << "]}";
  }
};

template <> class DataSer<std::string> {
public:
  static void dump(std::ostream &ss, const std::string &value,
                   dpct::experimental::StreamType stream) {
    ss << "{\"Type\":\"char *\",\"Data\":[";
    ss << value;
    ss << "]}";
  }
};

} // namespace detail
} // namespace experimental
} // namespace dpct

#endif
