//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// extract_typename.hpp

#pragma once

// C++ magic that lets us extract template parameters from SYCL pipes,
// `StreamingBeat` structs based off code from https://ideone.com/SDEgq

#include <stdint.h>

#include <sycl/ext/intel/prototype/pipes_ext.hpp>

template <typename T>
struct ExtractValueType {
  typedef T value_type;
};

template <template <typename, typename...> typename X, typename T,
          typename... Args>
struct ExtractValueType<X<T, Args...>>  // specialization
{
  typedef T value_type;
};

template <typename T>
struct ExtractPipeType {
  typedef T value_type;
};

template <template <class, class, int32_t, class, typename...> class PipeClass,
          class PipeName, class PipeDataT, int32_t kPipeMinCapacity,
          class PipeProperties, typename... Args>
struct ExtractPipeType<
    PipeClass<PipeName, PipeDataT, kPipeMinCapacity, PipeProperties,
               Args...>>  // specialization
{
  typedef PipeDataT value_type;
};

template <typename T>
struct ExtractStreamingBeatType {
  typedef T value_type;
  static constexpr bool use_packets = false;
  static constexpr bool use_empty = false;
};

template <template <class, bool, bool> class BeatClass, class BeatDataT,
          bool kBeatUsePackets, bool kBeatEmpty>
struct ExtractStreamingBeatType<
    BeatClass<BeatDataT, kBeatUsePackets, kBeatEmpty>>  // specialization
{
  typedef BeatDataT value_type;
  static constexpr bool use_packets = kBeatUsePackets;
  static constexpr bool use_empty = kBeatEmpty;
};

template <typename PipeWithStreamingType>
using BeatPayload = typename ExtractStreamingBeatType<
    typename ExtractPipeType<PipeWithStreamingType>::value_type>::value_type;

template <typename PipeWithStreamingType>
constexpr bool BeatUsePackets() {
  return ExtractStreamingBeatType<typename ExtractPipeType<
      PipeWithStreamingType>::value_type>::use_packets;
}

template <typename PipeWithStreamingType>
constexpr bool BeatUseEmpty() {
  return ExtractStreamingBeatType<
      typename ExtractPipeType<PipeWithStreamingType>::value_type>::use_empty;
}
