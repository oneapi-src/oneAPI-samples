#ifndef __STREAMINGDATA_HPP__
#define __STREAMINGDATA_HPP__
#pragma once

#include <type_traits>

#include "Tuple.hpp"

//
// Generic datatype for streaming data that holds 'Size' elements of type 'Type'
//
template <typename Type, int Size>
class StreamingData {
  // static asserts
  static_assert(Size > 0, "Size positive and non-zero");

public:
  StreamingData() : done(false), valid(false) {}
  StreamingData(bool done, bool valid) : done(done), valid(valid) {}
  StreamingData(bool done, bool valid, NTuple<Size, Type>& data)
      : done(done), valid(valid), data(data) {}
  
  // signals that the upstream component is done computing
  bool done;

  // marks if the entire tuple ('data') is valid
  bool valid;

  // the payload data
  NTuple<Size, Type> data;
};

#endif /* __STREAMINGDATA_HPP__ */