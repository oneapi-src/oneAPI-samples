#ifndef __MAPJOIN_HPP__
#define __MAPJOIN_HPP__
#pragma once

#include <functional>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Unroller.hpp"
#include "Tuple.hpp"
#include "StreamingData.hpp"
#include "ShannonIterator.hpp"

//
// ArrayMap class
//
template <typename Type, int size>
class ArrayMap {
  // static asserts
  static_assert(size > 0,
    "size must be positive and non-zero");
  static_assert(std::is_same<bool, decltype(Type().valid)>::value,
    "Type must have a 'valid' boolean member");

 public:
  void Init() {
    for (unsigned int i = 0; i < size; i++) {
      valid[i] = false;
    }
  }

  std::pair<bool, Type> Get(unsigned int key) {
    return {valid[key], map[key]};
  }

  void Set(unsigned int key, Type data) {
    map[key] = data;
    valid[key] = true;
  }

  Type map[size];
  bool valid[size];
};

//
// MapJoin implementation
//
template<typename MapType, typename T2Pipe, typename T2Data, int t2_win_size,
         typename JoinPipe, typename JoinType>
void MapJoin(MapType& map) {
  //////////////////////////////////////////////////////////////////////////////
  // static asserts
  static_assert(t2_win_size > 0,
    "Table 2 window size must be positive and non-zero");
  static_assert(
      std::is_same<unsigned int, decltype(T2Data().PrimaryKey())>::value,
      "T2Data must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(std::is_same<bool, decltype(T2Data().valid)>::value,
    "T2Data must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(JoinType().valid)>::value,
    "JoinType must have a 'valid' boolean member");
  //////////////////////////////////////////////////////////////////////////////

  bool done = false;

  while (!done) {
    // read from the input pipe
    bool valid_pipe_read;
    StreamingData<T2Data, t2_win_size> in_data = T2Pipe::read(valid_pipe_read);

    // check if the producer is done
    done = in_data.done && valid_pipe_read;

    if (!done && valid_pipe_read) {
      // join the input data windows into output data
      StreamingData<JoinType, t2_win_size> join_data(false, true);

      // initialize all outputs to false
      UnrolledLoop<0, t2_win_size>([&](auto i) { 
        join_data.data.template get<i>().valid = false;
      });

      // check for match in the map and join if valid
      UnrolledLoop<0, t2_win_size>([&](auto j) {
        const bool t2_win_valid = in_data.data.template get<j>().valid;
        const unsigned int t2_key =
            in_data.data.template get<j>().PrimaryKey();

        auto [data_valid, map_data] = map.Get(t2_key);

        if (t2_win_valid && data_valid) {
          // NOTE: order below important if Join() overrides valid
          join_data.data.template get<j>().valid = true;
          join_data.data.template get<j>().Join(map_data,
                                                in_data.data.template get<j>());
        }
      });

      // write out joined data
      JoinPipe::write(join_data);
    }
  }
}

#endif /* __MAPJOIN_HPP__ */
