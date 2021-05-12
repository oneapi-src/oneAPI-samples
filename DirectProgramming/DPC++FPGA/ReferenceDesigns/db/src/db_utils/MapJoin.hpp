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
template <typename MapType, int map_size, typename T2Data, int t2_win_size,
          typename JoinType, bool drain=false>
class MapJoiner {
  // static asserts
  static_assert(map_size > 0,
    "Map size must be positive and non-zero");
  static_assert(t2_win_size > 0,
    "Table 2 window size must be positive and non-zero");
  static_assert(
      std::is_same<unsigned int, decltype(T2Data().PrimaryKey())>::value,
      "T2Data must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(std::is_same<bool, decltype(MapType().valid)>::value,
    "MapType must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(T2Data().valid)>::value,
    "T2Data must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(JoinType().valid)>::value,
    "JoinType must have a 'valid' boolean member");

 public:
  MapJoiner(unsigned int t2_size) : t2_size_(t2_size) {}

  void Init() {
    map.Init();
  }

  //
  // do the join
  //
  template<typename JoinReadCallback, typename JoinWriteCallback>
  void Go(JoinReadCallback t2_reader, JoinWriteCallback out_writer) {
    ////////////////////////////////////////////////////////////////////////////
    // static asserts
    static_assert(std::is_invocable_r<StreamingData<T2Data, t2_win_size>,
                                      JoinReadCallback>::value,
      "JoinTable1ReadCallback must be invocable and return "
      "NTuple<T1WinSize,T2Data>");
    static_assert(std::is_invocable<JoinWriteCallback,
                                    StreamingData<JoinType, t2_win_size>>::value,
      "JoinWriteCallback must be invocable and accept one "
      "NTuple<t2_win_size,JoinType> argument");
    ////////////////////////////////////////////////////////////////////////////

    bool done = false;
    ShannonIterator<int,3,t2_win_size> i(0,t2_size_);

    do {
      // grab data from callback
      StreamingData<T2Data, t2_win_size> in_data = t2_reader();

      // check if upstream is telling us we are done
      done = in_data.done && in_data.valid;

      if (in_data.valid && !done) {
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

          bool dataValid;
          MapType mapData;
          std::tie(dataValid, mapData) = map.Get(t2_key);

          if (t2_win_valid && dataValid) {
            // NOTE: order below important if Join() overrides valid
            join_data.data.template get<j>().valid = true;
            join_data.data.template get<j>().Join(mapData,
                                                 in_data.data.template get<j>());
          }
        });

        // write out joined data
        out_writer(join_data);

        // move table 2 iterator
        i.Step();
      }

    } while (i.InRange() && !done);

    // drain the input if told to by template parameter
    if (drain) {
      while (!done && i.InRange()) {
        auto in_data = t2_reader();
        if(in_data.valid) {
          done = in_data.done;
          i.Step();
        }
      }
    }
  }

  // the map for table 1
  ArrayMap<MapType, map_size> map;

 private:
  // the size of table 2 (maximum number of rows we will see for table 2)
  unsigned int t2_size_;
};

#endif /* __MAPJOIN_HPP__ */
