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
template <typename Type, int Size>
class ArrayMap {
  // static asserts
  static_assert(Size > 0,
    "Size must be positive and non-zero");
  static_assert(std::is_same<bool, decltype(Type().valid)>::value,
    "Type must have a 'valid' boolean member");

 public:
  void Init() {
    for (unsigned int i = 0; i < Size; i++) {
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

  Type map[Size];
  bool valid[Size];
};

//
// MapJoin implementation
//
template <typename MapType, int MapSize, typename T2Data, int T2WinSize,
          typename JoinType, bool Drain=false>
class MapJoiner {
  // static asserts
  static_assert(MapSize > 0,
    "Map size must be positive and non-zero");
  static_assert(T2WinSize > 0,
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
    static_assert(std::is_invocable_r<StreamingData<T2Data, T2WinSize>,
                                      JoinReadCallback>::value,
      "JoinTable1ReadCallback must be invocable and return "
      "NTuple<T1WinSize,T2Data>");
    static_assert(std::is_invocable<JoinWriteCallback,
                                    StreamingData<JoinType, T2WinSize>>::value,
      "JoinWriteCallback must be invocable and accept one "
      "NTuple<T2WinSize,JoinType> argument");
    ////////////////////////////////////////////////////////////////////////////

    bool done = false;
    ShannonIterator<int,3,T2WinSize> i(0,t2_size_);

    do {
      // grab data from callback
      StreamingData<T2Data, T2WinSize> in_data = t2_reader();

      // check if upstream is telling us we are done
      done = in_data.done;

      if (in_data.valid && !done) {
        // join the input data windows into output data
        StreamingData<JoinType, T2WinSize> join_data(false, true);

        // initialize all outputs to false
        UnrolledLoop<0, T2WinSize>([&](auto i) { 
          join_data.data.template get<i>().valid = false;
        });

        // check for match in the map and join if valid
        UnrolledLoop<0, T2WinSize>([&](auto j) {
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
    if (Drain) {
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
  ArrayMap<MapType, MapSize> map;

 private:
  // the size of table 2 (maximum number of rows we will see for table 2)
  unsigned int t2_size_;
};

#endif /* __MAPJOIN_HPP__ */
