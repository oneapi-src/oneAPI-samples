#ifndef __MERGE_JOIN_HPP__
#define __MERGE_JOIN_HPP__
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
// Joins two tables into a single table
// Assumptions:
//      - Both tables sorted by same 'primary' key (PrimaryKey() function of
//      table types)
//      - Table 1 rows have unique primary keys
//      - Table 2 can have multiple rows with the same primary key
// More information and pseudocode can be found here:
// https://en.wikipedia.org/wiki/Sort-merge_join
//
template <typename T1Type, int t1_win_size, typename T2Type, int t2_win_size,
          typename JoinType, bool drain=false>
class MergeJoiner {
  //////////////////////////////////////////////////////////////////////////////
  // static asserts
  static_assert(t1_win_size > 0,
    "Table 1 window size must be positive and non-zero");
  static_assert(t2_win_size > 0,
    "Table 2 window size must be positive and non-zero");
  static_assert(
      std::is_same<unsigned int, decltype(T1Type().PrimaryKey())>::value,
      "T1Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(
      std::is_same<unsigned int, decltype(T2Type().PrimaryKey())>::value,
      "T2Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(std::is_same<bool, decltype(T1Type().valid)>::value,
    "T1Type must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(T2Type().valid)>::value,
    "T2Type must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(JoinType().valid)>::value,
    "JoinType must have a 'valid' boolean member");
  //////////////////////////////////////////////////////////////////////////////

 public:
  // constructor
  MergeJoiner(unsigned int t1_size, unsigned int t2_size)
      : t1_size_(t1_size), t2_size_(t2_size) {}

  // performs the join calling back to the caller for 
  // providing new input data and dealing with output data
  template<typename JoinTable1ReadCallback, typename JoinTable2ReadCallback,
           typename JoinWriteCallback>
  void Go(JoinTable1ReadCallback t1_reader, JoinTable2ReadCallback t2_reader,
          JoinWriteCallback out_writer) {
    ////////////////////////////////////////////////////////////////////////////
    // static asserts
    static_assert(std::is_invocable_r<StreamingData<T1Type, t1_win_size>,
                                      JoinTable1ReadCallback>::value,
        "JoinTable1ReadCallback must be invocable and return "
        "StreamingData<T1Type,t1_win_size>");
    static_assert(std::is_invocable_r<StreamingData<T2Type, t2_win_size>,
                                      JoinTable2ReadCallback>::value,
        "JoinTable2ReadCallback must be invocable and return "
        "StreamingData<T2Type,t2_win_size>");
    static_assert(std::is_invocable<JoinWriteCallback,
                                    StreamingData<JoinType, t2_win_size>>::value,
        "JoinWriteCallback must be invocable and accept one "
        "StreamingData<JoinType,t2_win_size> argument");
    ////////////////////////////////////////////////////////////////////////////

    // iterators for the two tables
    ShannonIterator<int,3,t1_win_size> t1_win_idx(0,t1_size_);
    ShannonIterator<int,3,t2_win_size> t2_win_idx(0,t2_size_);

    // whether to move table 1 or table 2 window
    bool move_t1_win_prev = false;
    bool move_t1_win = false;

    // whether to keep joining the tables
    bool keep_going = true;

    // table window data
    StreamingData<T1Type, t1_win_size> t1_win;
    StreamingData<T2Type, t2_win_size> t2_win;

    // track whether the data read for tables 1 and 2 are valid
    bool t1_data_valid = false, t2_data_valid = false;
    bool t1_done = false, t2_done = false;

    // initialize windows
    while (!t1_data_valid) {
      t1_win = t1_reader();
      t1_data_valid = t1_win.valid;
    }
    while (!t2_data_valid) {
      t2_win = t2_reader();
      t2_data_valid = t2_win.valid;
    }
    move_t1_win = 
      t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey();

    // work loop
    do {
      if (!t1_data_valid || !t2_data_valid) {
        ////////////////////////////////////////////////
        //// do nothing until data from producers is valid
        // keep reading input from producer until it is valid
        if (move_t1_win_prev) {
          t1_win = t1_reader();
          t1_data_valid = t1_win.valid;
          t1_done = t1_win.done && t1_data_valid;
          keep_going &= !t1_done;
        } else {
          t2_win = t2_reader();
          t2_data_valid = t2_win.valid;
          t2_done = t2_win.done && t2_data_valid;
          keep_going &= !t2_done;
        }

        move_t1_win = 
          t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey();
        ////////////////////////////////////////////////
      } else {
        //////////////////////////////////////////////////////
        //// join the input data windows into output data
        StreamingData<JoinType, t2_win_size> join_data(false, true);

        // initialize all outputs to false
        UnrolledLoop<0, t2_win_size>([&](auto i) {
          join_data.data.template get<i>().valid = false;
        });

        // crossbar join
        UnrolledLoop<0, t2_win_size>([&](auto i) {
          bool written = false;

          const bool t2_win_valid = t2_win.data.template get<i>().valid;
          const unsigned int t2_key = t2_win.data.template get<i>().PrimaryKey();

          UnrolledLoop<0, t1_win_size>([&](auto j) {
            const bool t1_win_valid = t1_win.data.template get<j>().valid;
            const unsigned int t1_key =
                t1_win.data.template get<j>().PrimaryKey();

            if(!written && t1_win_valid && t2_win_valid && (t1_key == t2_key)){
              // NOTE: order below important if Join() overrides valid
              join_data.data.template get<i>().valid = true;
              join_data.data.template get<i>().Join(
                  t1_win.data.template get<j>(), t2_win.data.template get<i>());
              written = true;
            }
          });
        });
        //////////////////////////////////////////////////////

        //////////////////////////////////////////////////////
        //// tell caller to write output data
        out_writer(join_data);
        //////////////////////////////////////////////////////

        //////////////////////////////////////////////////////
        //// move table window based on current state
        if (move_t1_win) {
          // move index
          t1_win_idx.Step();

          // ask caller to provide next window from table 1
          t1_win = t1_reader();
          t1_data_valid = t1_win.valid;
          t1_done = t1_win.done && t1_data_valid;

          keep_going = 
            (t1_win_idx.InRange() && t2_win_idx.InRange() && !t1_done);
        } else {
          // move index
          t2_win_idx.Step();

          // ask caller to provide next window from table 2
          t2_win = t2_reader();
          t2_data_valid = t2_win.valid;
          t2_done = t2_win.done && t2_data_valid;

          keep_going = 
            (t1_win_idx.InRange() && t2_win_idx.InRange() && !t2_done);
        }

        move_t1_win_prev = move_t1_win;
        move_t1_win =
          t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey();
        //////////////////////////////////////////////////////
      }

    } while (keep_going);

    // drain the input if told to by template parameter
    if (drain) {
      while (!t1_done && t1_win_idx.InRange()) {
        t1_win = t1_reader();
        if(t1_win.valid) {
          t1_done = t1_win.done;
          t1_win_idx.Step();
        }
      }

      while (!t2_done && t2_win_idx.InRange()) {
        t2_win = t2_reader();
        if(t2_win.valid) {
          t2_done = t2_win.done;
          t2_win_idx.Step();
        }
      }
    }
  }

 private:
  // size of tables
  unsigned int t1_size_;
  unsigned int t2_size_;
};

//
// Joins two tables into a single table
// Assumptions:
//      - Both tables sorted by same 'primary' key (PrimaryKey() function of
//      table types)
//      - Table 1 has a maximum of 't1_max_duplicates' rows with the same primary
//      key (sorted by this key, so consecutive)
//      - Table 2 can have unlimited numbers of rows with same primary key
//
// NOTE: If t1_max_duplicates == 1, MergeJoin should be used as it will be more
// efficient More information and pseudocode can be found here:
// https://en.wikipedia.org/wiki/Sort-merge_join
//
template <typename T1Type, int t1_max_duplicates, typename T2Type, int t2_win_size,
          typename JoinType, bool drain=false>
class DuplicateMergeJoiner {
  //////////////////////////////////////////////////////////////////////////////
  // static asserts
  static_assert(t1_max_duplicates > 0,
                "Table 1 maximum duplicates be positive and non-zero");
  static_assert(t2_win_size > 0,
                "Table 2 window size must be positive and non-zero");
  static_assert(
      std::is_same<unsigned int, decltype(T1Type().PrimaryKey())>::value,
      "T1Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(
      std::is_same<unsigned int, decltype(T2Type().PrimaryKey())>::value,
      "T2Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(std::is_same<bool, decltype(T1Type().valid)>::value,
                "T1Type must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(T2Type().valid)>::value,
                "T2Type must have a 'valid' boolean member");
  static_assert(std::is_same<bool, decltype(JoinType().valid)>::value,
                "JoinType must have a 'valid' boolean member");
  //////////////////////////////////////////////////////////////////////////////

 public:
  // constructor
  DuplicateMergeJoiner(unsigned int t1_size, unsigned int t2_size)
      : t1_size_(t1_size), t2_size_(t2_size) {}

  //
  // performs the join
  // calling back to the caller for providing new input data and dealing with
  // output data
  //
  template<typename JoinTable1ReadCallback, typename JoinTable2ReadCallback,
           typename JoinWriteCallback>
  void Go(JoinTable1ReadCallback t1_reader, JoinTable2ReadCallback t2_reader,
          JoinWriteCallback out_writer) {
    ////////////////////////////////////////////////////////////////////////////
    // static asserts
    static_assert(std::is_invocable_r<StreamingData<T1Type, t1_max_duplicates>,
                                      JoinTable1ReadCallback>::value,
                  "JoinTable1ReadCallback must be invocable and return "
                  "StreamingData<T1Type,t1_max_duplicates>");
    static_assert(std::is_invocable_r<StreamingData<T2Type, t2_win_size>,
                                      JoinTable2ReadCallback>::value,
                  "JoinTable2ReadCallback must be invocable and return "
                  "StreamingData<T2Type,t2_win_size>");
    static_assert(std::is_invocable<JoinWriteCallback,
                  StreamingData<JoinType, t1_max_duplicates * t2_win_size>>::value,
        "JoinWriteCallback must be invocable and accept one "
        "StreamingData<JoinType,t1_max_duplicates*t2_win_size> argument");
    ////////////////////////////////////////////////////////////////////////////

    // iterators for the two tables
    ShannonIterator<int,3,t1_max_duplicates> t1_win_idx(0,t1_size_);
    ShannonIterator<int,3,t2_win_size> t2_win_idx(0,t2_size_);

    // whether to move table 1 or table 2 window
    bool move_t1_win_prev = false;
    bool move_t1_win = false;

    // whether to keep joining the tables
    bool keep_going = true;

    // table window data
    StreamingData<T1Type, t1_max_duplicates> t1_win;
    StreamingData<T2Type, t2_win_size> t2_win;

    // track whether the data read for tables 1 and 2 are valid
    bool t1_data_valid = false;
    bool t2_data_valid = false;

    bool t1_done = false;
    bool t2_done = false;

    // initialize windows
    while (!t1_data_valid) {
      t1_win = t1_reader();
      t1_data_valid = t1_win.valid;
    }
    while (!t2_data_valid) {
      t2_win = t2_reader();
      t2_data_valid = t2_win.valid;
    }

    move_t1_win =
      t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey();

    // work loop
    do {
      if (!t1_data_valid || !t2_data_valid) {
        ////////////////////////////////////////////////
        //// do nothing until data from producers is valid
        // keep reading input from producer until it is valid
        if (move_t1_win_prev) {
          t1_win = t1_reader();
          t1_data_valid = t1_win.valid;
          t1_done = t1_win.done && t1_data_valid;
          keep_going &= !t1_done;
        } else {
          t2_win = t2_reader();
          t2_data_valid = t2_win.valid;
          t2_done = t2_win.done && t2_data_valid;
          keep_going &= !t2_done;
        }
        move_t1_win =
          t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey();
        ////////////////////////////////////////////////
      } else {
        //////////////////////////////////////////////////////
        //// join the input data windows into output data
        StreamingData<JoinType, t1_max_duplicates*t2_win_size> join_data(false,true);

        // initialize all validity to false
        UnrolledLoop<0, t1_max_duplicates * t2_win_size>([&](auto i) {
          join_data.data.template get<i>().valid = false;
        });

        // full crossbar join producing up to t1_max_duplicates*t2_win_size outputs
        UnrolledLoop<0, t1_max_duplicates>([&](auto i) {
          const bool t1_win_valid = t1_win.data.template get<i>().valid;
          const unsigned int t1_key = t1_win.data.template get<i>().PrimaryKey();

          UnrolledLoop<0, t2_win_size>([&](auto j) {
            const bool t2_win_valid = t2_win.data.template get<j>().valid;
            const unsigned int t2_key =
                t2_win.data.template get<j>().PrimaryKey();

            if (t1_win_valid && t2_win_valid && (t1_key == t2_key)) {
              // NOTE: order below important if Join() overrides valid
              join_data.data.template get<i * t2_win_size + j>().valid = true;
              join_data.data.template get<i * t2_win_size + j>().Join(
                  t1_win.data.template get<i>(), t2_win.data.template get<j>());
            }
          });
        });
        //////////////////////////////////////////////////////

        //////////////////////////////////////////////////////
        //// tell caller to write output data
        out_writer(join_data);
        //////////////////////////////////////////////////////

        //////////////////////////////////////////////////////
        //// move table window based on current state
        if (move_t1_win) {
          // move index
          t1_win_idx.Step();

          // ask caller to provide next window from table 1
          t1_win = t1_reader();
          t1_data_valid = t1_win.valid;
          t1_done = t1_win.done && t1_data_valid;

          keep_going =
            (t1_win_idx.InRange() && t2_win_idx.InRange() && !t1_done);
        } else {
          // move index
          t2_win_idx.Step();

          // ask caller to provide next window from table 2
          t2_win = t2_reader();
          t2_data_valid = t2_win.valid;
          t2_done = t2_win.done && t2_data_valid;
          
          keep_going = (t1_win_idx.InRange() && t2_win_idx.InRange() && !t2_done);
        }

        move_t1_win_prev = move_t1_win;
        move_t1_win =
          t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey();
        //////////////////////////////////////////////////////
      }

    } while (keep_going);

    // drain the input if told to by template parameter
    if (drain) {
      while (!t1_done && t1_win_idx.InRange()) {
        t1_win = t1_reader();
        if(t1_win.valid) {
          t1_done = t1_win.done;
          t1_win_idx.Step();
        }
      }

      while (!t2_done && t2_win_idx.InRange()) {
        t2_win = t2_reader();
        if(t2_win.valid) {
          t2_done = t2_win.done;
          t2_win_idx.Step();
        }
      }
    }
  }

 private:
  // size of tables
  unsigned int t1_size_;
  unsigned int t2_size_;
};

#endif /* __MERGE_JOIN_HPP__ */
