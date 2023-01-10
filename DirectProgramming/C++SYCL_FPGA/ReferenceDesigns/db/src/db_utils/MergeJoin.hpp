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

using namespace sycl;

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
template<typename T1Pipe, typename T1Type, int t1_win_size,
         typename T2Pipe, typename T2Type, int t2_win_size,
         typename OutPipe, typename JoinType>
void MergeJoin() {
  //////////////////////////////////////////////////////////////////////////////
  // static asserts
  static_assert(t1_win_size > 0,
                "Table 1 window size must be positive and non-zero");
  static_assert(t2_win_size > 0,
                "Table 2 window size must be positive and non-zero");
  static_assert(
      std::is_same_v<unsigned int, decltype(T1Type().PrimaryKey())>,
      "T1Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(
      std::is_same_v<unsigned int, decltype(T2Type().PrimaryKey())>,
      "T2Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(std::is_same_v<bool, decltype(T1Type().valid)>,
                "T1Type must have a 'valid' boolean member");
  static_assert(std::is_same_v<bool, decltype(T2Type().valid)>,
                "T2Type must have a 'valid' boolean member");
  static_assert(std::is_same_v<bool, decltype(JoinType().valid)>,
                "JoinType must have a 'valid' boolean member");
  //////////////////////////////////////////////////////////////////////////////

  // is the producer of table 1 and table 2 input are done streaming in
  bool t1_done = false, t2_done = false;

  // is the data read from the pipes are valid
  bool t1_win_valid = false, t2_win_valid = false;

  // are table 1 and 2 initialized with valid data 
  bool t1_initialized = false, t2_initialized = false;

  // whether to keep going in the processing loop
  bool keep_going = true;

  // whether the computation has finished or not
  bool done_comp = false;

  // boolean to select either moving the table 1 window, or the table 2 window
  bool move_t1_win = false;

  // the table window data
  StreamingData<T1Type, t1_win_size> t1_win;
  StreamingData<T2Type, t2_win_size> t2_win;

  [[intel::initiation_interval(1)]]
  while (keep_going) {
    //////////////////////////////////////////////////////
    // update T1 window
    if (!t1_initialized || !t1_win_valid || (move_t1_win && t2_win_valid) ||
        (done_comp && !t1_done)) {
      t1_win = T1Pipe::read(t1_win_valid);

      if (t1_win_valid) {
        t1_done = t1_win.done;
        t1_initialized = true;
      }
    }
    // update T2 window
    if (!t2_initialized || !t2_win_valid || (!move_t1_win && t1_win_valid) ||
        (done_comp && !t2_done)) {
      t2_win = T2Pipe::read(t2_win_valid);
      
      if (t2_win_valid) {
        t2_done = t2_win.done;
        t2_initialized = true;
      }
    }
    //////////////////////////////////////////////////////

    if (!done_comp && t1_win.valid && t2_win.valid && t1_win_valid &&
        t2_win_valid && !t1_done && !t2_done) {
      //////////////////////////////////////////////////////
      //// join the input data windows into output data
      StreamingData<JoinType, t2_win_size> join_data(false, true);

      // initialize all outputs to false
      UnrolledLoop<0, t2_win_size>([&](auto i) {
        join_data.data.template get<i>().valid = false;
      });

      // crossbar join
      UnrolledLoop<0, t2_win_size>([&](auto i) {
        const bool t2_data_valid = t2_win.data.template get<i>().valid;
        const auto t2_key = t2_win.data.template get<i>().PrimaryKey();

        UnrolledLoop<0, t1_win_size>([&](auto j) {
          const bool t1_data_valid = t1_win.data.template get<j>().valid;
          const auto t1_key = t1_win.data.template get<j>().PrimaryKey();

          if (t1_data_valid && t2_data_valid &&
              (t1_key == t2_key)) {
            // NOTE: order below important if Join() overrides valid
            join_data.data.template get<i>().valid = true;
            join_data.data.template get<i>().Join(
                t1_win.data.template get<j>(), t2_win.data.template get<i>());
          }
        });
      });
      //////////////////////////////////////////////////////

      //////////////////////////////////////////////////////
      //// tell caller to write output data
      OutPipe::write(join_data);
      //////////////////////////////////////////////////////
    }

    //////////////////////////////////////////////////////
    //// state variables
    move_t1_win = 
      (t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey());

    keep_going = !t1_done || !t2_done;
    done_comp = t1_done || t2_done;
    //////////////////////////////////////////////////////
  }
}

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
template<typename T1Pipe, typename T1Type, int t1_max_duplicates,
         typename T2Pipe, typename T2Type, int t2_win_size,
         typename OutPipe, typename JoinType>
void DuplicateMergeJoin() {
  //////////////////////////////////////////////////////////////////////////////
  // static asserts
  static_assert(t1_max_duplicates > 0,
                "Table 1 maximum duplicates be positive and non-zero");
  static_assert(t2_win_size > 0,
                "Table 2 window size must be positive and non-zero");
  static_assert(
      std::is_same_v<unsigned int, decltype(T1Type().PrimaryKey())>,
      "T1Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(
      std::is_same_v<unsigned int, decltype(T2Type().PrimaryKey())>,
      "T2Type must have 'PrimaryKey()' function that returns an 'unsigned "
      "int'");
  static_assert(std::is_same_v<bool, decltype(T1Type().valid)>,
                "T1Type must have a 'valid' boolean member");
  static_assert(std::is_same_v<bool, decltype(T2Type().valid)>,
                "T2Type must have a 'valid' boolean member");
  static_assert(std::is_same_v<bool, decltype(JoinType().valid)>,
                "JoinType must have a 'valid' boolean member");
  //////////////////////////////////////////////////////////////////////////////

  // is the producer of table 1 and table 2 input are done streaming in
  bool t1_done = false, t2_done = false;

  // is the data read from the pipes are valid
  bool t1_win_valid = false, t2_win_valid = false;

  // are table 1 and 2 initialized with valid data 
  bool t1_initialized = false, t2_initialized = false;

  // whether to keep going in the processing loop
  bool keep_going = true;

  // whether the computation has finished or not
  bool done_comp = false;

  // boolean to select either moving the table 1 window, or the table 2 window
  bool move_t1_win = false;

  // the table window data
  StreamingData<T1Type, t1_max_duplicates> t1_win;
  StreamingData<T2Type, t2_win_size> t2_win;

  [[intel::initiation_interval(1)]]
  while (keep_going) {
    //////////////////////////////////////////////////////
    // update T1 window
    if (!t1_initialized || !t1_win_valid || (move_t1_win && t2_win_valid) ||
        (done_comp && !t1_done)) {
      t1_win = T1Pipe::read(t1_win_valid);

      if (t1_win_valid) {
        t1_done = t1_win.done;
        t1_initialized = true;
      }
    }
    // update T2 window
    if (!t2_initialized || !t2_win_valid || (!move_t1_win && t1_win_valid) ||
        (done_comp && !t2_done)) {
      t2_win = T2Pipe::read(t2_win_valid);
      
      if (t2_win_valid) {
        t2_done = t2_win.done;
        t2_initialized = true;
      }
    }
    //////////////////////////////////////////////////////

    if (!done_comp && t1_win.valid && t2_win.valid && t1_win_valid &&
        t2_win_valid && !t1_done && !t2_done) {
      //////////////////////////////////////////////////////
      //// join the input data windows into output data
      StreamingData<JoinType, t1_max_duplicates * t2_win_size> join_data(false,
                                                                         true);

      // initialize all validity to false
      UnrolledLoop<0, t1_max_duplicates * t2_win_size>([&](auto i) {
        join_data.data.template get<i>().valid = false;
      });

      // full crossbar join producing up to t1_max_duplicates*t2_win_size
      UnrolledLoop<0, t1_max_duplicates>([&](auto i) {
        const bool t1_data_valid = t1_win.data.template get<i>().valid;
        const unsigned int t1_key = t1_win.data.template get<i>().PrimaryKey();

        UnrolledLoop<0, t2_win_size>([&](auto j) {
          const bool t2_data_valid = t2_win.data.template get<j>().valid;
          const unsigned int t2_key =
              t2_win.data.template get<j>().PrimaryKey();

          if (t1_data_valid && t2_data_valid && (t1_key == t2_key)) {
            // NOTE: order below important if Join() overrides valid
            join_data.data.template get<i * t2_win_size + j>().valid = true;
            join_data.data.template get<i * t2_win_size + j>().Join(
                t1_win.data.template get<i>(), t2_win.data.template get<j>());
          }
        });
      });
      //////////////////////////////////////////////////////

      //////////////////////////////////////////////////////
      //// write output data
      OutPipe::write(join_data);
      //////////////////////////////////////////////////////
    }

    //////////////////////////////////////////////////////
    //// state variables
    move_t1_win = 
      (t1_win.data.last().PrimaryKey() < t2_win.data.last().PrimaryKey());

    keep_going = !t1_done || !t2_done;
    done_comp = t1_done || t2_done;
    //////////////////////////////////////////////////////
  }
}

#endif /* __MERGE_JOIN_HPP__ */
