#ifndef __FIFO_SORT_H__
#define __FIFO_SORT_H__

#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <iostream>
#include <tuple>
#include <utility>

#include "Misc.hpp"

/*
The sort() function in this header implements a local-memory-based FIFO-Based
Merge Sorter, which has been based on the architecture shown in Figure 8c) of
the following paper on a high performance sorting architecture for large
datasets on FPGA:

[1] D. Koch and J. Torresen, "FPGASort: a high performance sorting architecture
exploiting run-time reconfiguration on fpgas for large problem sorting",
in FPGA '11: ACM/SIGDA International Symposium on Field Programmable Gate
Arrays, Monterey CA USA, 2011. https://dl.acm.org/doi/10.1145/1950413.1950427

The design is templated on the number of elements to sort, 'sort_size',
and is composed of 'num_stages' (= log2(sort_size)) merge stages, where each
stage contains 2 FIFOs for storing sorted data, and output-selection logic.
The FIFOs in each stage have size 'sz_fifo'.
_______________________________________________________________________
              Stage 1  | Stage 2 |  ....  | Stage 'num_stages' |
   sz_fifo:      1          2      4,8..,     'sort_size'/2

               FIFO A     FIFO A                 FIFO A
              /      \   /      \               /      \
input_pipe -->        -->        -----....----->        --> output_pipe
              \      /   \      /               \      /
               FIFO B     FIFO B                 FIFO B
_______________________________________________________________________

For each stage, the high-level algorithm in the paper is as follows:
1. [Ramp Up] Store the incoming data into FIFO A for sz_fifo iterations.
2. [Steady state] Now that FIFO A has enough data for comparison, FIFO B may
begin receiving data and each iteration, depending on which data at the front of
FIFO A or FIFO B is selected by the comparison (the lesser for ascending sort),
one piece of data is outputted. Alternate the input-receiving FIFO every sz_fifo
iterations, and ensure that for each two separate sorted sets of sz_fifo
inputted, one merged set of 2*sz_fifo is outputted. It is guaranteed based on
the fill pattern that neither FIFO will ever exceed its fill capacity.

To optimize the Fmax of the design, the loading of data from FIFO A/B was
removed from the critical path by making use of a set of 'Preloaders', Preloader
A and Preloader B, in addition to the FIFOs in each stage. These caching storage
units are of size sz_preload (a single-digit number) and are implemented in
registers. With these preloaders, the algorithm for each stage is changed as
follows:
1. [Ramp Up] Store sz_preload incoming elements into Preloader A. Store sz_fifo
elements into FIFO A.
2. [Steady state] Now that FIFO A and Preloader A have enough data for
comparison, Preloader B, followed by FIFO B when the Preloader is full, may
begin receiving data and each iteration, depending on which data at the front of
Preloader A or Preloader B is selected by the comparison (the lesser for
ascending sort), one piece of data is outputted. Alternate the input-receiving
FIFO/Preloader (preloader receives until full, then FIFO receives) every sz_fifo
iterations, and ensure that for each two separate sorted sets of sz_fifo
inputted, one merged set of 2*sz_fifo is outputted. It is guaranteed based on
the fill pattern that neither FIFO will ever exceed its fill capacity.

The total latency from first element in to last element out (number of cycles
for ii=1) is ~ 2*sort_size, and the total element-capacity of all the FIFOs is
2*(sort_size-1).

Consecutive sets of sort_size data can be streamed into the kernel, thus at
steady state, the effective total latency does not comprise the ramp up and is
reduced to ~sort_size.

The motivation for this design using FIFOs implemented with local memory arrays
instead of pipes is primarily so that the FIFOs/storage units for the merged
data in each stage could themselves be left as an abstract unit which can be
optimized. Furthermore, the paper cited above presents the idea of using a
shared memory block of size N instead of of two FIFOs of size N, which could
nearly half the area usage (depending on how the addressing is implemented).
The reasoning behind this is that for each merge stage at steady state, one
element will be inputted and one element outputted each cyle, so after the
initial filling of N elements, the number of elements stored will remain
constant at N.
*/

namespace ihc {
//========================= Sort Function Signature ==========================//

// Convenience Functor structs LessThan and GreaterThan that can be passed to
// sort(Compare compare) by the user if SortType has a compare function
// 'bool operator<(const SortType &t) const' or
// 'bool operator>(const SortType &t) const'
struct LessThan {
  template <class T>
  bool operator()(T const &a, T const &b) const {
    return a < b;
  }
};
struct GreaterThan {
  template <class T>
  bool operator()(T const &a, T const &b) const {
    return a > b;
  }
};

// For SortType being 4 bytes wide, supported sort_size template values are
// powers of 2 from [2 to 2^16] for A10 and [2 to 2^19] for S10.
template <class SortType, int sort_size, class input_pipe, class output_pipe,
          class Compare>
void sort(Compare compare);

// Sort function overload for no compare parameter - ascending order by default.
// Assumes that SortType has a compare function equivalent to:
// 'bool operator<(const SortType &t) const'
template <class SortType, int sort_size, class input_pipe,
          class output_pipe>
void sort() {
  sort<SortType, sort_size, input_pipe, output_pipe>(LessThan());
}

//============================ Utility Functions =============================//
template <int begin, int end, int sz_fifo>
struct stage_unroller {
  template <typename Action>
  static void step(const Action &action) {
    action(std::integral_constant<int, begin>(),
           std::integral_constant<int, sz_fifo>());
    stage_unroller<begin + 1, end, sz_fifo + sz_fifo>::step(action);
  }
};

template <int end, int sz_fifo>
struct stage_unroller<end, end, sz_fifo> {
  template <typename Action>
  static void step(const Action &action) {}
};

template <int It, int end>
struct unroller {
  template <typename Action>
  static void step(const Action& action) {
    action(std::integral_constant<int, It>());
    unroller<It + 1, end>::step(action);
  }
};

template <int end>
struct unroller<end, end> {
  template <typename Action>
  static void step(const Action&) {}
};

template <typename T_arr, int size>
static void ShiftLeft(T_arr (&Array)[size]) {
  unroller<0, size - 1>::step([&](auto j) { Array[j] = Array[j + 1]; });
}

//============================= FIFO Definition ==============================//

// FIFO class of size fixed_sz_fifo (should be a power of 2 for optimal
// performance).
//
// The Enqueue(), Dequeue(), and Peek() functions can be used to insert,
// remove, and read data that is in the FIFO. Whenever Enqueue() or
// Dequeue() are used, UpdateSize() should
// be used to update the size trackers for the FIFO as they are not
// updated by default to optimize the performance of the data transaction
// functions. If it is never necessary for the user to check whether the FIFO
// is full or empty, UpdateSize() does not need to be used.
template <class T, int fixed_sz_fifo>
class FIFO {
 private:
  T memory_[fixed_sz_fifo];

  // front_/back_ indicate the next location where an Enqueue/Dequeue operation
  // will take place respectively.
  int front_;
  int back_;
  int increment_front_;  // stores front_ + 1
  int increment_back_;   // stores back_ + 1

  // empty_counter_ is initialized to -1 and indicates an empty FIFO when < 0
  // (single bit check). empty_counter_ is incremented on
  // UpdateSize(did_dequeue=false, did_enqueue=true), decremented upon
  // UpdateSize(did_dequeue=true, did_enqueue=false). empty_counter_dec_,
  // empty_counter_inc_, etc. store the decrements and increments of
  // empty_counter_ and are used to quickly update empty_counter_ upon calling
  // UpdateSize().
  int empty_counter_;
  int empty_counter_dec_;      // stores empty_counter_ - 1
  int empty_counter_inc_;      // stores empty_counter_ + 1
  int empty_counter_dec_inc_;  // stores empty_counter_dec_ + 1
  int empty_counter_inc_inc_;  // stores empty_counter_inc_ + 1
  int empty_counter_dec_dec_;  // stores empty_counter_dec_ - 1
  int empty_counter_inc_dec_;  // stores empty_counter_inc_ - 1

  bool CheckEmpty() { return empty_counter_ < 0; }

 public:
  // Precomputed value indicating whether the FIFO is empty, providing
  // a no-op empty query.
  bool empty;

  FIFO() {}

  void Enqueue(T data) {
    memory_[back_] = data;
    back_ = increment_back_ % (fixed_sz_fifo);
    increment_back_ = back_ + 1;
  }

  T Peek() { return memory_[front_]; }

  void Dequeue() {
    front_ = increment_front_ % fixed_sz_fifo;
    increment_front_ = front_ + 1;
  }

  void UpdateSize(bool did_dequeue, bool did_enqueue) {
    if (did_dequeue && !did_enqueue) {
      // Equivalent to empty_counter_--
      empty_counter_ = empty_counter_dec_;
      empty_counter_inc_ = empty_counter_inc_dec_;
      empty_counter_dec_ = empty_counter_dec_dec_;
    } else if (!did_dequeue && did_enqueue) {
      // Equivalent to empty_counter_++
      empty_counter_ = empty_counter_inc_;
      empty_counter_inc_ = empty_counter_inc_inc_;
      empty_counter_dec_ = empty_counter_dec_inc_;
    }
    empty = CheckEmpty();  // check if empty now
    empty_counter_inc_dec_ = empty_counter_inc_ - 1;
    empty_counter_inc_inc_ = empty_counter_inc_ + 1;
    empty_counter_dec_dec_ = empty_counter_dec_ - 1;
    empty_counter_dec_inc_ = empty_counter_dec_ + 1;
  }

  void Initialize() {
    front_ = 0;
    back_ = 0;
    increment_front_ = 1;
    increment_back_ = 1;
    empty_counter_ = -1;
    empty = true;
    empty_counter_dec_ = empty_counter_ - 1;
    empty_counter_inc_ = empty_counter_ + 1;
    empty_counter_inc_dec_ = empty_counter_inc_ - 1;
    empty_counter_inc_inc_ = empty_counter_inc_ + 1;
    empty_counter_dec_dec_ = empty_counter_dec_ - 1;
    empty_counter_dec_inc_ = empty_counter_dec_ + 1;
  }

#ifdef DEBUG
  void PrintFIFO(std::string name, bool neverFull) {
    int numEntries = empty_counter_ != fixed_sz_fifo - 1
                         ? (back_ - front_ + fixed_sz_fifo) % fixed_sz_fifo
                         : fixed_sz_fifo;
    if (neverFull) numEntries = (back_ - front_ + fixed_sz_fifo) % fixed_sz_fifo;
    std::cout << "FIFO [" << name << "] Contents (Num=" << numEntries << "): ";
    for (int i = 0; i < numEntries; i++) {
      std::cout << memory_[(front_ + i) % fixed_sz_fifo] << " ";
    }
    std::cout << std::endl;
  }
#endif
};

//=========================== Preloader Definition ===========================//

template <class T, char sz_preload, char ld_dist>
class Preloader {
 private:
  // preloaded_data_: registers for storing the preloaded data
  // data_in_flight_: data in flight
  // valids_in_flight_: load decisions in flight
  [[intel::fpga_register]] T preloaded_data_[sz_preload];
  [[intel::fpga_register]] T data_in_flight_[ld_dist];
  [[intel::fpga_register]] bool valids_in_flight_[ld_dist];

  // preload_count_ stores the address where to insert the next item in
  // preloaded_data_.
  unsigned char preload_count_;
  char preload_count_dec_;      // stores preload_count_ - 1
  char preload_count_inc_;      // stores preload_count_ + 1
  char preload_count_dec_dec_;  // stores preload_count_dec_ - 1
  char preload_count_dec_inc_;  // stores preload_count_dec_ + 1

  // full_counter_ is initialized to sz_preload-1 and indicates a full Preloader
  // when < 0 (single bit check). Decremented on an increase in preload_count_ or
  // new data being inserted in flight. full_counter_dec_, full_counter_inc_, etc.
  // store the decrements and increments of full_counter_ and are used to quickly
  // update full_counter_.
  char full_counter_;
  char full_counter_inc_;      // stores full_counter_ + 1
  char full_counter_dec_;      // stores full_counter_ - 1
  char full_counter_inc_inc_;  // stores full_counter_inc_ + 1
  char full_counter_dec_inc_;  // stores full_counter_dec_ + 1
  char full_counter_inc_dec_;  // stores full_counter_inc_ - 1
  char full_counter_dec_dec_;  // stores full_counter_dec_ - 1

  // empty_counter_ is initialized to -1 and indicates an empty Preloader when <
  // 0 (single bit check). Incremented on an increase in preload_count_ or new
  // data being inserted in flight. empty_counter_dec_, empty_counter_inc_, etc.
  // store the decrements and increments of empty_counter_ and are used to
  // quickly update empty_counter_.
  char empty_counter_;
  char empty_counter_dec_;      // stores empty_counter_ - 1
  char empty_counter_inc_;      // stores empty_counter_ + 1
  char empty_counter_inc_dec_;  // stores empty_counter_inc_ - 1
  char empty_counter_dec_dec_;  // stores empty_counter_dec_ - 1
  char empty_counter_inc_inc_;  // stores empty_counter_inc_ + 1
  char empty_counter_dec_inc_;  // stores empty_counter_dec_ + 1

  // Computation of each index of preloaded_data_ == preload_count_, precomputed
  // in advance of EnqueueFront to remove compare from the critical path
  [[intel::fpga_register]] bool preload_count_equal_indices_[sz_preload];

  // Computation of each index of preloaded_data_ == preload_count_dec_,
  // precomputed in advance of EnqueueFront to remove compare from the critical
  // path
  [[intel::fpga_register]] bool preload_count_dec_equal_indices_[sz_preload];

  bool CheckEmpty() { return empty_counter_ < 0; }

  bool CheckFull() { return full_counter_ < 0; }

  void EnqueueFront(T data) {
    // Equivalent to preloaded_data_[preload_count_] = data;
    // Implemented this way to convince the compiler to implement
    // preloaded_data_ in registers because it wouldn't cooperate even with the
    // intel::fpga_register attribute.
    unroller<0, sz_preload>::step([&](auto s) {
      if (preload_count_equal_indices_[s]) preloaded_data_[s] = data;
    });

    // Equivalent to preload_count_++
    preload_count_ = preload_count_inc_;
    preload_count_inc_ = preload_count_ + 1;
  }

  // Used to precompute computation of each index of preloaded_data_ ==
  // preload_count_dec_ and computation of each index of preloaded_data_ ==
  // preload_count_.
  void UpdateComparePrecomputations() {
    unroller<0, sz_preload>::step([&](auto s) {
      const bool preload_count_equal = (s == preload_count_);
      const bool preload_count_dec_equal = (s == preload_count_dec_);
      preload_count_equal_indices_[s] = preload_count_equal;
      preload_count_dec_equal_indices_[s] = preload_count_dec_equal;
    });
  }

 public:
  // Precomputed values indicating whether the FIFO is empty/full, providing
  // no-op empty/full queries.
  bool empty;
  bool full;

  Preloader() {}

  T Peek() { return preloaded_data_[0]; }

  void Dequeue() {
    ShiftLeft(preloaded_data_);

    // Equivalent to preload_count_--
    preload_count_inc_ = preload_count_;
    preload_count_ = preload_count_dec_;
    unroller<0, sz_preload>::step([&](auto s) {
      preload_count_equal_indices_[s] = preload_count_dec_equal_indices_[s];
    });
  }

  void prestore(T data) {
    EnqueueFront(data);

    // Equivalent to preload_count_dec_++
    preload_count_dec_ = preload_count_dec_inc_;
    preload_count_dec_dec_++;
    preload_count_dec_inc_++;

    // Equivalent to full_counter_--
    full_counter_ = full_counter_dec_;
    full = CheckFull();
    full_counter_inc_ = full_counter_inc_dec_;
    full_counter_dec_ = full_counter_dec_dec_;
    full_counter_inc_inc_ = full_counter_inc_ + 1;
    full_counter_dec_inc_ = full_counter_dec_ + 1;
    full_counter_inc_dec_ = full_counter_inc_ - 1;
    full_counter_dec_dec_ = full_counter_dec_ - 1;

    // Equivalent to empty_counter_++
    empty_counter_ = empty_counter_inc_;
    empty = false;
    empty_counter_dec_ = empty_counter_dec_inc_;
    empty_counter_inc_ = empty_counter_inc_inc_;
    empty_counter_inc_dec_ = empty_counter_inc_ - 1;
    empty_counter_dec_dec_ = empty_counter_dec_ - 1;
    empty_counter_inc_inc_ = empty_counter_inc_ + 1;
    empty_counter_dec_inc_ = empty_counter_dec_ + 1;

    UpdateComparePrecomputations();
  }

  // Insert the decision to preload, and the corresponding data (garbage data
  // if insert = false). Shift the data in flight, and store the data from
  // ld_dist-1 iterations ago into the preloader if valid.
  // Set did_dequeue = true if Dequeue() was called after the last call
  // of AdvanceCycle().
  void AdvanceCycle(bool insert, T insert_data, bool did_dequeue) {
    data_in_flight_[ld_dist - 1] = insert_data;
    valids_in_flight_[ld_dist - 1] = insert;
    ShiftLeft(valids_in_flight_);
    ShiftLeft(data_in_flight_);
    bool valid_data_arrived = valids_in_flight_[0];
    if (valid_data_arrived) EnqueueFront(data_in_flight_[0]);

    // If Dequeue() was called and no valid data was stored into preloaded_data_
    // during this call, the number of elements in the preloader decreased.
    // If Dequeue() was not called and valid data was just stored
    // into preloaded_data_ during this call, the number of elements in the
    // preloader increased.
    if (did_dequeue && !valid_data_arrived)  // then preload_count_dec_--
      preload_count_dec_ = preload_count_dec_dec_;
    else if (!did_dequeue && valid_data_arrived)  // then preload_count_dec_++
      preload_count_dec_ = preload_count_dec_inc_;
    preload_count_dec_dec_ = preload_count_dec_ - 1;
    preload_count_dec_inc_ = preload_count_dec_ + 1;

    // If Dequeue() was called and didn't add new valid in-flight data,
    // the [eventual] number of elements in the preloader decreased.
    // If Dequeue() wasn't called and did add new valid in-flight data,
    // the [eventual] number of elements in the preloader increased.
    if (did_dequeue && !insert) {
      // Equivalent to full_counter_++
      full_counter_ = full_counter_inc_;
      full_counter_inc_ = full_counter_inc_inc_;
      full_counter_dec_ = full_counter_dec_inc_;
      // Equivalent to empty_counter_--
      empty_counter_ = empty_counter_dec_;
      empty_counter_inc_ = empty_counter_inc_dec_;
      empty_counter_dec_ = empty_counter_dec_dec_;
    } else if (!did_dequeue && insert) {
      // Equivalent to full_counter_--
      full_counter_ = full_counter_dec_;
      full_counter_inc_ = full_counter_inc_dec_;
      full_counter_dec_ = full_counter_dec_dec_;
      // Equivalent to empty_counter_++
      empty_counter_ = empty_counter_inc_;
      empty_counter_inc_ = empty_counter_inc_inc_;
      empty_counter_dec_ = empty_counter_dec_inc_;
    }
    empty = CheckEmpty();
    full = CheckFull();
    full_counter_inc_inc_ = full_counter_inc_ + 1;
    full_counter_dec_inc_ = full_counter_dec_ + 1;
    full_counter_inc_dec_ = full_counter_inc_ - 1;
    full_counter_dec_dec_ = full_counter_dec_ - 1;
    empty_counter_inc_dec_ = empty_counter_inc_ - 1;
    empty_counter_dec_dec_ = empty_counter_dec_ - 1;
    empty_counter_inc_inc_ = empty_counter_inc_ + 1;
    empty_counter_dec_inc_ = empty_counter_dec_ + 1;

    UpdateComparePrecomputations();
  }

  void Initialize() {
    unroller<0, ld_dist>::step([&](auto j) { valids_in_flight_[j] = false; });
    preload_count_ = 0;
    preload_count_dec_ = -1;
    preload_count_inc_ = 1;
    preload_count_dec_dec_ = preload_count_dec_ - 1;
    preload_count_dec_inc_ = preload_count_dec_ + 1;
    full = false;
    full_counter_ = sz_preload - 1;
    full_counter_inc_ = full_counter_ + 1;
    full_counter_dec_ = full_counter_ - 1;
    full_counter_inc_inc_ = full_counter_inc_ + 1;
    full_counter_dec_inc_ = full_counter_dec_ + 1;
    full_counter_inc_dec_ = full_counter_inc_ - 1;
    full_counter_dec_dec_ = full_counter_dec_ - 1;
    empty = false;
    empty_counter_ = -1;
    empty_counter_dec_ = empty_counter_ - 1;
    empty_counter_inc_ = empty_counter_ + 1;
    empty_counter_inc_dec_ = empty_counter_inc_ - 1;
    empty_counter_dec_dec_ = empty_counter_dec_ - 1;
    empty_counter_inc_inc_ = empty_counter_inc_ + 1;
    empty_counter_dec_inc_ = empty_counter_dec_ + 1;
    UpdateComparePrecomputations();
  }

#ifdef DEBUG
  void PrintPreloadedData() {
    int N = (int)preload_count <= sz_preload ? preload_count : 0;
    for (int i = 0; i < N; i++) std::cout << preloaded_data_[i] << " ";
    std::cout << std::endl;
  }
#endif
};

//============================= Merge Stage =============================//

// A merge iteration for a single stage of the FIFO Merge Sorter. The first
// output_start iterations are an initial loading phase, after which valid data
// can be outputted. sz_fifo: Number of cycles after which the receiving FIFO is
// switched. sort_size: Total number of elements to be sorted.
template <class SortType, int sort_size, int sz_fifo, char sz_preload,
          char ld_dist, class Compare>
void merge(FIFO<SortType, sz_fifo> &fifo_a, FIFO<SortType, sz_fifo> &fifo_b,
           const int output_start, int &remove_count_a, int &remove_count_b,
           bool &removed_n_a, bool &removed_n_b, int &receive_count,
           bool &is_receiving_b, const int i, SortType &stream_in_data,
           SortType &out_data, int &remove_count_a_inc, int &remove_count_b_inc,
           bool &removed_n_a_increment, bool &removed_n_b_increment,
           Preloader<SortType, sz_preload, ld_dist> &preloader_a,
           Preloader<SortType, sz_preload, ld_dist> &preloader_b,
           int &remove_count_a_inc_inc, int &remove_count_b_inc_inc,
           Compare compare) {
  const bool reading_in_stream = i < sort_size;
#ifdef DEBUG
  if (reading_in_stream)
    std::cout << "Input stream data: " << stream_in_data << std::endl;
#endif

  // Main block for comparing FIFO data and selecting an output
  if (i >= output_start) {
    // Main decision: Choose which Preloader's data should be outputted
    const bool force_extract_b =
        removed_n_a || (!reading_in_stream && preloader_a.empty);
    const bool force_extract_a =
        removed_n_b || (!reading_in_stream && preloader_b.empty);
    SortType fifo_a_front = preloader_a.Peek();
    SortType fifo_b_front = preloader_b.Peek();
    const bool extract_b =
        !force_extract_a && (force_extract_b || compare(fifo_b_front, fifo_a_front));

    // Check whether each FIFO has arriving data, and has available data
    const bool a_data_incoming = reading_in_stream && !is_receiving_b;
    const bool b_data_incoming = reading_in_stream && is_receiving_b;
    const bool a_has_data = (a_data_incoming || !fifo_a.empty);
    const bool b_has_data = (b_data_incoming || !fifo_b.empty);

    // Make decision for whether should store FIFO data into preloader
    const bool front_b_not_full = !preloader_b.full || extract_b;
    const bool store_front_b = (extract_b && b_has_data) ||
                              (b_data_incoming && fifo_b.empty && front_b_not_full);
    const bool front_a_not_full = !preloader_a.full || !extract_b;
    const bool store_front_a = (!extract_b && a_has_data) ||
                              (a_data_incoming && fifo_a.empty && front_a_not_full);

    // Select output data, update counters, and shift preloader data
    if (extract_b) {
      out_data = fifo_b_front;

      // Equivalent to remove_count_b++; removed_n_b = remove_count_b == sz_fifo;
      remove_count_b = remove_count_b_inc;
      remove_count_b_inc = remove_count_b_inc_inc;
      removed_n_b = removed_n_b_increment;

      // Shift preloaded data
      preloader_b.Dequeue();
    } else {
      out_data = fifo_a_front;

      // Equivalent to remove_count_a++; removed_n_a = remove_count_a == sz_fifo;
      remove_count_a = remove_count_a_inc;
      remove_count_a_inc = remove_count_a_inc_inc;
      removed_n_a = removed_n_a_increment;

      // Shift preloaded data
      preloader_a.Dequeue();
    }
#ifdef DEBUG
    std::cout << " out data: " << out_data << std::endl;
#endif

    // Precompute these operations
    removed_n_a_increment = remove_count_a_inc == sz_fifo;
    removed_n_b_increment = remove_count_b_inc == sz_fifo;
    remove_count_a_inc_inc = remove_count_a_inc + 1;
    remove_count_b_inc_inc = remove_count_b_inc + 1;

    // Grant permission for the next sz_fifo-set to go through to the output
    // stream when valid
    if (removed_n_a && removed_n_b) {
      remove_count_a = 0;
      remove_count_b = 0;
      remove_count_a_inc = 1;
      remove_count_b_inc = 1;
      removed_n_a = false;
      removed_n_b = false;
      removed_n_a_increment = 1 == sz_fifo;
      removed_n_b_increment = 1 == sz_fifo;
      remove_count_a_inc_inc = 2;
      remove_count_b_inc_inc = 2;
    }

    // Select the data that that should be stored in the preloader. Either data
    // from the FIFO or the in_data just received if the FIFO is empty
    // (bypassing the FIFO).
    SortType data_fifo_a = fifo_a.Peek();
    SortType data_fifo_b = fifo_b.Peek();
    if (is_receiving_b && fifo_b.empty)
      data_fifo_b = stream_in_data;
    else if (!is_receiving_b && fifo_a.empty)
      data_fifo_a = stream_in_data;

    // Indicates whether the stream_in_data bypassed the receiving FIFO and
    // was inserted directly into the Preloader for that FIFO. If true, the
    // stream_in_data should not enter the receiving FIFO.
    bool bypass = (store_front_b && fifo_b.empty && is_receiving_b) ||
                  (store_front_a && fifo_a.empty && !is_receiving_b);

    // Dequeue from FIFO if storing into preloader and did not bypass.
    // Store (must be after loading otherwise FIFO can get full) to FIFO if data
    // incoming and didn't store to preloader with bypass. Update size trackers
    // of FIFOs.
    const bool deq_b = store_front_b && !fifo_b.empty;
    const bool deq_a = store_front_a && !fifo_a.empty;
    const bool store_b = b_data_incoming && !bypass;
    const bool store_a = a_data_incoming && !bypass;
    if (deq_b) fifo_b.Dequeue();
    if (deq_a) fifo_a.Dequeue();
    if (store_b)
      fifo_b.Enqueue(stream_in_data);
    else if (store_a)
      fifo_a.Enqueue(stream_in_data);
    fifo_a.UpdateSize(deq_a, store_a);
    fifo_b.UpdateSize(deq_b, store_b);

    // Send the to-be-loaded FIFO data to the preloader (validity of data
    // given by store_frontX), and update the data in flight. This to-be-loaded
    // data is realized ld_dist-1 iterations later.
    // Update size trackers of preloaders.
    preloader_a.AdvanceCycle(store_front_a, data_fifo_a, !extract_b);
    preloader_b.AdvanceCycle(store_front_b, data_fifo_b, extract_b);
  }
  // Initial loading phase, fill Preloader before FIFO
  else if (reading_in_stream) {
    if (is_receiving_b) {
      if (!preloader_b.full)
        preloader_b.prestore(stream_in_data);
      else {
        fifo_b.Enqueue(stream_in_data);
        fifo_b.UpdateSize(false, true);
      }
    } else {
      if (!preloader_a.full)
        preloader_a.prestore(stream_in_data);
      else {
        fifo_a.Enqueue(stream_in_data);
        fifo_a.UpdateSize(false, true);
      }
    }
  }

  // Determine the receiving FIFO, alternates every sz_fifo iterations
  if (++receive_count == sz_fifo) {
    is_receiving_b = !is_receiving_b;
    receive_count = 0;
  }

#ifdef DEBUG
  fifo_a.PrintFIFO("fifo_a " + std::to_string(sz_fifo), false);
  fifo_b.PrintFIFO("fifo_b " + std::to_string(sz_fifo), false);
  preloader_a.PrintPreloadedData();
  preloader_b.PrintPreloadedData();
  std::cout << std::endl;
#endif
}

//========================== SortStages Definition ===========================//

// The Stages alias references a tuple of (FIFO<SZ=1>, FIFO<SZ=2>, FIFO<SZ=4>,
// ... FIFO<SZ=sort_size/2>). i.e. FIFOs with power of 2 memory sizes.
template <class SortType, char num_stages>
struct SortStages {
  template <int... indices>
  static std::tuple<FIFO<SortType, Pow2(indices)>...> make_array(
      std::integer_sequence<int, indices...>);

  using Stages =
      decltype(make_array(std::make_integer_sequence<int, num_stages>()));
};

//=================================== Sort ===================================//

template <class SortType, int sort_size, class input_pipe, class output_pipe,
          class Compare>
void sort(Compare compare) {
  static_assert(
      std::is_same<bool, decltype(compare(std::declval<SortType>(),
                                          std::declval<SortType>()))>::value,
      "The signature of the compare function should be equivalent to the "
      "following: bool cmp(const Type1 &a, const Type2 &b);");

  //constexpr char num_stages = std::integral_constant<char, Log2(sort_size)>();
  constexpr unsigned char num_stages = std::integral_constant<unsigned char, Log2(sort_size)>();

  // Create FIFOs
  typename SortStages<SortType, num_stages>::Stages fifosA;
  typename SortStages<SortType, num_stages>::Stages fifosB;

  // Create Preloaders
  const char sz_preload = 15;
  const char ld_dist = sz_preload + 1;  // can be at most sz_preload+1
  Preloader<SortType, sz_preload, ld_dist> preloadersA[num_stages];
  Preloader<SortType, sz_preload, ld_dist> preloadersB[num_stages];

  // Number of elements removed from FIFO A/B in each stage
  int remove_count_a[num_stages];
  int remove_count_b[num_stages];

  // Whether sz_fifo elements have been removed from FIFO A/B in each stage
  bool removed_n_a[num_stages];
  bool removed_n_b[num_stages];

  // Precomputations for remove_count_A/B and removed_SZ_FIFO_A/B variables
  int remove_count_a_inc[num_stages];
  int remove_count_b_inc[num_stages];
  int remove_count_a_inc_inc[num_stages];
  int remove_count_b_inc_inc[num_stages];
  bool removed_n_a_increment[num_stages];
  bool removed_n_b_increment[num_stages];

  // How many elements the current receiving FIFO (A or B) has received. Reset
  // when equal to sz_fifo.
  int receive_count[num_stages];

  // Whether FIFO B is receiving input data. If false, FIFO A is receiving.
  bool is_receiving_b[num_stages];

  // The data transfered between the sort stages
  [[intel::fpga_register]] SortType stream_data[num_stages + 1];

  // Used in stage unroller to determine when a stage should begin, and when
  // a stage should start outputting data.
  int output_start[num_stages];
  int stage_start[num_stages];

  stage_unroller<0, num_stages, 1>::step([&](auto s, auto sz_fifo) {
    // Sort variables
    remove_count_a[s] = 0;
    remove_count_b[s] = 0;
    receive_count[s] = 0;
    removed_n_a[s] = false;
    removed_n_b[s] = false;
    is_receiving_b[s] = false;
    remove_count_a_inc[s] = 1;
    remove_count_b_inc[s] = 1;
    remove_count_a_inc_inc[s] = 2;
    remove_count_b_inc_inc[s] = 2;
    removed_n_a_increment[s] = s == 0 ? true : false;
    removed_n_b_increment[s] = s == 0 ? true : false;

    // FIFOs and Preloaders
    std::get<s>(fifosA).Initialize();
    std::get<s>(fifosB).Initialize();
    preloadersA[s].Initialize();
    preloadersB[s].Initialize();

    // Stage selection variables
    output_start[s] = sz_preload + sz_fifo;
    stage_start[s] = (s == 0) ? 0 : stage_start[s - 1] + output_start[s - 1];
  });

  constexpr int kOutputStartLastStage =
      num_stages * sz_preload + sort_size - 1;
  constexpr int kTotalIter = kOutputStartLastStage + sort_size;

  // Sort
  //[[intel::initiation_interval(1)]]
  for (int i = 0; i < kTotalIter; i++) {
#ifdef DEBUG
    std::cout << "I: " << i << std::endl;
#endif

    if (i < sort_size) stream_data[0] = input_pipe::read();

    // All sort stages
    stage_unroller<0, num_stages, 1>::step([&](auto s, auto sz_fifo) {
      if (i >= stage_start[s])
        merge<SortType, sort_size, sz_fifo, sz_preload, ld_dist>(
            std::get<s>(fifosA), std::get<s>(fifosB), output_start[s],
            remove_count_a[s], remove_count_b[s], removed_n_a[s], removed_n_b[s],
            receive_count[s], is_receiving_b[s], i - stage_start[s],
            stream_data[s], stream_data[s + 1], remove_count_a_inc[s],
            remove_count_b_inc[s], removed_n_a_increment[s],
            removed_n_b_increment[s], preloadersA[s], preloadersB[s],
            remove_count_a_inc_inc[s], remove_count_b_inc_inc[s], compare);
    });

    if (i >= kOutputStartLastStage)
      output_pipe::write(stream_data[num_stages]);
  }
}

}  // namespace ihc

#endif  //__FIFO_SORT_H__
