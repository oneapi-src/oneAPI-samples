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

The design is templated on the number of elements to sort, 'SORT_SIZE',
and is composed of 'NUM_STAGES' (= log2(SORT_SIZE)) merge stages, where each
stage contains 2 FIFOs for storing sorted data, and output-selection logic.
The FIFOs in each stage have size 'SZ_FIFO'.
_______________________________________________________________________
              Stage 1  | Stage 2 |  ....  | Stage 'NUM_STAGES' |
   SZ_FIFO:      1          2      4,8..,     'SORT_SIZE'/2

               FIFO A     FIFO A                 FIFO A
              /      \   /      \               /      \
input_pipe -->        -->        -----....----->        --> output_pipe
              \      /   \      /               \      /
               FIFO B     FIFO B                 FIFO B
_______________________________________________________________________

For each stage, the high-level algorithm in the paper is as follows:
1. [Ramp Up] Store the incoming data into FIFO A for SZ_FIFO iterations.
2. [Steady state] Now that FIFO A has enough data for comparison, FIFO B may
begin receiving data and each iteration, depending on which data at the front of
FIFO A or FIFO B is selected by the comparison (the lesser for ascending sort),
one piece of data is outputted. Alternate the input-receiving FIFO every SZ_FIFO
iterations, and ensure that for each two separate sorted sets of SZ_FIFO
inputted, one merged set of 2*SZ_FIFO is outputted. It is guaranteed based on
the fill pattern that neither FIFO will ever exceed its fill capacity.

To optimize the Fmax of the design, the loading of data from FIFO A/B was
removed from the critical path by making use of a set of 'Preloaders', Preloader
A and Preloader B, in addition to the FIFOs in each stage. These caching storage
units are of size SZ_PRELOAD (a single-digit number) and are implemented in
registers. With these preloaders, the algorithm for each stage is changed as
follows:
1. [Ramp Up] Store SZ_PRELOAD incoming elements into Preloader A. Store SZ_FIFO
elements into FIFO A.
2. [Steady state] Now that FIFO A and Preloader A have enough data for
comparison, Preloader B, followed by FIFO B when the Preloader is full, may
begin receiving data and each iteration, depending on which data at the front of
Preloader A or Preloader B is selected by the comparison (the lesser for
ascending sort), one piece of data is outputted. Alternate the input-receiving
FIFO/Preloader (preloader receives until full, then FIFO receives) every SZ_FIFO
iterations, and ensure that for each two separate sorted sets of SZ_FIFO
inputted, one merged set of 2*SZ_FIFO is outputted. It is guaranteed based on
the fill pattern that neither FIFO will ever exceed its fill capacity.

The total latency from first element in to last element out (number of cycles
for ii=1) is ~ 2*SORT_SIZE, and the total element-capacity of all the FIFOs is
2*(SORT_SIZE-1).

Consecutive sets of SORT_SIZE data can be streamed into the kernel, thus at
steady state, the effective total latency does not comprise the ramp up and is
reduced to ~SORT_SIZE.

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

// For SortType being 4 bytes wide, supported SORT_SIZE template values are
// powers of 2 from [2 to 2^16] for A10 and [2 to 2^19] for S10.
template <class SortType, int SORT_SIZE, class input_pipe, class output_pipe,
          class Compare>
void sort(Compare compare);

// Sort function overload for no compare parameter - ascending order by default.
// Assumes that SortType has a compare function equivalent to:
// 'bool operator<(const SortType &t) const'
template <class SortType, int SORT_SIZE, class input_pipe,
          class output_pipe>
void sort() {
  sort<SortType, SORT_SIZE, input_pipe, output_pipe>(LessThan());
}

//============================ Utility Functions =============================//
template <int Begin, int End, int sz_fifo>
struct stage_unroller {
  template <typename Action>
  static void step(const Action &action) {
    action(std::integral_constant<int, Begin>(),
           std::integral_constant<int, sz_fifo>());
    stage_unroller<Begin + 1, End, sz_fifo + sz_fifo>::step(action);
  }
};

template <int End, int sz_fifo>
struct stage_unroller<End, End, sz_fifo> {
  template <typename Action>
  static void step(const Action &action) {}
};

template <int It, int End>
struct unroller {
  template <typename Action>
  static void step(const Action& action) {
    action(std::integral_constant<int, It>());
    unroller<It + 1, End>::step(action);
  }
};

template <int End>
struct unroller<End, End> {
  template <typename Action>
  static void step(const Action&) {}
};

template <typename T_arr, int size>
static void shift_left(T_arr (&Array)[size]) {
  unroller<0, size - 1>::step([&](auto j) { Array[j] = Array[j + 1]; });
}

//============================= FIFO Definition ==============================//

// FIFO class of size FIXED_SZ_FIFO (should be a power of 2 for optimal
// performance).
//
// The enqueue(), dequeue(), and peek() functions can be used to insert,
// remove, and read data that is in the FIFO. Whenever enqueue() or
// dequeue() are used, updateSize() should
// be used to update the size trackers for the FIFO as they are not
// updated by default to optimize the performance of the data transaction
// functions. If it is never necessary for the user to check whether the FIFO
// is full or empty, updateSize() does not need to be used.
template <class T, int FIXED_SZ_FIFO>
class FIFO {
 private:
  T memory[FIXED_SZ_FIFO];

  // front/back indicate the next location where an enqueue/dequeue operation
  // will take place respectively.
  int front;
  int back;
  int increment_front;  // stores front + 1
  int increment_back;   // stores back + 1

  // empty_counter is initialized to -1 and indicates an empty FIFO when < 0
  // (single bit check). empty_counter is incremented on
  // updateSize(didDequeue=false, didEnqueue=true), decremented upon
  // updateSize(didDequeue=true, didEnqueue=false). empty_counter_dec,
  // empty_counter_inc, etc. store the decrements and increments of
  // empty_counter and are used to quickly update empty_counter upon calling
  // updateSize().
  int empty_counter;
  int empty_counter_dec;      // stores empty_counter - 1
  int empty_counter_inc;      // stores empty_counter + 1
  int empty_counter_dec_inc;  // stores empty_counter_dec + 1
  int empty_counter_inc_inc;  // stores empty_counter_inc + 1
  int empty_counter_dec_dec;  // stores empty_counter_dec - 1
  int empty_counter_inc_dec;  // stores empty_counter_inc - 1

  bool checkEmpty() { return empty_counter < 0; }

 public:
  // Precomputed value indicating whether the FIFO is empty, providing
  // a no-op empty query.
  bool empty;

  FIFO() {}

  void enqueue(T data) {
    memory[back] = data;
    back = increment_back % (FIXED_SZ_FIFO);
    increment_back = back + 1;
  }

  T peek() { return memory[front]; }

  void dequeue() {
    front = increment_front % FIXED_SZ_FIFO;
    increment_front = front + 1;
  }

  void updateSize(bool didDequeue, bool didEnqueue) {
    if (didDequeue && !didEnqueue) {
      // Equivalent to empty_counter--
      empty_counter = empty_counter_dec;
      empty_counter_inc = empty_counter_inc_dec;
      empty_counter_dec = empty_counter_dec_dec;
    } else if (!didDequeue && didEnqueue) {
      // Equivalent to empty_counter++
      empty_counter = empty_counter_inc;
      empty_counter_inc = empty_counter_inc_inc;
      empty_counter_dec = empty_counter_dec_inc;
    }
    empty = checkEmpty();  // check if empty now
    empty_counter_inc_dec = empty_counter_inc - 1;
    empty_counter_inc_inc = empty_counter_inc + 1;
    empty_counter_dec_dec = empty_counter_dec - 1;
    empty_counter_dec_inc = empty_counter_dec + 1;
  }

  void initialize() {
    front = 0;
    back = 0;
    increment_front = 1;
    increment_back = 1;
    empty_counter = -1;
    empty = true;
    empty_counter_dec = empty_counter - 1;
    empty_counter_inc = empty_counter + 1;
    empty_counter_inc_dec = empty_counter_inc - 1;
    empty_counter_inc_inc = empty_counter_inc + 1;
    empty_counter_dec_dec = empty_counter_dec - 1;
    empty_counter_dec_inc = empty_counter_dec + 1;
  }

#ifdef DEBUG
  void printFIFO(std::string name, bool neverFull) {
    int numEntries = empty_counter != FIXED_SZ_FIFO - 1
                         ? (back - front + FIXED_SZ_FIFO) % FIXED_SZ_FIFO
                         : FIXED_SZ_FIFO;
    if (neverFull) numEntries = (back - front + FIXED_SZ_FIFO) % FIXED_SZ_FIFO;
    std::cout << "FIFO [" << name << "] Contents (Num=" << numEntries << "): ";
    for (int i = 0; i < numEntries; i++) {
      std::cout << memory[(front + i) % FIXED_SZ_FIFO] << " ";
    }
    std::cout << std::endl;
  }
#endif
};

//=========================== Preloader Definition ===========================//

template <class T, char SZ_PRELOAD, char LD_DIST>
class Preloader {
 private:
  // preloaded_data: registers for storing the preloaded data
  // data_in_flight: data in flight
  // valids_in_flight: load decisions in flight
  [[intelfpga::register]] T preloaded_data[SZ_PRELOAD];
  [[intelfpga::register]] T data_in_flight[LD_DIST];
  [[intelfpga::register]] bool valids_in_flight[LD_DIST];

  // preload_count stores the address where to insert the next item in
  // preloaded_data.
  unsigned char preload_count;
  char preload_count_dec;      // stores preload_count - 1
  char preload_count_inc;      // stores preload_count + 1
  char preload_count_dec_dec;  // stores preload_count_dec - 1
  char preload_count_dec_inc;  // stores preload_count_dec + 1

  // full_counter is initialized to SZ_PRELOAD-1 and indicates a full Preloader
  // when < 0 (single bit check). Decremented on an increase in preload_count or
  // new data being inserted in flight. full_counter_dec, full_counter_inc, etc.
  // store the decrements and increments of full_counter and are used to quickly
  // update full_counter.
  char full_counter;
  char full_counter_inc;      // stores full_counter + 1
  char full_counter_dec;      // stores full_counter - 1
  char full_counter_inc_inc;  // stores full_counter_inc + 1
  char full_counter_dec_inc;  // stores full_counter_dec + 1
  char full_counter_inc_dec;  // stores full_counter_inc - 1
  char full_counter_dec_dec;  // stores full_counter_dec - 1

  // empty_counter is initialized to -1 and indicates an empty Preloader when <
  // 0 (single bit check). Incremented on an increase in preload_count or new
  // data being inserted in flight. empty_counter_dec, empty_counter_inc, etc.
  // store the decrements and increments of empty_counter and are used to
  // quickly update empty_counter.
  char empty_counter;
  char empty_counter_dec;      // stores empty_counter - 1
  char empty_counter_inc;      // stores empty_counter + 1
  char empty_counter_inc_dec;  // stores empty_counter_inc - 1
  char empty_counter_dec_dec;  // stores empty_counter_dec - 1
  char empty_counter_inc_inc;  // stores empty_counter_inc + 1
  char empty_counter_dec_inc;  // stores empty_counter_dec + 1

  // Computation of each index of preloaded_data == preload_count, precomputed
  // in advance of enqueueFront to remove compare from the critical path
  [[intelfpga::register]] bool preload_count_equal_indices[SZ_PRELOAD];

  // Computation of each index of preloaded_data == preload_count_dec,
  // precomputed in advance of enqueueFront to remove compare from the critical
  // path
  [[intelfpga::register]] bool preload_count_dec_equal_indices[SZ_PRELOAD];

  bool checkEmpty() { return empty_counter < 0; }

  bool checkFull() { return full_counter < 0; }

  void enqueueFront(T data) {
    // Equivalent to preloaded_data[preload_count] = data;
    // Implemented this way to convince the compiler to implement
    // preloaded_data in registers because it wouldn't cooperate even with the
    // intelfpga::register attribute.
    unroller<0, SZ_PRELOAD>::step([&](auto s) {
      if (preload_count_equal_indices[s]) preloaded_data[s] = data;
    });

    // Equivalent to preload_count++
    preload_count = preload_count_inc;
    preload_count_inc = preload_count + 1;
  }

  // Used to precompute computation of each index of preloaded_data ==
  // preload_count_dec and computation of each index of preloaded_data ==
  // preload_count.
  void updateComparePrecomputations() {
    unroller<0, SZ_PRELOAD>::step([&](auto s) {
      const bool preload_count_equal = (s == preload_count);
      const bool preload_count_dec_equal = (s == preload_count_dec);
      preload_count_equal_indices[s] = preload_count_equal;
      preload_count_dec_equal_indices[s] = preload_count_dec_equal;
    });
  }

 public:
  // Precomputed values indicating whether the FIFO is empty/full, providing
  // no-op empty/full queries.
  bool empty;
  bool full;

  Preloader() {}

  T peek() { return preloaded_data[0]; }

  void dequeue() {
    shift_left(preloaded_data);

    // Equivalent to preload_count--
    preload_count_inc = preload_count;
    preload_count = preload_count_dec;
    unroller<0, SZ_PRELOAD>::step([&](auto s) {
      preload_count_equal_indices[s] = preload_count_dec_equal_indices[s];
    });
  }

  void prestore(T data) {
    enqueueFront(data);

    // Equivalent to preload_count_dec++
    preload_count_dec = preload_count_dec_inc;
    preload_count_dec_dec++;
    preload_count_dec_inc++;

    // Equivalent to full_counter--
    full_counter = full_counter_dec;
    full = checkFull();
    full_counter_inc = full_counter_inc_dec;
    full_counter_dec = full_counter_dec_dec;
    full_counter_inc_inc = full_counter_inc + 1;
    full_counter_dec_inc = full_counter_dec + 1;
    full_counter_inc_dec = full_counter_inc - 1;
    full_counter_dec_dec = full_counter_dec - 1;

    // Equivalent to empty_counter++
    empty_counter = empty_counter_inc;
    empty = false;
    empty_counter_dec = empty_counter_dec_inc;
    empty_counter_inc = empty_counter_inc_inc;
    empty_counter_inc_dec = empty_counter_inc - 1;
    empty_counter_dec_dec = empty_counter_dec - 1;
    empty_counter_inc_inc = empty_counter_inc + 1;
    empty_counter_dec_inc = empty_counter_dec + 1;

    updateComparePrecomputations();
  }

  // Insert the decision to preload, and the corresponding data (garbage data
  // if insert = false). Shift the data in flight, and store the data from
  // LD_DIST-1 iterations ago into the preloader if valid.
  // Set didDequeue = true if dequeue() was called after the last call
  // of advanceCycle().
  void advanceCycle(bool insert, T insert_data, bool didDequeue) {
    data_in_flight[LD_DIST - 1] = insert_data;
    valids_in_flight[LD_DIST - 1] = insert;
    shift_left(valids_in_flight);
    shift_left(data_in_flight);
    bool valid_data_arrived = valids_in_flight[0];
    if (valid_data_arrived) enqueueFront(data_in_flight[0]);

    // If dequeue() was called and no valid data was stored into preloaded_data
    // during this call, the number of elements in the preloader decreased.
    // If dequeue() was not called and valid data was just stored
    // into preloaded_data during this call, the number of elements in the
    // preloader increased.
    if (didDequeue && !valid_data_arrived)  // then preload_count_dec--
      preload_count_dec = preload_count_dec_dec;
    else if (!didDequeue && valid_data_arrived)  // then preload_count_dec++
      preload_count_dec = preload_count_dec_inc;
    preload_count_dec_dec = preload_count_dec - 1;
    preload_count_dec_inc = preload_count_dec + 1;

    // If dequeue() was called and didn't add new valid in-flight data,
    // the [eventual] number of elements in the preloader decreased.
    // If dequeue() wasn't called and did add new valid in-flight data,
    // the [eventual] number of elements in the preloader increased.
    if (didDequeue && !insert) {
      // Equivalent to full_counter++
      full_counter = full_counter_inc;
      full_counter_inc = full_counter_inc_inc;
      full_counter_dec = full_counter_dec_inc;
      // Equivalent to empty_counter--
      empty_counter = empty_counter_dec;
      empty_counter_inc = empty_counter_inc_dec;
      empty_counter_dec = empty_counter_dec_dec;
    } else if (!didDequeue && insert) {
      // Equivalent to full_counter--
      full_counter = full_counter_dec;
      full_counter_inc = full_counter_inc_dec;
      full_counter_dec = full_counter_dec_dec;
      // Equivalent to empty_counter++
      empty_counter = empty_counter_inc;
      empty_counter_inc = empty_counter_inc_inc;
      empty_counter_dec = empty_counter_dec_inc;
    }
    empty = checkEmpty();
    full = checkFull();
    full_counter_inc_inc = full_counter_inc + 1;
    full_counter_dec_inc = full_counter_dec + 1;
    full_counter_inc_dec = full_counter_inc - 1;
    full_counter_dec_dec = full_counter_dec - 1;
    empty_counter_inc_dec = empty_counter_inc - 1;
    empty_counter_dec_dec = empty_counter_dec - 1;
    empty_counter_inc_inc = empty_counter_inc + 1;
    empty_counter_dec_inc = empty_counter_dec + 1;

    updateComparePrecomputations();
  }

  void initialize() {
    unroller<0, LD_DIST>::step([&](auto j) { valids_in_flight[j] = false; });
    preload_count = 0;
    preload_count_dec = -1;
    preload_count_inc = 1;
    preload_count_dec_dec = preload_count_dec - 1;
    preload_count_dec_inc = preload_count_dec + 1;
    full = false;
    full_counter = SZ_PRELOAD - 1;
    full_counter_inc = full_counter + 1;
    full_counter_dec = full_counter - 1;
    full_counter_inc_inc = full_counter_inc + 1;
    full_counter_dec_inc = full_counter_dec + 1;
    full_counter_inc_dec = full_counter_inc - 1;
    full_counter_dec_dec = full_counter_dec - 1;
    empty = false;
    empty_counter = -1;
    empty_counter_dec = empty_counter - 1;
    empty_counter_inc = empty_counter + 1;
    empty_counter_inc_dec = empty_counter_inc - 1;
    empty_counter_dec_dec = empty_counter_dec - 1;
    empty_counter_inc_inc = empty_counter_inc + 1;
    empty_counter_dec_inc = empty_counter_dec + 1;
    updateComparePrecomputations();
  }

#ifdef DEBUG
  void printPreloadedData() {
    int N = (int)preload_count <= SZ_PRELOAD ? preload_count : 0;
    for (int i = 0; i < N; i++) std::cout << preloaded_data[i] << " ";
    std::cout << std::endl;
  }
#endif
};

//============================= Merge Stage =============================//

// A merge iteration for a single stage of the FIFO Merge Sorter. The first
// OUTPUT_START iterations are an initial loading phase, after which valid data
// can be outputted. SZ_FIFO: Number of cycles after which the receiving FIFO is
// switched. SORT_SIZE: Total number of elements to be sorted.
template <class SortType, int SORT_SIZE, int SZ_FIFO, char SZ_PRELOAD,
          char LD_DIST, class Compare>
void merge(FIFO<SortType, SZ_FIFO> &fA, FIFO<SortType, SZ_FIFO> &fB,
           const int OUTPUT_START, int &removeCountA, int &removeCountB,
           bool &removed_N_A, bool &removed_N_B, int &receiveCount,
           bool &isReceivingB, const int i, SortType &stream_in_data,
           SortType &out_data, int &removeCountA_inc, int &removeCountB_inc,
           bool &removed_N_A_increment, bool &removed_N_B_increment,
           Preloader<SortType, SZ_PRELOAD, LD_DIST> &preloaderA,
           Preloader<SortType, SZ_PRELOAD, LD_DIST> &preloaderB,
           int &removeCountA_inc_inc, int &removeCountB_inc_inc,
           Compare compare) {
  const bool reading_in_stream = i < SORT_SIZE;
#ifdef DEBUG
  if (reading_in_stream)
    std::cout << "Input stream data: " << stream_in_data << std::endl;
#endif

  // Main block for comparing FIFO data and selecting an output
  if (i >= OUTPUT_START) {
    // Main decision: Choose which Preloader's data should be outputted
    const bool force_extract_B =
        removed_N_A || (!reading_in_stream && preloaderA.empty);
    const bool force_extract_A =
        removed_N_B || (!reading_in_stream && preloaderB.empty);
    SortType fA_front = preloaderA.peek();
    SortType fB_front = preloaderB.peek();
    const bool extractB =
        !force_extract_A && (force_extract_B || compare(fB_front, fA_front));

    // Check whether each FIFO has arriving data, and has available data
    const bool a_data_incoming = reading_in_stream && !isReceivingB;
    const bool b_data_incoming = reading_in_stream && isReceivingB;
    const bool a_has_data = (a_data_incoming || !fA.empty);
    const bool b_has_data = (b_data_incoming || !fB.empty);

    // Make decision for whether should store FIFO data into preloader
    const bool frontB_not_full = !preloaderB.full || extractB;
    const bool store_frontB = (extractB && b_has_data) ||
                              (b_data_incoming && fB.empty && frontB_not_full);
    const bool frontA_not_full = !preloaderA.full || !extractB;
    const bool store_frontA = (!extractB && a_has_data) ||
                              (a_data_incoming && fA.empty && frontA_not_full);

    // Select output data, update counters, and shift preloader data
    if (extractB) {
      out_data = fB_front;

      // Equivalent to removeCountB++; removed_N_B = removeCountB == SZ_FIFO;
      removeCountB = removeCountB_inc;
      removeCountB_inc = removeCountB_inc_inc;
      removed_N_B = removed_N_B_increment;

      // Shift preloaded data
      preloaderB.dequeue();
    } else {
      out_data = fA_front;

      // Equivalent to removeCountA++; removed_N_A = removeCountA == SZ_FIFO;
      removeCountA = removeCountA_inc;
      removeCountA_inc = removeCountA_inc_inc;
      removed_N_A = removed_N_A_increment;

      // Shift preloaded data
      preloaderA.dequeue();
    }
#ifdef DEBUG
    std::cout << " out data: " << out_data << std::endl;
#endif

    // Precompute these operations
    removed_N_A_increment = removeCountA_inc == SZ_FIFO;
    removed_N_B_increment = removeCountB_inc == SZ_FIFO;
    removeCountA_inc_inc = removeCountA_inc + 1;
    removeCountB_inc_inc = removeCountB_inc + 1;

    // Grant permission for the next SZ_FIFO-set to go through to the output
    // stream when valid
    if (removed_N_A && removed_N_B) {
      removeCountA = 0;
      removeCountB = 0;
      removeCountA_inc = 1;
      removeCountB_inc = 1;
      removed_N_A = false;
      removed_N_B = false;
      removed_N_A_increment = 1 == SZ_FIFO;
      removed_N_B_increment = 1 == SZ_FIFO;
      removeCountA_inc_inc = 2;
      removeCountB_inc_inc = 2;
    }

    // Select the data that that should be stored in the preloader. Either data
    // from the FIFO or the in_data just received if the FIFO is empty
    // (bypassing the FIFO).
    SortType data_fA = fA.peek();
    SortType data_fB = fB.peek();
    if (isReceivingB && fB.empty)
      data_fB = stream_in_data;
    else if (!isReceivingB && fA.empty)
      data_fA = stream_in_data;

    // Indicates whether the stream_in_data bypassed the receiving FIFO and
    // was inserted directly into the Preloader for that FIFO. If true, the
    // stream_in_data should not enter the receiving FIFO.
    bool bypass = (store_frontB && fB.empty && isReceivingB) ||
                  (store_frontA && fA.empty && !isReceivingB);

    // Dequeue from FIFO if storing into preloader and did not bypass.
    // Store (must be after loading otherwise FIFO can get full) to FIFO if data
    // incoming and didn't store to preloader with bypass. Update size trackers
    // of FIFOs.
    const bool deqB = store_frontB && !fB.empty;
    const bool deqA = store_frontA && !fA.empty;
    const bool storeB = b_data_incoming && !bypass;
    const bool storeA = a_data_incoming && !bypass;
    if (deqB) fB.dequeue();
    if (deqA) fA.dequeue();
    if (storeB)
      fB.enqueue(stream_in_data);
    else if (storeA)
      fA.enqueue(stream_in_data);
    fA.updateSize(deqA, storeA);
    fB.updateSize(deqB, storeB);

    // Send the to-be-loaded FIFO data to the preloader (validity of data
    // given by store_frontX), and update the data in flight. This to-be-loaded
    // data is realized LD_DIST-1 iterations later.
    // Update size trackers of preloaders.
    preloaderA.advanceCycle(store_frontA, data_fA, !extractB);
    preloaderB.advanceCycle(store_frontB, data_fB, extractB);
  }
  // Initial loading phase, fill Preloader before FIFO
  else if (reading_in_stream) {
    if (isReceivingB) {
      if (!preloaderB.full)
        preloaderB.prestore(stream_in_data);
      else {
        fB.enqueue(stream_in_data);
        fB.updateSize(false, true);
      }
    } else {
      if (!preloaderA.full)
        preloaderA.prestore(stream_in_data);
      else {
        fA.enqueue(stream_in_data);
        fA.updateSize(false, true);
      }
    }
  }

  // Determine the receiving FIFO, alternates every SZ_FIFO iterations
  if (++receiveCount == SZ_FIFO) {
    isReceivingB = !isReceivingB;
    receiveCount = 0;
  }

#ifdef DEBUG
  fifoA.printFIFO("fifoA " + std::to_string(SZ_FIFO), false);
  fifoB.printFIFO("fifoB " + std::to_string(SZ_FIFO), false);
  preloaderA.printPreloadedData();
  preloaderB.printPreloadedData();
  std::cout << std::endl;
#endif
}

//========================== SortStages Definition ===========================//

// The Stages alias references a tuple of (FIFO<SZ=1>, FIFO<SZ=2>, FIFO<SZ=4>,
// ... FIFO<SZ=SORT_SIZE/2>). i.e. FIFOs with power of 2 memory sizes.
template <class SortType, char NUM_STAGES>
struct SortStages {
  template <int... indices>
  static std::tuple<FIFO<SortType, Pow2(indices)>...> make_array(
      std::integer_sequence<int, indices...>);

  using Stages =
      decltype(make_array(std::make_integer_sequence<int, NUM_STAGES>()));
};

//=================================== Sort ===================================//

template <class SortType, int SORT_SIZE, class input_pipe, class output_pipe,
          class Compare>
void sort(Compare compare) {
  static_assert(
      std::is_same<bool, decltype(compare(std::declval<SortType>(),
                                          std::declval<SortType>()))>::value,
      "The signature of the compare function should be equivalent to the "
      "following: bool cmp(const Type1 &a, const Type2 &b);");

  //constexpr char NUM_STAGES = std::integral_constant<char, Log2(SORT_SIZE)>();
  constexpr unsigned char NUM_STAGES = std::integral_constant<unsigned char, Log2(SORT_SIZE)>();

  // Create FIFOs
  typename SortStages<SortType, NUM_STAGES>::Stages fifosA;
  typename SortStages<SortType, NUM_STAGES>::Stages fifosB;

  // Create Preloaders
  const char SZ_PRELOAD = 15;
  const char LD_DIST = SZ_PRELOAD + 1;  // can be at most SZ_PRELOAD+1
  Preloader<SortType, SZ_PRELOAD, LD_DIST> preloadersA[NUM_STAGES];
  Preloader<SortType, SZ_PRELOAD, LD_DIST> preloadersB[NUM_STAGES];

  // Number of elements removed from FIFO A/B in each stage
  int removeCountA[NUM_STAGES];
  int removeCountB[NUM_STAGES];

  // Whether SZ_FIFO elements have been removed from FIFO A/B in each stage
  bool removed_N_A[NUM_STAGES];
  bool removed_N_B[NUM_STAGES];

  // Precomputations for remove_count_A/B and removed_SZ_FIFO_A/B variables
  int removeCountA_inc[NUM_STAGES];
  int removeCountB_inc[NUM_STAGES];
  int removeCountA_inc_inc[NUM_STAGES];
  int removeCountB_inc_inc[NUM_STAGES];
  bool removed_N_A_increment[NUM_STAGES];
  bool removed_N_B_increment[NUM_STAGES];

  // How many elements the current receiving FIFO (A or B) has received. Reset
  // when equal to SZ_FIFO.
  int receiveCount[NUM_STAGES];

  // Whether FIFO B is receiving input data. If false, FIFO A is receiving.
  bool isReceivingB[NUM_STAGES];

  // The data transfered between the sort stages
  [[intelfpga::register]] SortType stream_data[NUM_STAGES + 1];

  // Used in stage unroller to determine when a stage should begin, and when
  // a stage should start outputting data.
  int output_start[NUM_STAGES];
  int stage_start[NUM_STAGES];

  stage_unroller<0, NUM_STAGES, 1>::step([&](auto s, auto sz_fifo) {
    // Sort variables
    removeCountA[s] = 0;
    removeCountB[s] = 0;
    receiveCount[s] = 0;
    removed_N_A[s] = false;
    removed_N_B[s] = false;
    isReceivingB[s] = false;
    removeCountA_inc[s] = 1;
    removeCountB_inc[s] = 1;
    removeCountA_inc_inc[s] = 2;
    removeCountB_inc_inc[s] = 2;
    removed_N_A_increment[s] = s == 0 ? true : false;
    removed_N_B_increment[s] = s == 0 ? true : false;

    // FIFOs and Preloaders
    std::get<s>(fifosA).initialize();
    std::get<s>(fifosB).initialize();
    preloadersA[s].initialize();
    preloadersB[s].initialize();

    // Stage selection variables
    output_start[s] = SZ_PRELOAD + sz_fifo;
    stage_start[s] = (s == 0) ? 0 : stage_start[s - 1] + output_start[s - 1];
  });

  constexpr int OUTPUT_START_LAST_STAGE =
      NUM_STAGES * SZ_PRELOAD + SORT_SIZE - 1;
  constexpr int TOTAL_ITER = OUTPUT_START_LAST_STAGE + SORT_SIZE;

  // Sort
  //[[intelfpga::ii(1)]]
  for (int i = 0; i < TOTAL_ITER; i++) {
#ifdef DEBUG
    std::cout << "I: " << i << std::endl;
#endif

    if (i < SORT_SIZE) stream_data[0] = input_pipe::read();

    // All sort stages
    stage_unroller<0, NUM_STAGES, 1>::step([&](auto s, auto sz_fifo) {
      if (i >= stage_start[s])
        merge<SortType, SORT_SIZE, sz_fifo, SZ_PRELOAD, LD_DIST>(
            std::get<s>(fifosA), std::get<s>(fifosB), output_start[s],
            removeCountA[s], removeCountB[s], removed_N_A[s], removed_N_B[s],
            receiveCount[s], isReceivingB[s], i - stage_start[s],
            stream_data[s], stream_data[s + 1], removeCountA_inc[s],
            removeCountB_inc[s], removed_N_A_increment[s],
            removed_N_B_increment[s], preloadersA[s], preloadersB[s],
            removeCountA_inc_inc[s], removeCountB_inc_inc[s], compare);
    });

    if (i >= OUTPUT_START_LAST_STAGE)
      output_pipe::write(stream_data[NUM_STAGES]);
  }
}

}  // namespace ihc

#endif  //__FIFO_SORT_H__
