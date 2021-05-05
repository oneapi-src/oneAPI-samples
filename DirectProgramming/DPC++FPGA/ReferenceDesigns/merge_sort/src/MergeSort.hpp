#ifndef __MERGESORT_HPP__
#define __MERGESORT_HPP__

#include <array>
#include <limits>
#include <iostream>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Consume.hpp"
#include "Merge.hpp"
#include "Misc.hpp"
#include "Produce.hpp"
#include "Shuffle.hpp"
#include "UnrolledLoop.hpp"

using namespace sycl;

///////////////////////////////////////////////////////////////
// Convenient default comparators
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
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
// Kernel and pipe ID classes for the merge units
template<int u>
class ProduceAKernelID;
template<int u>
class ProduceBKernelID;
template<int u>
class MergeKernelID;
template<int u>
class ConsumeKernelID;

class ShuffleKernelID;

class APipeID;
class BPipeID;
class MergePipeID;
class AShufflePipeID;
class BShufflePipeID;
class InternalOutPipeID;

// Kernel and pipe ID classes for the merge tree
template<int u, int v>
class MergeTreeMergeKernelID;

class InternalMergeTreePipeID;
///////////////////////////////////////////////////////////////

//
// Submits all of the merge sort kernels necessary to sort 'count' elements.
// Returns all of the events to wait on.
// NOTE: there is no need to worry about returing a std::vector by value here;
// C++ return-value-optimization (RVO) will take care of it!
//
template<typename ValueT, typename IndexT, typename InPipe, typename OutPipe,
         size_t units, typename Compare>
std::vector<event> SubmitMergeSort(queue& q, size_t count,
                                   ValueT* buf_0, ValueT* buf_1, Compare comp) {
  // sanity check the number of merge units
  static_assert(units >= 1);
  static_assert(IsPow2(units));

  // sanity check on IndexT
  static_assert(std::is_integral_v<IndexT>);

  // ensure we have a valid compare function
  static_assert(std::is_invocable_r_v<bool, Compare, ValueT, ValueT>,
    "The 'Compare' function type must be invocable (i.e. operator()) with two"
    "'ValueT' arguments and returning a boolean");

  // depth of the merge tree to reduce the partial results of each merge unit
  constexpr size_t kReductionLevels = Log2(units);

  // the various pipes connecting the different kernels
  using APipes = PipeArray<APipeID, ValueT>;
  using BPipes = PipeArray<BPipeID, ValueT>;
  using MergePipes = PipeArray<MergePipeID, ValueT>;
  using AShufflePipes = PipeArray<AShufflePipeID, ValueT>;
  using BShufflePipes = PipeArray<BShufflePipeID, ValueT>;
  using InternalOutPipes = PipeArray<InternalOutPipeID, ValueT>;

  // validate 'count'
  if (count == 0) {
    std::cerr << "ERROR: 'count' must be greater than 0\n";
    std::terminate();
  } else if (!IsPow2(count)) {
    std::cerr << "ERROR: 'count' must be a power of 2\n";
    std::terminate();
  } else if (count < 2*units) {
    std::cerr << "ERROR: 'count' must be at least 2x greater than "
              << "the number of merge units (" << units << ")\n";
    std::terminate();
  } else if (count > std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the index type does not have bits to count to "
              << "'count'\n";
    std::terminate();
  }

  // validate the input buffers
  if (buf_0 == nullptr) {
    std::cerr << "ERROR: 'buf_0' is nullptr\n";
    std::terminate();
  }
  if (buf_1 == nullptr) {
    std::cerr << "ERROR: 'buf_1' is nullptr\n";
    std::terminate();
  }

  // double buffering is more convenient with an array of pointers,
  // so create one from the two buffers passed by the caller
  ValueT* buf[2] = {buf_0, buf_1};

  // using double buffering, so track the current buffer and have a simple
  // lamda to compute the next buffer index
  unsigned buf_idx = 0;
  auto next_buf_idx = [](unsigned buf_idx) {
    return buf_idx ^ 0x1;
  };

  // the number of elements each merge unit will sort
  const IndexT count_per_unit = count / units;

  // the number of sorting iterations each merge unit will perform
  const size_t iterations = Log2(count_per_unit);

  // memory to store the various merge unit and merge tree kernel events
  std::array<std::vector<event>, units> produce_a_events, produce_b_events,
                                        merge_events, consume_events;
  std::array<std::vector<event>, kReductionLevels> mt_merge_events;
  for (size_t i = 0; i < units; i++) {
    produce_a_events[i].resize(iterations);
    produce_b_events[i].resize(iterations);
    merge_events[i].resize(iterations);
    consume_events[i].resize(iterations);
  }

  ////////////////////////////////////////////////////////////////////////////
  // The initial shuffle stage which feeds the producers of each merge unit
  // with data coming from the input pipe to the sorter
  event shuffle_event = Shuffle<ShuffleKernelID, ValueT, IndexT, InPipe,
                                AShufflePipes, BShufflePipes, units>(q, count);
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Launching all of the merge unit kernels
  // start with inputs of size 1
  IndexT in_count = 1;

  for (size_t i = 0; i < iterations; i++) {
    // Producers will read from pipe on first iteration (from shuffle),
    // Consumers will write to pipe on last iteration
    bool producer_from_pipe = (i == 0);
    bool consumer_to_pipe = (i == (iterations-1));

    // launch the merge unit kernels for this iteration of the sort
    UnrolledLoop<units>([&](auto u) {
      // the intra merge unit pipes
      using APipe = typename APipes::template pipe<u>;
      using BPipe = typename BPipes::template pipe<u>;
      using MergePipe = typename MergePipes::template pipe<u>;
      using AShufflePipe = typename AShufflePipes::template pipe<u>;
      using BShufflePipe = typename BShufflePipes::template pipe<u>;

      // if there is only 1 merge unit, there will be no merge tree, so the
      // single merge unit's output pipe will be the entire sort's output pipe
      using InternalOutPipe =
        std::conditional_t<(units == 1),
                            OutPipe,
                            typename InternalOutPipes::template pipe<u>>;

      // alias the templated function names to make them shorter
      #define SubmitProduceA \
        Produce<ProduceAKernelID<u>, ValueT, IndexT, AShufflePipe, APipe>
      #define SubmitProduceB \
        Produce<ProduceBKernelID<u>, ValueT, IndexT, BShufflePipe, BPipe>
      #define SubmitMerge \
        Merge<MergeKernelID<u>, ValueT, IndexT, APipe, BPipe, MergePipe>
      #define SubmitConsume \
        Consume<ConsumeKernelID<u>, ValueT, IndexT, MergePipe, InternalOutPipe>

      // Except for the first iteration, producers for the current iteration
      // must wait for the consumer to be done writing to global memory from the
      // previous iteration. This is coarse grain synchronization between the
      // producer and consumer of each merge unit.
      std::vector<event> wait_events;
      if (i != 0) {
        wait_events.push_back(consume_events[u][i-1]);
      }

      // the temporary device buffers reside in a single device allocation,
      // so compute the offset into the buffer for each merge unit.
      const size_t unit_buf_offset = count_per_unit * u;

      // get device pointers for this merge unit's producers and consumer
      ValueT* in_buf_a = buf[buf_idx] + unit_buf_offset;
      ValueT* in_buf_b = buf[buf_idx] + unit_buf_offset + in_count;
      ValueT* out_buf = buf[next_buf_idx(buf_idx)] + unit_buf_offset;

      ////////////////////////////////////////////////////////////////////////
      // Enqueue the merge unit kernels
      // Produce A
      produce_a_events[u][i] = SubmitProduceA(q, in_buf_a, count_per_unit,
                                              in_count, producer_from_pipe,
                                              wait_events);

      // Produce B
      produce_b_events[u][i] = SubmitProduceB(q, in_buf_b, count_per_unit,
                                              in_count, producer_from_pipe,
                                              wait_events); 
      
      // Merge
      merge_events[u][i] = SubmitMerge(q, count_per_unit, in_count, comp);

      // Consume
      consume_events[u][i] = SubmitConsume(q, out_buf, count_per_unit,
                                           consumer_to_pipe);
    });
    ////////////////////////////////////////////////////////////////////////

    // swap buffers
    buf_idx = next_buf_idx(buf_idx);

    // increase the input size
    in_count *= 2;
  }
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Launching all of the merge tree kernels
  // static merge tree connected by pipes to merge the sorted output of
  // each merge unit into a single sorted output through OutPipe
  // NOTE: if units==1, then there is no merge tree!
  using InternalMTPipes = PipeArray2D<InternalMergeTreePipeID, ValueT>;
  UnrolledLoop<kReductionLevels>([&](auto level) {
    // each level of the merge tree reduces the number merge kernels necessary.
    // level 0 has 'units' merge kernels, level 1 has 'units/2', and so on.
    constexpr size_t level_merge_units = units / ((1 << level)*2);

    UnrolledLoop<level_merge_units>([&](auto merge_unit) {
      // InPipeA for this merge kernel in the merge tree.
      // If the merge tree level is 0, the pipe is from a merge unit
      // otherwise it is from the previous level of the merge tree.
      using MTAPipeFromMergeUnit = 
        typename InternalOutPipes::template pipe<merge_unit*2>;
      using MTAPipeFromMergeTree =
        typename InternalMTPipes::template pipe<level-1, merge_unit*2>;
      using MTAPipe =
        typename std::conditional_t<(level == 0),
                                    MTAPipeFromMergeUnit,
                                    MTAPipeFromMergeTree>;

      // InPipeB for this merge kernel in the merge tree.
      // If the merge tree level is 0, the pipe is from a merge unit
      // otherwise it is from the previous level of the merge tree.
      using MTBPipeFromMergeUnit =
        typename InternalOutPipes::template pipe<merge_unit*2+1>;
      using MTBPipeFromMergeTree =
        typename InternalMTPipes::template pipe<level-1, merge_unit*2+1>;
      using MTBPipe =
        typename std::conditional_t<(level == 0),
                                      MTBPipeFromMergeUnit,
                                      MTBPipeFromMergeTree>;

      // OutPipe for this merge kernel in the merge tree.
      // If this is the last level, then the output pipe is the output pipe
      // of the entire sorter, otherwise it is going to another level of the
      // merge tree.
      using MTOutPipeToMT = 
        typename InternalMTPipes::template pipe<level, merge_unit>;
      using MTOutPipe =
        typename std::conditional_t<(level == (kReductionLevels-1)),
                                      OutPipe,
                                      MTOutPipeToMT>;
      
      // create an alias for the long templated function call
      #define SubmitMTMerge \
        Merge<MergeTreeMergeKernelID<level, merge_unit>, ValueT, IndexT, \
              MTAPipe, MTBPipe, MTOutPipe>
      
      // Launch the merge kernel
      mt_merge_events[level].push_back(SubmitMTMerge(q, in_count*2, in_count,
                                                     comp));
    });
    
    // increase the input size
    in_count *= 2;
  });
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Combine all kernel events into a single return vector
  std::vector<event> ret;

  // add the shuffle event
  ret.push_back(shuffle_event);
  
  // add each merge unit's sorting events
  for (size_t u = 0; u < units; u++) {
    ret.insert(ret.end(), produce_a_events[u].begin(),
               produce_a_events[u].end());
    ret.insert(ret.end(), produce_b_events[u].begin(),
               produce_b_events[u].end());
    ret.insert(ret.end(), merge_events[u].begin(), merge_events[u].end());
    ret.insert(ret.end(), consume_events[u].begin(), consume_events[u].end());
  }

  // add the merge tree kernel events
  for (size_t level = 0; level < kReductionLevels; level++) {
    ret.insert(ret.end(), mt_merge_events[level].begin(),
               mt_merge_events[level].end());
  }

  return ret;
  ////////////////////////////////////////////////////////////////////////////
}

//
// A convenience method that defaults the sorters comparator to 'LessThan'
// (i.e., operator<)
//
template<typename ValueT, typename IndexT, typename InPipe, typename OutPipe, size_t units>
std::vector<event> SubmitMergeSort(queue& q, size_t count,
                                   ValueT* buf_0, ValueT* buf_1) {
  return SubmitMergeSort<ValueT, IndexT, InPipe, OutPipe, units>(q, count,
                                                                 buf_0, buf_1,
                                                                 LessThan());
}

#endif /* __MERGESORT_HPP__ */