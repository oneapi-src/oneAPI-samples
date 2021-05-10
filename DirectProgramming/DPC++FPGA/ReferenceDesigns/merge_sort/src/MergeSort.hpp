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
#include "Partition.hpp"
#include "UnrolledLoop.hpp"

#include "pipe_array.hpp"

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

class PartitionKernelID;

class APipeID;
class BPipeID;
class MergePipeID;
class APartitionPipeID;
class BPartitionPipeID;
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

  // a depth of 0 allows the compiler to pick a depth to try and
  // balance the pipeline
  constexpr size_t kDefaultPipeDepth = 0;

  // the various pipes connecting the different kernels
  using APipes = PipeArray<APipeID, ValueT, kDefaultPipeDepth, units>;
  using BPipes = PipeArray<BPipeID, ValueT, kDefaultPipeDepth, units>;
  using MergePipes = PipeArray<MergePipeID, ValueT, kDefaultPipeDepth, units>;
  using InternalOutPipes =
    PipeArray<InternalOutPipeID, ValueT, kDefaultPipeDepth, units>;

  using APartitionPipe = sycl::INTEL::pipe<APartitionPipeID, ValueT>;
  using BPartitionPipe = sycl::INTEL::pipe<BPartitionPipeID, ValueT>;

  // depth of the merge tree to reduce the partial results of each merge unit
  constexpr size_t kReductionLevels = Log2(units);

  // validate 'count'
  if (count == 0) {
    std::cerr << "ERROR: 'count' must be greater than 0\n";
    std::terminate();
  } else if (!IsPow2(count)) {
    std::cerr << "ERROR: 'count' must be a power of 2\n";
    std::terminate();
  } else if (count < 4*units) {
    std::cerr << "ERROR: MergeSorter 'count' must be at least 4x greater than "
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
  // NOTE: we subtract 1 because the first partition iteration will use merge
  // unit 0 to perform the first iteration of sorting
  const size_t iterations = Log2(count_per_unit) - 1;

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
  // Submit the kernels to do the initial partition of the input data from the
  // input pipe. We will use merge unit 0 (which is always there, regardless
  // of the number of merge units) to do the initial partitioning
  using APipe = typename APipes::template PipeAt<0>;
  using BPipe = typename BPipes::template PipeAt<0>;
  using MergePipe = typename MergePipes::template PipeAt<0>;
  using InternalOutPipe =
    std::conditional_t<(units == 1), OutPipe,
                        typename InternalOutPipes::template PipeAt<0>>;
  
  // empty vector of events to signal no dependencies for the producer kernels
  std::vector<event> no_deps;

  // Partition Kernel
  auto partition_event =
    Partition<PartitionKernelID, ValueT, IndexT, InPipe,
              APartitionPipe, BPartitionPipe>(q, count);

  // NOTE: 'buf[buf_idx]' are unused by the producers below
  // Produce A (merge unit 0)
  auto partition_produce_a_event =
    Produce<ProduceAKernelID<0>, ValueT, IndexT,
            APartitionPipe, APipe>(q, buf[buf_idx], count, 1, 0, true, no_deps);

  // Produce B (merge unit 0)
  auto partition_produce_b_event =
    Produce<ProduceBKernelID<0>, ValueT, IndexT,
            BPartitionPipe, BPipe>(q, buf[buf_idx], count, 1, 1, true, no_deps);

  // Merge
  auto partition_merge_event =
    Merge<MergeKernelID<0>, ValueT, IndexT,
          APipe, BPipe, MergePipe>(q, count, 1, comp);

  // Consume
  auto partition_consume_event =
    Consume<ConsumeKernelID<0>, ValueT, IndexT,
            MergePipe, InternalOutPipe>(q, buf[buf_idx], count, false);
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Launching all of the merge unit kernels
  // start with inputs of size 2 since the partition stage used merge unit 0
  // to perform the first sort iteration that sorted inputs of size 1 into 2
  IndexT in_count = 2;

  // perform the sort iterations for each unit
  for (size_t i = 0; i < iterations; i++) {
    // Consumer will write to pipe on last iteration
    bool consumer_to_pipe = (i == (iterations-1));

    // launch the merge unit kernels for this iteration of the sort
    UnrolledLoop<units>([&](auto u) {
      // the intra merge unit pipes
      using APipe = typename APipes::template PipeAt<u>;
      using BPipe = typename BPipes::template PipeAt<u>;
      using MergePipe = typename MergePipes::template PipeAt<u>;

      // if there is only 1 merge unit, there will be no merge tree, so the
      // single merge unit's output pipe will be the entire sort's output pipe
      using InternalOutPipe =
        std::conditional_t<(units == 1),
                            OutPipe,
                            typename InternalOutPipes::template PipeAt<u>>;
      
      // build the dependency event vector
      std::vector<event> wait_events;
      if (i == 0) {
        // on the first iteration, wait for the consume kernel from the
        // partition stage to be done so that all of the data is in the temp
        // buffers in device memory
        wait_events.push_back(partition_consume_event);
      } else {
        // on all iterations (except the first), producers for the current
        // iteration must wait for the consumer to be done writing to global
        // memory from the previous iteration. This is coarse grain
        // synchronization between the producer and consumer of each merge unit.
        wait_events.push_back(consume_events[u][i-1]);
      }

      // the temporary device buffers reside in a single device allocation,
      // so compute the offset into the buffer for each merge unit.
      const size_t unit_buf_offset = count_per_unit * u;

      // get device pointers for this merge unit's producers and consumer
      ValueT *in_buf = buf[buf_idx] + unit_buf_offset;
      ValueT *out_buf = buf[next_buf_idx(buf_idx)] + unit_buf_offset;

      ////////////////////////////////////////////////////////////////////////
      // Enqueue the merge unit kernels

      // Produce A and Produce B
      // merge unit 0 is different from the rest, since it is connected to the
      // Partition kernel, so it must be handled specially.
      // NOTE: See the SubmitPartitionProduce* and SubmitProduce* macros
      // defined earlier in this function
      if constexpr (u == 0) {
        // the first merge unit is special, since it is connected to the 
        // partition kernel.
        produce_a_events[u][i] =
          Produce<ProduceAKernelID<u>, ValueT, IndexT,
                  APartitionPipe, APipe>(q, in_buf, count_per_unit, in_count,
                                         0, false, wait_events);
        produce_b_events[u][i] =
          Produce<ProduceBKernelID<u>, ValueT, IndexT,
                  BPartitionPipe, BPipe>(q, in_buf, count_per_unit, in_count,
                                         in_count, false, wait_events);
      } else {
        // all other merge units (not the first) are not connected to pipes
        produce_a_events[u][i] =
          Produce<ProduceAKernelID<u>, ValueT,
                  IndexT, APipe>(q, in_buf, count_per_unit,
                                 in_count, 0, wait_events);
        produce_b_events[u][i] =
          Produce<ProduceBKernelID<u>, ValueT,
                  IndexT, BPipe>(q, in_buf, count_per_unit,
                                 in_count, in_count, wait_events);
      }

      // Merge
      merge_events[u][i] =
        Merge<MergeKernelID<u>, ValueT, IndexT,
              APipe, BPipe, MergePipe>(q, count_per_unit, in_count, comp);

      // Consume
      consume_events[u][i] =
        Consume<ConsumeKernelID<u>, ValueT, IndexT,
                MergePipe, InternalOutPipe>(q, out_buf, count_per_unit,
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

  // the merge tree pipe array
  // NOTE: we actually only need 2^(kReductionLevels) - 2 total pipes, 
  // but we have created a 2D pipe array with kReductionLevels*units*2
  // pipes. The 2D pipe array makes the programming much easier and the
  // front-end compiler will not use the extra pipes and therefore they
  // will NOT be instantiated in hardware
  using InternalMTPipes =
    PipeArray<InternalMergeTreePipeID, ValueT, kDefaultPipeDepth,
              kReductionLevels, units>;

  // create the static merge tree connected by pipes to merge the sorted output
  // of each merge unit into a single sorted output through OutPipe
  // NOTE: if units==1, then there is no merge tree!
  UnrolledLoop<kReductionLevels>([&](auto level) {
    // each level of the merge tree reduces the number merge kernels necessary.
    // level 0 has 'units' merge kernels, level 1 has 'units/2', and so on.
    constexpr size_t kLevelMergeUnits = units / ((1 << level)*2);

    UnrolledLoop<kLevelMergeUnits>([&](auto merge_unit) {
      // When level == 0, we know we will use MTAPipeFromMergeUnit and
      // MTBPipeFromMergeUnit below. However, we cannot access
      // PipeAt<-1, ...> without a compiler error. So, we will set the previous
      // level to 0, knowing that we will not use MTAPipeFromMergeTree nor
      // MTBPipeFromMergeTree
      constexpr size_t prev_level = (level == 0) ? 0 : level - 1;

      // InPipeA for this merge kernel in the merge tree.
      // If the merge tree level is 0, the pipe is from a merge unit
      // otherwise it is from the previous level of the merge tree.
      using MTAPipeFromMergeUnit = 
        typename InternalOutPipes::template PipeAt<merge_unit*2>;
      using MTAPipeFromMergeTree =
        typename InternalMTPipes::template PipeAt<prev_level, merge_unit*2>;
      using MTAPipe =
        typename std::conditional_t<(level == 0),
                                    MTAPipeFromMergeUnit,
                                    MTAPipeFromMergeTree>;

      // InPipeB for this merge kernel in the merge tree.
      // If the merge tree level is 0, the pipe is from a merge unit
      // otherwise it is from the previous level of the merge tree.
      using MTBPipeFromMergeUnit =
        typename InternalOutPipes::template PipeAt<merge_unit*2+1>;
      using MTBPipeFromMergeTree =
        typename InternalMTPipes::template PipeAt<prev_level, merge_unit*2+1>;
      using MTBPipe =
        typename std::conditional_t<(level == 0),
                                      MTBPipeFromMergeUnit,
                                      MTBPipeFromMergeTree>;

      // OutPipe for this merge kernel in the merge tree.
      // If this is the last level, then the output pipe is the output pipe
      // of the entire sorter, otherwise it is going to another level of the
      // merge tree.
      using MTOutPipeToMT = 
        typename InternalMTPipes::template PipeAt<level, merge_unit>;
      using MTOutPipe =
        typename std::conditional_t<(level == (kReductionLevels-1)),
                                     OutPipe,
                                     MTOutPipeToMT>;
      
      // Launch the merge kernel
      const auto e =
        Merge<MergeTreeMergeKernelID<level, merge_unit>, ValueT, IndexT,
              MTAPipe, MTBPipe, MTOutPipe>(q, in_count*2, in_count, comp);
      mt_merge_events[level].push_back(e);
    }); 
    
    // increase the input size
    in_count *= 2;
  });
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Combine all kernel events into a single return vector
  std::vector<event> ret;

  // add events from the partition stage
  ret.push_back(partition_event);
  ret.push_back(partition_produce_a_event);
  ret.push_back(partition_produce_b_event);
  ret.push_back(partition_merge_event);
  ret.push_back(partition_consume_event);
  
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
template<typename ValueT, typename IndexT, typename InPipe, typename OutPipe,
         size_t units>
std::vector<event> SubmitMergeSort(queue& q, size_t count,
                                   ValueT* buf_0, ValueT* buf_1) {
  return SubmitMergeSort<ValueT, IndexT, InPipe, OutPipe, units>(q, count,
                                                                 buf_0, buf_1,
                                                                 LessThan());
}

#endif /* __MERGESORT_HPP__ */