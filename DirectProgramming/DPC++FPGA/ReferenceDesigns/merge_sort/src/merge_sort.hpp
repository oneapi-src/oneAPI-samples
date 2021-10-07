#ifndef __MERGESORT_HPP__
#define __MERGESORT_HPP__

#include <array>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "consume.hpp"
#include "merge.hpp"
#include "produce.hpp"
#include "sorting_networks.hpp"
#include "unrolled_loop.hpp"
#include "pipe_utils.hpp" // Included from DirectProgramming/DPC++FPGA/include/
#include "impu_math.hpp"

using namespace sycl;

///////////////////////////////////////////////////////////////
// Convenient default comparators
struct LessThan {
  template <class T>
  bool operator()(T const& a, T const& b) const {
    return a < b;
  }
};

struct GreaterThan {
  template <class T>
  bool operator()(T const& a, T const& b) const {
    return a > b;
  }
};
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
// Forward declare kernel and pipe IDs to reduce name mangling
// Kernel and pipe ID classes for the merge units
template <int u>
class ProduceAKernelID;
template <int u>
class ProduceBKernelID;
template <int u>
class MergeKernelID;
template <int u>
class ConsumeKernelID;

class SortNetworkID;

class APipeID;
class BPipeID;
class MergePipeID;
class InternalOutPipeID;

// Kernel and pipe ID classes for the merge tree
template <int u, int v>
class MergeTreeMergeKernelID;

class InternalMergeTreePipeID;
///////////////////////////////////////////////////////////////

//
// Submits all of the merge sort kernels necessary to sort 'count' elements.
// Returns all of the events for the caller to wait on.
// NOTE: there is no need to worry about returing a std::vector by value here;
// C++ return-value-optimization (RVO) will take care of it!
//
template <typename ValueT, typename IndexT, typename InPipe, typename OutPipe,
          unsigned char k_width, size_t units, typename Compare>
std::vector<event> SubmitMergeSort(queue& q, size_t count, ValueT* buf_0,
                                   ValueT* buf_1, Compare comp) {
  // sanity check the number of merge units and the width of the sorter
  static_assert(units >= 1);
  static_assert(impu::math::IsPow2(units));
  static_assert(k_width >= 1);
  static_assert(impu::math::IsPow2(k_width));

  // sanity check on IndexT
  static_assert(std::is_integral_v<IndexT>);

  // ensure we have a valid compare function
  static_assert(
      std::is_invocable_r_v<bool, Compare, ValueT, ValueT>,
      "The 'Compare' function type must be invocable (i.e. operator()) with two"
      "'ValueT' arguments and returning a boolean");

  // A depth of 0 allows the compiler to pick the depth for each pipe, which
  // allows it to balance the depth of the pipeline.
  constexpr size_t kDefaultPipeDepth = 0;

  // the type that is passed around the pipes
  using PipeType = sycl::vec<ValueT, k_width>;

  // the pipes connecting the different kernels of each merge unit
  // one set of pipes for each 'units' merge units
  using APipes =
    PipeArray<APipeID, PipeType, kDefaultPipeDepth, units>;
  using BPipes =
    PipeArray<BPipeID, PipeType, kDefaultPipeDepth, units>;
  using MergePipes =
    PipeArray<MergePipeID, PipeType, kDefaultPipeDepth, units>;
  using InternalOutPipes =
      PipeArray<InternalOutPipeID, PipeType, kDefaultPipeDepth, units>;

  //////////////////////////////////////////////////////////////////////////////
  // These defines make the latter code cleaner
  #define SubmitSortNetworkKernel \
    SortNetworkKernel<SortNetworkID, ValueT, IndexT, InPipe, k_width>
  #define SubmitProduceA \
    Produce<ProduceAKernelID<u>, ValueT, IndexT, APipe, k_width>
  #define SubmitProduceB \
    Produce<ProduceBKernelID<u>, ValueT, IndexT, BPipe, k_width>
  #define SubmitMerge \
    Merge<MergeKernelID<u>, ValueT, IndexT, APipe, BPipe, MergePipe, k_width>
  #define SubmitConsume                                                     \
    Consume<ConsumeKernelID<u>, ValueT, IndexT, MergePipe, InternalOutPipe, \
            k_width>
  #define SubmitMTMerge                                                       \
    Merge<MergeTreeMergeKernelID<level, merge_unit>, ValueT, IndexT, MTAPipe, \
          MTBPipe, MTOutPipe, k_width>
  //////////////////////////////////////////////////////////////////////////////

  // depth of the merge tree to reduce the sorted partitions of each merge unit
  constexpr size_t kReductionLevels = impu::math::Log2(units);

  // validate 'count'
  if (count == 0) {
    std::cerr << "ERROR: 'count' must be greater than 0\n";
    std::terminate();
  } else if (!impu::math::IsPow2(count)) {
    std::cerr << "ERROR: 'count' must be a power of 2\n";
    std::terminate();
  } else if (count < 4 * units) {
    std::cerr << "ERROR: 'count' must be at least 4x greater than "
              << "the number of merge units (" << units << ")\n";
    std::terminate();
  } else if (count > std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the index type does not have enough bits to count to "
              << "'count'\n";
    std::terminate();
  } else if ((count / units) <= k_width) {
    std::cerr << "ERROR: 'count/units' (elements per merge unit) "
              << "must be greater than k_width\n";
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
  // so create one from the two buffers passed in by the caller
  ValueT* buf[2] = {buf_0, buf_1};

  // using double buffering, so track the current buffer and have a simple
  // lamda to compute the next buffer index
  unsigned buf_idx = 0;
  auto next_buf_idx = [](unsigned buf_idx) { return buf_idx ^ 0x1; };

  // the number of elements each merge unit will sort
  const IndexT count_per_unit = count / units;

  // each producer will produce half of the data for each merge unit
  const IndexT half_count_per_unit = count_per_unit / 2;

  // the number of sorting iterations each merge unit will perform
  // NOTE: we subtract log2(k_width) because the bitonic sorting network
  // performs the first log2(k_width) iterations of the sort while streaming
  // the input data from the input pipe into device memory.
  const IndexT iterations =
    impu::math::Log2(count_per_unit) - impu::math::Log2(k_width);

  // store the various merge unit and merge tree kernel events
  std::array<std::vector<event>, units> produce_a_events, produce_b_events,
      merge_events, consume_events;
  std::array<std::vector<event>, kReductionLevels> mt_merge_events;
  for (size_t i = 0; i < units; i++) {
    produce_a_events[i].resize(iterations);
    produce_b_events[i].resize(iterations);
    merge_events[i].resize(iterations);
    consume_events[i].resize(iterations);
  }

  // launch the sorting network kernel that performs the first log2(k_width)
  // iterations of the sort. For example, if k_width=4, the sorting network
  // sorts 4 elements per cycle, in the steady state. This means we need
  // log2(4)=2 less iterations of the merge sort since we start with sorted
  // sublists of size 4.
  auto sort_network_event =
      SubmitSortNetworkKernel(q, buf[buf_idx], count, comp);

  ////////////////////////////////////////////////////////////////////////////
  // Launching all of the merge unit kernels
  // start with inputs of size 'k_width' since the data from the input pipe
  // was sent through a sorting network that sorted sublists of size 'k_width'.
  IndexT in_count = k_width;

  // perform the sort iterations for each merge unit
  for (size_t i = 0; i < iterations; i++) {
    // The Consume kernels will write to a pipe on the last iteration
    bool consumer_to_pipe = (i == (iterations - 1));

    // launch the merge unit kernels for this iteration of the sort using
    // a front-end meta-programming unroller
    impu::UnrolledLoop<units>([&](auto u) {
      // the intra merge unit pipes
      using APipe = typename APipes::template PipeAt<u>;
      using BPipe = typename BPipes::template PipeAt<u>;
      using MergePipe = typename MergePipes::template PipeAt<u>;

      // if there is only 1 merge unit, there will be no merge tree, so the
      // single merge unit's output pipe will be the entire sort's output pipe
      using InternalOutPipe =
          std::conditional_t<(units == 1), OutPipe,
                             typename InternalOutPipes::template PipeAt<u>>;

      // build the dependency event vector
      std::vector<event> wait_events;
      if (i == 0) {
        // on the first iteration, wait for sorting network kernel to be done so
        // that all of the data is in the temp buffers in device memory
        wait_events.push_back(sort_network_event);
      } else {
        // on all iterations (except the first), Produce kernels for the
        // current iteration must wait for the Consume kernels to be done
        // writing to device memory from the previous iteration.
        // This is coarse grain synchronization between the Produce and Consume
        // kernels of each merge unit.
        wait_events.push_back(consume_events[u][i - 1]);
      }

      // the temporary device buffers reside in a single device allocation,
      // so compute the offset into the buffer for each merge unit.
      const size_t unit_buf_offset = count_per_unit * u;

      // get device pointers for this merge unit's Produce and Consume kernels
      ValueT* in_buf = buf[buf_idx];
      ValueT* out_buf = buf[next_buf_idx(buf_idx)];

      ////////////////////////////////////////////////////////////////////////
      // Enqueue the merge unit kernels
      // Produce A
      produce_a_events[u][i] =
        SubmitProduceA(q, in_buf, half_count_per_unit, in_count,
                       unit_buf_offset, wait_events);

      // Produce B
      produce_b_events[u][i] =
          SubmitProduceB(q, in_buf, half_count_per_unit, in_count,
                         unit_buf_offset + half_count_per_unit, wait_events);

      // Merge
      merge_events[u][i] = SubmitMerge(q, count_per_unit, in_count, comp);

      // Consume
      consume_events[u][i] = SubmitConsume(q, out_buf, count_per_unit,
                                           unit_buf_offset, consumer_to_pipe);
      ////////////////////////////////////////////////////////////////////////
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
  // NOTE: we actually only need 2^(kReductionLevels)-2 total pipes,
  // but we have created a 2D pipe array with kReductionLevels*units
  // pipes. The 2D pipe array makes the metaprogramming much easier and the
  // front-end compiler will not use the extra pipes and therefore they
  // will NOT be instantiated in hardware
  using InternalMTPipes =
    PipeArray<InternalMergeTreePipeID, PipeType, kDefaultPipeDepth,
                           kReductionLevels, units>;

  // create the merge tree connected by pipes to merge the sorted output
  // of each merge unit into a single sorted output. The output of the last
  // level of the merge tree will stream out of 'OutPipe'.
  // NOTE: if units==1, then there is no merge tree!
  impu::UnrolledLoop<kReductionLevels>([&](auto level) {
    // each level of the merge tree reduces the number of sorted partitions
    // by a factor of 2.
    // level 0 has 'units' merge kernels, level 1 has 'units/2', and so on...
    // See README.md for a good illustration.
    constexpr size_t kLevelMergeUnits = units / ((1 << level) * 2);

    impu::UnrolledLoop<kLevelMergeUnits>([&](auto merge_unit) {
      // When level == 0, we know we will use 'MTAPipeFromMergeUnit' and
      // 'MTBPipeFromMergeUnit' below. However, we cannot access
      // PipeAt<-1, ...> without a compiler error. So, we will set the previous
      // level to 0, knowing that we will NOT use 'MTAPipeFromMergeTree' nor
      // 'MTBPipeFromMergeTree' in the case that level == 0.
      constexpr size_t prev_level = (level == 0) ? 0 : level - 1;

      // 'PipeA' for this merge kernel in the merge tree.
      // If the merge tree level is 0, the pipe is from a merge unit,
      // otherwise it is from the previous level of the merge tree.
      using MTAPipeFromMergeUnit =
          typename InternalOutPipes::template PipeAt<merge_unit * 2>;
      using MTAPipeFromMergeTree =
          typename InternalMTPipes::template PipeAt<prev_level, merge_unit * 2>;
      using MTAPipe =
          typename std::conditional_t<(level == 0), MTAPipeFromMergeUnit,
                                      MTAPipeFromMergeTree>;

      // 'PipeB' for this merge kernel in the merge tree.
      // If the merge tree level is 0, the pipe is from a merge unit,
      // otherwise it is from the previous level of the merge tree.
      using MTBPipeFromMergeUnit =
          typename InternalOutPipes::template PipeAt<merge_unit * 2 + 1>;
      using MTBPipeFromMergeTree =
          typename InternalMTPipes::template PipeAt<prev_level,
                                                    merge_unit * 2 + 1>;
      using MTBPipe =
          typename std::conditional_t<(level == 0), MTBPipeFromMergeUnit,
                                      MTBPipeFromMergeTree>;

      // 'OutPipe' for this merge kernel in the merge tree.
      // If this is the last level, then the output pipe is the output pipe
      // of the entire sorter, otherwise it is going to another level of the
      // merge tree.
      using MTOutPipeToMT =
          typename InternalMTPipes::template PipeAt<level, merge_unit>;
      using MTOutPipe =
          typename std::conditional_t<(level == (kReductionLevels - 1)),
                                      OutPipe, MTOutPipeToMT>;

      // Launch the merge kernel
      const auto e = SubmitMTMerge(q, in_count * 2, in_count, comp);
      mt_merge_events[level].push_back(e);
    });

    // increase the input size
    in_count *= 2;
  });
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Combine all kernel events into a single return vector
  std::vector<event> ret;

  // add event from the sorting network stage
  ret.push_back(sort_network_event);

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
// A convenient function that defaults the sorter's comparator to 'LessThan'
// (i.e., operator<)
//
template <typename ValueT, typename IndexT, typename InPipe, typename OutPipe,
          unsigned char k_width, size_t units>
std::vector<event> SubmitMergeSort(queue& q, IndexT count, ValueT* buf_0,
                                   ValueT* buf_1) {
  return SubmitMergeSort<ValueT, IndexT, InPipe, OutPipe, k_width, units>(
      q, count, buf_0, buf_1, LessThan());
}

#endif /* __MERGESORT_HPP__ */
