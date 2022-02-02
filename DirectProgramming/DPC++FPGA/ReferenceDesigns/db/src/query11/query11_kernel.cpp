#include <stdio.h>

#include <type_traits>

#include "query11_kernel.hpp"
#include "pipe_types.hpp"
#include "../db_utils/CachedMemory.hpp"
#include "../db_utils/MapJoin.hpp"
#include "../db_utils/Misc.hpp"
#include "../db_utils/Tuple.hpp"
#include "../db_utils/Unroller.hpp"
#include "../db_utils/fifo_sort.hpp"

using namespace std::chrono;

// kernel class names
class ProducePartSupplier;
class JoinPartSupplierParts;
class Compute;
class FifoSort;
class ConsumeSort;

///////////////////////////////////////////////////////////////////////////////
// sort configuration
using SortType = OutputData;
constexpr int kNumSortStages = CeilLog2(kPartTableSize);
constexpr int kSortSize = Pow2(kNumSortStages);

static_assert(kPartTableSize <= kSortSize,
              "Must be able to sort all part keys");

// comparator for the sorter to sort in descending order
struct GreaterThan {
  inline bool operator()(const SortType& a, const SortType& b) {
    return a.partvalue > b.partvalue;
  }
};

// input and output pipes for the sorter
using SortInPipe = pipe<class SortInputPipe, SortType>;
using SortOutPipe = pipe<class SortOutputPipe, SortType>;
///////////////////////////////////////////////////////////////////////////////

bool SubmitQuery11(queue& q, Database& dbinfo, std::string& nation,
                    std::vector<DBIdentifier>& partkeys,
                    std::vector<DBDecimal>& values,
                    double& kernel_latency, double& total_latency) {
  // find the nationkey based on the nation name
  assert(dbinfo.n.name_key_map.find(nation) != dbinfo.n.name_key_map.end());
  unsigned char nationkey = dbinfo.n.name_key_map[nation];

  // ensure correctly sized output buffers
  partkeys.resize(kPartTableSize);
  values.resize(kPartTableSize);

  // create space for the input buffers
  // SUPPLIER
  buffer s_suppkey_buf(dbinfo.s.suppkey);
  buffer s_nationkey_buf(dbinfo.s.nationkey);
  
  // PARTSUPPLIER
  buffer ps_partkey_buf(dbinfo.ps.partkey);
  buffer ps_suppkey_buf(dbinfo.ps.suppkey);
  buffer ps_availqty_buf(dbinfo.ps.availqty);
  buffer ps_supplycost_buf(dbinfo.ps.supplycost);

  // setup the output buffers
  buffer partkeys_buf(partkeys);
  buffer values_buf(values);

  // number of producing iterations depends on the number of elements per cycle
  const size_t ps_rows = dbinfo.ps.rows;
  const size_t ps_iters = (ps_rows + kJoinWinSize - 1) / kJoinWinSize;

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  ///////////////////////////////////////////////////////////////////////////
  //// ProducePartSupplier Kernel
  auto produce_ps_event = q.submit([&](handler& h) {
    // PARTSUPPLIER table accessors
    accessor ps_partkey_accessor(ps_partkey_buf, h, read_only);
    accessor ps_suppkey_accessor(ps_suppkey_buf, h, read_only);
    accessor ps_availqty_accessor(ps_availqty_buf, h, read_only);
    accessor ps_supplycost_accessor(ps_supplycost_buf, h, read_only);

    // kernel to produce the PARTSUPPLIER table
    h.single_task<ProducePartSupplier>([=]() [[intel::kernel_args_restrict]] {
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < ps_iters; i++) {
        // bulk read of data from global memory
        NTuple<kJoinWinSize, PartSupplierRow> data;

        UnrolledLoop<0, kJoinWinSize>([&](auto j) {
          size_t idx = i * kJoinWinSize + j;
          bool in_range = idx < ps_rows;

          DBIdentifier partkey = ps_partkey_accessor[idx];
          DBIdentifier suppkey = ps_suppkey_accessor[idx];
          int availqty = ps_availqty_accessor[idx];
          DBDecimal supplycost = ps_supplycost_accessor[idx];

          data.get<j>() =
              PartSupplierRow(in_range, partkey, suppkey, availqty, supplycost);
        });

        // write to pipe
        ProducePartSupplierPipe::write(
            PartSupplierRowPipeData(false, true, data));
      }

      // tell the downstream kernel we are done producing data
      ProducePartSupplierPipe::write(PartSupplierRowPipeData(true, false));
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// JoinPartSupplierParts Kernel
  auto join_event = q.submit([&](handler& h) {
    // SUPPLIER table accessors
    size_t s_rows = dbinfo.s.rows;
    accessor s_suppkey_accessor(s_suppkey_buf, h, read_only);
    accessor s_nationkey_accessor(s_nationkey_buf, h, read_only);

    h.single_task<JoinPartSupplierParts>([=]() [[intel::kernel_args_restrict]] {
      // initialize the array map
      // +1 is to account for fact that SUPPKEY is [1,kSF*10000]
      using ArrayMapType = ArrayMap<SupplierRow, kSupplierTableSize + 1>;
      ArrayMapType array_map;
      array_map.Init();

      // populate MapJoiner map
      // why a map? keys may not be sequential
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < s_rows; i++) {
        // read in supplier and nation key
        // NOTE: based on TPCH docs, SUPPKEY is guaranteed to be unique
        // in the range [1:kSF*10000]
        DBIdentifier s_suppkey = s_suppkey_accessor[i];
        unsigned char s_nationkey = s_nationkey_accessor[i];

        array_map.Set(s_suppkey, SupplierRow(true, s_suppkey, s_nationkey));
      }

      // MAPJOIN PARTSUPPLIER and SUPPLIER tables by suppkey
      MapJoin<ArrayMapType, ProducePartSupplierPipe, PartSupplierRow,
              kJoinWinSize, PartSupplierPartsPipe,
              SupplierPartSupplierJoined>(array_map);
      
      // tell downstream we are done
      PartSupplierPartsPipe::write(
          SupplierPartSupplierJoinedPipeData(true,false));
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// Compute Kernel
  auto compute_event = q.submit([&](handler& h) {
    // kernel to produce the PARTSUPPLIER table
    h.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      constexpr int kAccumCacheSize = 15;
      CachedMemory<DBDecimal,
                   kPartTableSize,
                   kAccumCacheSize,
                   DBIdentifier> partkey_values;

      // initialize accumulator
      partkey_values.Init();

      bool done = false;

      [[intel::initiation_interval(1), intel::ivdep(kAccumCacheSize)]]
      while (!done) {
        bool valid_pipe_read;
        SupplierPartSupplierJoinedPipeData pipe_data = 
            PartSupplierPartsPipe::read(valid_pipe_read);

        done = pipe_data.done && valid_pipe_read;

        if (valid_pipe_read && !done) {
          UnrolledLoop<0, kJoinWinSize>([&](auto j) {
            SupplierPartSupplierJoined data = pipe_data.data.template get<j>();

            if (data.valid && data.nationkey == nationkey) {
              // partkeys start at 1
              DBIdentifier index = data.partkey - 1;
              DBDecimal val = data.supplycost * (DBDecimal)(data.availqty);
              auto curr_val = partkey_values.Get(index);
              partkey_values.Set(index, curr_val + val);
            }
          });
        }
      }

      // sort the {partkey, partvalue} pairs based on partvalue.
      // we will send in kSortSize - kPartTableSize dummy values with a
      // minimum value so that they are last (sorting from highest to lowest)
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < kSortSize; i++) {
        size_t key = (i < kPartTableSize) ? (i + 1) : 0;
        auto val = (i < kPartTableSize) ? partkey_values.Get(i)
                                        : std::numeric_limits<DBDecimal>::min();
        SortInPipe::write(OutputData(key, val));
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// ConsumeSort kernel
  auto consume_sort_event = q.submit([&](handler& h) {
    // output buffer accessors
    accessor partkeys_accessor(partkeys_buf, h, write_only, no_init);
    accessor values_accessor(values_buf, h, write_only, no_init);

    h.single_task<ConsumeSort>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      bool i_in_range = 0 < kSortSize;
      bool i_next_in_range = 1 < kSortSize;
      bool i_in_parttable_range = 0 < kPartTableSize;
      bool i_next_in_parttable_range = 1 < kPartTableSize;

      // grab all kSortSize elements from the sorter
      [[intel::initiation_interval(1)]]
      while (i_in_range) {
        bool pipe_read_valid;
        OutputData D = SortOutPipe::read(pipe_read_valid);

        if (pipe_read_valid) {
          if (i_in_parttable_range) {
            partkeys_accessor[i] = D.partkey;
            values_accessor[i] = D.partvalue;
          }

          i_in_range = i_next_in_range;
          i_next_in_range = i < kSortSize - 2;
          i_in_parttable_range = i_next_in_parttable_range;
          i_next_in_parttable_range = i < kPartTableSize - 2;
          i++;
        }
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// FifoSort Kernel
  auto sort_event = q.single_task<FifoSort>([=] {
    ihc::sort<SortType, kSortSize, SortInPipe, SortOutPipe>(GreaterThan());
  });
  ///////////////////////////////////////////////////////////////////////////

  // wait for kernels to finish
  produce_ps_event.wait();
  std::cout << "produce_ps_event done" << std::endl;
  join_event.wait();
  std::cout << "join_event done" << std::endl;
  compute_event.wait();
  std::cout << "compute_event done" << std::endl;
  sort_event.wait();
  std::cout << "sort_event done" << std::endl;
  consume_sort_event.wait();
  std::cout << "consume_sort_event done" << std::endl;

  high_resolution_clock::time_point host_end = high_resolution_clock::now();
  duration<double, std::milli> diff = host_end - host_start;

  // gather profiling info
  auto start_time =
      consume_sort_event
          .get_profiling_info<info::event_profiling::command_start>();
  auto end_time = consume_sort_event
          .get_profiling_info<info::event_profiling::command_end>();

  // calculating the kernel execution time in ms
  auto kernel_execution_time = (end_time - start_time) * 1e-6;

  kernel_latency = kernel_execution_time;
  total_latency = diff.count();

  return true;
}
