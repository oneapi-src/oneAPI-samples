#include <stdio.h>

#include <type_traits>

#include "query11_kernel.hpp"
#include "pipe_types.hpp"
#include "../db_utils/Accumulator.hpp"
#include "../db_utils/MapJoin.hpp"
#include "../db_utils/Misc.hpp"
#include "../db_utils/Tuple.hpp"
#include "../db_utils/Unroller.hpp"
#include "../db_utils/ShannonIterator.hpp"
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
  buffer<DBIdentifier,1> s_suppkey_buf(dbinfo.s.suppkey.size());
  buffer<unsigned char,1> s_nationkey_buf(dbinfo.s.nationkey.size());
  
  // PARTSUPPLIER
  buffer<DBIdentifier,1> ps_partkey_buf(dbinfo.ps.partkey.size());
  buffer<DBIdentifier,1> ps_suppkey_buf(dbinfo.ps.suppkey.size());
  buffer<int,1> ps_availqty_buf(dbinfo.ps.availqty.size());
  buffer<DBDecimal,1> ps_supplycost_buf(dbinfo.ps.supplycost.size());

  // a convenient lamda to make the explicit copy code less verbose
  auto submit_copy = [&](auto& buf, const auto& host_data) {
    return q.submit([&](handler &h) {
      accessor accessor(buf, h, write_only, noinit);
      h.copy(host_data, accessor);
    });
  };

  // start the transers of the input buffers
  event copy_s_suppkey = submit_copy(s_suppkey_buf, dbinfo.s.suppkey.data());
  event copy_s_nationkey = 
    submit_copy(s_nationkey_buf, dbinfo.s.nationkey.data());

  event copy_ps_partkey = 
    submit_copy(ps_partkey_buf, dbinfo.ps.partkey.data());
  event copy_ps_suppkey = 
    submit_copy(ps_suppkey_buf, dbinfo.ps.suppkey.data());
  event copy_ps_availqty = 
    submit_copy(ps_availqty_buf, dbinfo.ps.availqty.data());
  event copy_ps_supplycost = 
    submit_copy(ps_supplycost_buf, dbinfo.ps.supplycost.data());

  // setup the output buffers
  buffer partkeys_buf(partkeys);
  buffer values_buf(values);

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  ///////////////////////////////////////////////////////////////////////////
  //// ProducePartSupplier Kernel
  auto produce_ps_event = q.submit([&](handler& h) {
    // this kernel depends on the transfers of the PARTSUPPLIER table
    h.depends_on({copy_ps_partkey, copy_ps_suppkey, copy_ps_availqty,
                  copy_ps_supplycost});

    // PARTSUPPLIER table accessors
    size_t ps_rows = dbinfo.ps.rows;
    accessor ps_partkey_accessor(ps_partkey_buf, h, read_only);
    accessor ps_suppkey_accessor(ps_suppkey_buf, h, read_only);
    accessor ps_availqty_accessor(ps_availqty_buf, h, read_only);
    accessor ps_supplycost_accessor(ps_supplycost_buf, h, read_only);

    // kernel to produce the PARTSUPPLIER table
    h.single_task<ProducePartSupplier>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < ps_rows; i += kJoinWinSize) {
        // bulk read of data from global memory
        NTuple<kJoinWinSize, PartSupplierRow> data;

        UnrolledLoop<0, kJoinWinSize>([&](auto j) {
          bool in_range = (i + j) < ps_rows;
          DBIdentifier partkey = in_range ? ps_partkey_accessor[i + j] : 0;
          DBIdentifier suppkey = in_range ? ps_suppkey_accessor[i + j] : 0;
          int availqty = in_range ? ps_availqty_accessor[i + j] : 0;
          DBDecimal supplycost = in_range ? ps_supplycost_accessor[i + j] : 0;

          data.get<j>() =
              PartSupplierRow(in_range, partkey, suppkey, availqty, supplycost);
        });

        // write to pipe
        ProducePartSupplierPipe::write(
            PartSupplierRowPipeData(false, true, data));
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// JoinPartSupplierParts Kernel
  auto join_event = q.submit([&](handler& h) {
    // this kernel depends on the transfers of the PARTSUPPLIER table
    h.depends_on({copy_s_suppkey, copy_s_nationkey});

    // PARTSUPPLIER table accessors
    size_t ps_rows = dbinfo.ps.rows;

    // SUPPLIER table accessors
    size_t s_rows = dbinfo.s.rows;
    accessor s_suppkey_accessor(s_suppkey_buf, h, read_only);
    accessor s_nationkey_accessor(s_nationkey_buf, h, read_only);

    h.single_task<JoinPartSupplierParts>([=]() [[intel::kernel_args_restrict]] {
      // callbacks for reading and writing data
      // reader
      GenericPipeReader<ProducePartSupplierPipe,
                        PartSupplierRowPipeData> partsupplier_reader;
      // writer
      GenericPipeWriter<PartSupplierPartsPipe,
                        SupplierPartSupplierJoinedPipeData> joined_writer;

      // +1 is to account for fact that SUPPKEY is [1,kSF*10000]
      MapJoiner<SupplierRow, kSupplierTableSize + 1, PartSupplierRow,
                kJoinWinSize, SupplierPartSupplierJoined>
          mapJoiner(ps_rows);

      // initialize the mapper
      mapJoiner.Init();

      // populate MapJoiner map
      // why a map? keys may not be sequential
      [[intel::ivdep]]
      for (size_t i = 0; i < s_rows; i++) {
        // read in supplier and nation key
        // NOTE: based on TPCH docs, SUPPKEY is guaranteed to be unique
        // in the range [1:kSF*10000]
        DBIdentifier s_suppkey = s_suppkey_accessor[i];
        unsigned char s_nationkey = s_nationkey_accessor[i];

        mapJoiner.map.Set(s_suppkey, SupplierRow(true,s_suppkey,s_nationkey));
      }

      // MAPJOIN PARTSUPPLIER and SUPPLIER tables by suppkey
      mapJoiner.Go(partsupplier_reader, joined_writer);
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// Compute Kernel
  auto compute_event = q.submit([&](handler& h) {
    // this kernel doesn't depend on any memory copies; all data is fed
    // to/from it via SYCL pipes (see the README)

    // PARTSUPPLIER table accessors
    size_t ps_rows = dbinfo.ps.rows;

    // kernel to produce the PARTSUPPLIER table
    h.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      constexpr int ACCUM_CACHE_SIZE = 5;
      BRAMAccumulator<DBDecimal,
                      kPartTableSize,
                      ACCUM_CACHE_SIZE,
                      DBIdentifier> partkey_values;

      // initialize accumulator
      partkey_values.Init();

      [[intel::ivdep]]
      for (size_t i = 0; i < ps_rows; i += kJoinWinSize) {
        SupplierPartSupplierJoinedPipeData pipe_data = 
            PartSupplierPartsPipe::read();

        if (pipe_data.valid) {
          UnrolledLoop<0, kJoinWinSize>([&](auto j) {
            SupplierPartSupplierJoined data = pipe_data.data.template get<j>();

            if (data.valid && data.nationkey == nationkey) {
              // partkeys start at 1
              DBIdentifier index = data.partkey - 1;
              DBDecimal val = data.supplycost*(DBDecimal)(data.availqty);
              partkey_values.Accumulate(index, val);
            }
          });
        }
      }

      // sort the {partkey, partvalue} pairs based on partvalue.
      // send in first kPartTableSize valid pairs
      for (size_t i = 0; i < kPartTableSize; i++) {
        SortInPipe::write(OutputData(i + 1, partkey_values.Get(i)));
      }

      // The sort kernel is expecting kSortSize elements, but we only have
      // kPartTableSize (kPartTableSize <= kSortSize) elements.
      // So, send it kSortSize-kPartTableSize pieces of invalid data.
      // We are sorting elements by partvalue in descending order,
      // so use 'min' for partvalue and 0 for partkey (valid partkeys are in
      // [1,PARTSIZE]). Using min ensures these elements come out of the
      // sorter LAST.
      for (size_t i = 0; i < kSortSize - kPartTableSize; i++) {
        SortInPipe::write(
            OutputData(0, std::numeric_limits<DBDecimal>::min()));
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// ConsumeSort kernel
  auto consume_sort_event = q.submit([&](handler& h) {
    // output buffer accessors
    accessor partkeys_accessor(partkeys_buf, h, write_only, noinit);
    accessor values_accessor(values_buf, h, write_only, noinit);

    h.single_task<ConsumeSort>([=]() [[intel::kernel_args_restrict]] {
      // use a ShannonIterator to track how many items
      // we have seen from the sorter
      // Using a ShannonIterator sacrifices a (marginal)
      // amount of area to improve Fmax/II
      ShannonIterator<int, 3> i(0, kSortSize);

      // grab all kSortSize elements from the sorter
      while (i.InRange()) {
        bool valid;
        OutputData D = SortOutPipe::read(valid);

        if (valid) {
          // first kPartTableSize elements are valid
          // i.Index() is the index of the iterator
          if (i.Index() < kPartTableSize) {
            partkeys_accessor[i.Index()] = D.partkey;
            values_accessor[i.Index()] = D.partvalue;
          }

          // increment the ShannonIterator
          i.Step();
        }
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// FifoSort Kernel
  auto sort_event = q.submit([&](handler& h) {
    h.single_task<FifoSort>([=]() [[intel::kernel_args_restrict]] {
      ihc::sort<SortType, kSortSize, SortInPipe, SortOutPipe>(GreaterThan());
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  // wait for kernels to finish
  consume_sort_event.wait();

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
