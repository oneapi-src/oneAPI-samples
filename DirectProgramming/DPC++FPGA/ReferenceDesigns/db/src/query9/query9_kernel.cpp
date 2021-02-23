#include <array>
#include <stdio.h>
#include <type_traits>
#include <vector>

#include "query9_kernel.hpp"
#include "pipe_types.hpp"

#include "../db_utils/Accumulator.hpp"
#include "../db_utils/LikeRegex.hpp"
#include "../db_utils/MapJoin.hpp"
#include "../db_utils/MergeJoin.hpp"
#include "../db_utils/Misc.hpp"
#include "../db_utils/ShannonIterator.hpp"
#include "../db_utils/Tuple.hpp"
#include "../db_utils/Unroller.hpp"
#include "../db_utils/fifo_sort.hpp"

using namespace std::chrono;

//
// NOTE: See the README file for a diagram of how the different kernels are
// connected
//

// kernel class names
class ProducerOrders;
class FilterParts;
class ProducePartSupplier;
class JoinPartSupplierSupplier;
class JoineLineItemOrders;
class FeedSort;
class FifoSort;
class ConsumeSort;
class JoinEverything;
class Compute;

/////////////////////////////////////////////////////////////////////////////
// sort configuration
using SortType = SortData;

// need to sort at most 6% of the lineitem table
constexpr int kNumSortStages = CeilLog2(kLineItemTableSize * 0.06);
constexpr int kSortSize = Pow2(kNumSortStages);

using SortInPipe = pipe<class SortInputPipe, SortType>;
using SortOutPipe = pipe<class SortOutputPipe, SortType>;

static_assert(kLineItemTableSize * 0.06 <= kSortSize,
              "Must be able to sort all part keys");
/////////////////////////////////////////////////////////////////////////////

//
// Helper function to shuffle the valid values in 'input' into 'output' using
// the bits template
// For example, consider this simple case:
//  input = {7,8}
//  if bits = 1 (2'b01), then output = {0,7}
//  if bits = 2 (2'b01), then output = {0,8}
//  if bits = 3 (2'b01), then output = {7,8}
//
template <char bits, int tuple_size, typename TupleType>
void Shuffle(NTuple<tuple_size, TupleType>& input,
             NTuple<tuple_size, TupleType>& output) {
  // get number of ones (number of valid entries) in the input
  constexpr char kNumOnes = CountOnes<char>(bits);

  // static asserts
  static_assert(tuple_size > 0,
      "tuple_size must strictly positive");
  static_assert(kNumOnes <= tuple_size,
      "Number of valid bits in bits cannot exceed the size of the tuple");

  // full crossbar to reorder valid entries of 'input'
  UnrolledLoop<0, kNumOnes>([&](auto i) {
    constexpr char pos = PositionOfNthOne<char>(i + 1, bits) - 1;
    output.template get<i>() = input.template get<pos>();
  });
}

bool SubmitQuery9(queue& q, Database& dbinfo, std::string colour,
                  std::array<DBDecimal, 25 * 2020>& sum_profit,
                  double& kernel_latency, double& total_latency) {
  // copy the regex string to character array, pad with NULL characters
  std::array<char, 11> regex_word;
  for (size_t i = 0; i < 11; i++) {
    regex_word[i] = (i < colour.size()) ? colour[i] : '\0';
  }

  // create space for the input buffers
  // the REGEX
  buffer<char,1> regex_word_buf(regex_word.size());

  // PARTS
  buffer<char,1> p_name_buf(dbinfo.p.name.size());

  // SUPPLIER
  buffer<unsigned char,1> s_nationkey_buf(dbinfo.s.nationkey.size());

  // PARTSUPPLIER
  buffer<DBIdentifier,1> ps_partkey_buf(dbinfo.ps.partkey.size());
  buffer<DBIdentifier,1> ps_suppkey_buf(dbinfo.ps.suppkey.size());
  buffer<DBDecimal,1> ps_supplycost_buf(dbinfo.ps.supplycost.size());

  // ORDERS
  buffer<DBIdentifier,1> o_orderkey_buf(dbinfo.o.orderkey.size());
  buffer<DBDate,1> o_orderdate_buf(dbinfo.o.orderdate.size());

  // LINEITEM
  buffer<DBIdentifier,1> l_orderkey_buf(dbinfo.l.orderkey.size());
  buffer<DBIdentifier,1> l_partkey_buf(dbinfo.l.partkey.size());
  buffer<DBIdentifier,1> l_suppkey_buf(dbinfo.l.suppkey.size());
  buffer<DBDecimal,1> l_quantity_buf(dbinfo.l.quantity.size());
  buffer<DBDecimal,1> l_extendedprice_buf(dbinfo.l.extendedprice.size());
  buffer<DBDecimal,1> l_discount_buf(dbinfo.l.discount.size());

  // a convenient lamda to make the explicit copy code less verbose
  auto submit_copy = [&](auto& buf, const auto& host_data) {
    return q.submit([&](handler &h) {
      accessor accessor(buf, h, write_only, noinit);
      h.copy(host_data, accessor);
    });
  };

  // start the transers of the input buffers
  event copy_regex_word = submit_copy(regex_word_buf, regex_word.data());

  event copy_p_name = submit_copy(p_name_buf, dbinfo.p.name.data());

  event copy_o_orderkey =
    submit_copy(o_orderkey_buf, dbinfo.o.orderkey.data());
  event copy_o_orderdate = 
    submit_copy(o_orderdate_buf, dbinfo.o.orderdate.data());

  event copy_l_orderkey = submit_copy(l_orderkey_buf, dbinfo.l.orderkey.data());
  event copy_l_partkey = submit_copy(l_partkey_buf, dbinfo.l.partkey.data());
  event copy_l_suppkey = submit_copy(l_suppkey_buf, dbinfo.l.suppkey.data());

  event copy_s_nationkey = 
    submit_copy(s_nationkey_buf, dbinfo.s.nationkey.data());

  event copy_ps_partkey = 
    submit_copy(ps_partkey_buf, dbinfo.ps.partkey.data());
  event copy_ps_suppkey = 
    submit_copy(ps_suppkey_buf, dbinfo.ps.suppkey.data());
  event copy_ps_supplycost = 
    submit_copy(ps_supplycost_buf, dbinfo.ps.supplycost.data());

  event copy_l_quantity = submit_copy(l_quantity_buf, dbinfo.l.quantity.data());
  event copy_l_extendedprice = 
    submit_copy(l_extendedprice_buf, dbinfo.l.extendedprice.data());
  event copy_l_discount = submit_copy(l_discount_buf, dbinfo.l.discount.data());

  // setup the output buffer (the profit for each nation and year)
  buffer sum_profit_buf(sum_profit);

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  /////////////////////////////////////////////////////////////////////////////
  //// FilterParts Kernel:
  ////    Filter the PARTS table and produce the filtered LINEITEM table
  auto filter_parts_event = q.submit([&](handler& h) {
    // this kernel depends on the copy of the regex, the PARTS table, and a 
    // portion of the LINEITEM table (just the keys).
    h.depends_on({copy_regex_word, copy_p_name, copy_l_orderkey, copy_l_partkey,
                  copy_l_suppkey});

    // REGEX word accessor
    accessor regex_word_accessor(regex_word_buf, h, read_only);

    // PARTS table accessors
    size_t p_rows = dbinfo.p.rows;
    accessor p_name_accessor(p_name_buf, h, read_only);

    // LINEITEM table accessors
    size_t l_rows = dbinfo.l.rows;
    accessor l_orderkey_accessor(l_orderkey_buf, h, read_only);
    accessor l_partkey_accessor(l_partkey_buf, h, read_only);
    accessor l_suppkey_accessor(l_suppkey_buf, h, read_only);

    // kernel to filter parts table based on REGEX
    h.single_task<FilterParts>([=]() [[intel::kernel_args_restrict]] {
      // a map where the key is the partkey and the value is whether
      // that partkeys name matches the given regex
      bool partkeys_matching_regex[kPartTableSize + 1];

      ///////////////////////////////////////////////
      //// Stage 1
      // find valid parts with REGEX
      LikeRegex<11, 55> regex[kRegexFilterElementsPerCycle];

      // initialize regex word
      for (size_t i = 0; i < 11; i++) {
        const char c = regex_word_accessor[i];
        UnrolledLoop<0, kRegexFilterElementsPerCycle>([&](auto re) { 
          regex[re].word[i] = c;
        });
      }

      // stream in rows of PARTS table and check partname against REGEX
      [[intel::ivdep]]
      for (size_t i = 0; i < p_rows; i += kRegexFilterElementsPerCycle) {
        UnrolledLoop<0, kRegexFilterElementsPerCycle>([&](auto re) {
          const size_t idx = i + re;
          const bool idx_range = idx < p_rows;

          // read in partkey
          // valid partkeys in range [1,kPartTableSize]
          const DBIdentifier partkey = idx_range ? idx + 1 : 0;

          // read in regex string
          UnrolledLoop<0, 55>([&](auto k) {
            regex[re].str[k] = idx_range ? p_name_accessor[idx * 55 + k] : 0;
          });

          // run regex matching
          regex[re].Match();

          // mark valid partkey
          partkeys_matching_regex[partkey] = regex[re].Contains();
        });
      }
      ///////////////////////////////////////////////

      ///////////////////////////////////////////////
      //// Stage 2
      // read in the LINEITEM table (kLineItemJoinWinSize rows at a time)
      // row is valid if its PARTKEY matched the REGEX
      for (size_t i = 0; i < l_rows; i += kLineItemJoinWinSize) {
        // bulk read of data from global memory
        NTuple<kLineItemJoinWinSize, LineItemMinimalRow> data;

        UnrolledLoop<0, kLineItemJoinWinSize>([&](auto j) {
          bool in_range = i + j < l_rows;
          DBIdentifier orderkey = in_range ? l_orderkey_accessor[i + j] : 0;
          DBIdentifier partkey = in_range ? l_partkey_accessor[i + j] : 0;
          DBIdentifier suppkey = in_range ? l_suppkey_accessor[i + j] : 0;

          bool matches_partkey_name_regex = partkeys_matching_regex[partkey];
          bool data_is_valid = in_range && matches_partkey_name_regex;

          data.get<j>() = LineItemMinimalRow(data_is_valid, i + j, orderkey,
                                             partkey, suppkey);
        });

        // write to pipe
        LineItemPipe::write(LineItemMinimalRowPipeData(false, true, data));
      }
      ///////////////////////////////////////////////
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// ProducerOrders Kernel: produce the ORDERS table
  auto producer_orders_event = q.submit([&](handler& h) {
    // this kernel depends on the copy of the ORDERS table
    h.depends_on({copy_o_orderkey, copy_o_orderdate});

    // ORDERS table accessors
    size_t o_rows = dbinfo.o.rows;
    accessor o_orderkey_accessor(o_orderkey_buf, h, read_only);
    accessor o_orderdate_accessor(o_orderdate_buf, h, read_only);

    // produce ORDERS table (kOrdersJoinWinSize rows at a time)
    h.single_task<ProducerOrders>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < o_rows; i += kOrdersJoinWinSize) {
        // bulk read of data from global memory
        NTuple<kOrdersJoinWinSize, OrdersRow> data;

        UnrolledLoop<0, kOrdersJoinWinSize>([&](auto j) {
          bool in_range = (i + j) < o_rows;
          DBIdentifier orderkey = in_range ? o_orderkey_accessor[i + j]
                                  : std::numeric_limits<DBIdentifier>::max();
          DBDate orderdate = in_range ? o_orderdate_accessor[i + j] : 0;

          data.get<j>() = OrdersRow(in_range, orderkey, orderdate);
        });

        // write to pipe
        OrdersPipe::write(OrdersRowPipeData(false, true, data));
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// JoinLineItemOrders Kernel: join the LINEITEM and ORDERS table
  auto join_lineitem_orders_event = q.submit([&](handler& h) {
    // this kernel doesn't depend on any memory copies; all data is fed
    // to/from it via SYCL pipes (see the README).

    // ORDERS table accessors
    size_t o_rows = dbinfo.o.rows;

    // LINEITEM table accessors
    size_t l_rows = dbinfo.l.rows;

    // kernel to join LINEITEM and ORDERS table
    h.single_task<JoineLineItemOrders>([=]() [[intel::kernel_args_restrict]] {
      // JOIN LINEITEM and ORDERS table
      // callbacks for reading and writing data
      // readers
      GenericPipeReader<OrdersPipe,
                        OrdersRowPipeData> orders_reader;
      GenericPipeReader<LineItemPipe,
                        LineItemMinimalRowPipeData> lineitem_reader;

      // writer
      GenericPipeWriter<LineItemOrdersPipe,
                        LineItemOrdersMinimalJoinedPipeData> joined_writer;

      // declare the joiner
      MergeJoiner<OrdersRow, kOrdersJoinWinSize, LineItemMinimalRow,
                  kLineItemJoinWinSize, LineItemOrdersMinimalJoined>
          joiner(o_rows, l_rows);

      // do join
      joiner.Go(orders_reader, lineitem_reader, joined_writer);

      // join is done, tell downstream
      LineItemOrdersPipe::write(
          LineItemOrdersMinimalJoinedPipeData(true, false));
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// JoinPartSupplierSupplier Kernel: join the PARTSUPPLIER and SUPPLIER tables
  auto join_partsupplier_supplier_event = q.submit([&](handler& h) {
    // this kernel depends on the copy of the SUPPLIERS table
    h.depends_on({copy_s_nationkey});

    // SUPPLIER table accessors
    size_t s_rows = dbinfo.s.rows;
    accessor s_nationkey_accessor(s_nationkey_buf, h, read_only);

    // PARTSUPPLIER table size
    size_t ps_rows = dbinfo.ps.rows;

    // kernel to join partsupplier and supplier tables
    h.single_task<JoinPartSupplierSupplier>(
          [=]() [[intel::kernel_args_restrict]] {
      // callbacks for reading and writing data
      // reader
      GenericPipeReader<PartSupplierPipe,
                        PartSupplierRowPipeData> partsupplier_reader;

      // writer
      GenericPipeWriter<PartSupplierPartsPipe,
                        SupplierPartSupplierJoinedPipeData> joined_writer;

      // +1 is to account for fact that SUPPKEY is [1,kSF*10000]
      MapJoiner<SupplierRow, kSupplierTableSize + 1, PartSupplierRow,
                kPartSupplierDuplicatePartkeys, SupplierPartSupplierJoined>
          mapJoiner(ps_rows);

      ///////////////////////////////////////////////
      //// Stage 1
      // populate MapJoiner map
      // why a map? keys may not be sequential
      [[intel::ivdep]]
      for (size_t i = 0; i < s_rows; i++) {
        // read in supplier and nation key
        // NOTE: based on TPCH docs, SUPPKEY is guaranteed
        // to be unique in range [1:kSF*10000]
        DBIdentifier s_suppkey = i + 1;
        unsigned char s_nationkey = s_nationkey_accessor[i];

        mapJoiner.map.Set(s_suppkey, SupplierRow(true, s_suppkey, s_nationkey));
      }
      ///////////////////////////////////////////////

      ///////////////////////////////////////////////
      //// Stage 2
      // MAPJOIN PARTSUPPLIER and SUPPLIER tables by suppkey
      mapJoiner.Go(partsupplier_reader, joined_writer);
      ///////////////////////////////////////////////

      // tell downstream we are done
      PartSupplierPartsPipe::write(
        SupplierPartSupplierJoinedPipeData(true, false));
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// ProducePartSupplier Kernel: produce the PARTSUPPLIER table
  auto produce_part_supplier_event = q.submit([&](handler& h) {
    // this kernel depends on the copy of the PARTSUPPLIER table
    h.depends_on({copy_ps_partkey, copy_ps_suppkey, copy_ps_supplycost});

    // PARTSUPPLIER table accessors
    size_t ps_rows = dbinfo.ps.rows;
    accessor ps_partkey_accessor(ps_partkey_buf, h, read_only);
    accessor ps_suppkey_accessor(ps_suppkey_buf, h, read_only);
    accessor ps_supplycost_accessor(ps_supplycost_buf, h, read_only);

    // kernel to produce the PARTSUPPLIER table
    h.single_task<ProducePartSupplier>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < ps_rows; i += kPartSupplierDuplicatePartkeys) {
        // bulk read of data from global memory
        NTuple<kPartSupplierDuplicatePartkeys, PartSupplierRow> data;

        UnrolledLoop<0, kPartSupplierDuplicatePartkeys>([&](auto j) {
          bool in_range = (i + j) < ps_rows;
          DBIdentifier partkey = in_range ? ps_partkey_accessor[i + j] : 0;
          DBIdentifier suppkey = in_range ? ps_suppkey_accessor[i + j] : 0;
          DBDecimal supplycost = in_range ? ps_supplycost_accessor[i + j] : 0;

          data.get<j>() = 
              PartSupplierRow(in_range, partkey, suppkey, supplycost);
        });

        // write to pipe
        PartSupplierPipe::write(PartSupplierRowPipeData(false, true, data));
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// Compute Kernel: do the final computation on the data
  auto computation_kernel_event = q.submit([&](handler& h) {
    // this kernel depends on the copy of the LINEITEM table
    h.depends_on({copy_l_quantity, copy_l_extendedprice, copy_l_discount});

    // LINEITEM table accessors
    accessor l_quantity_accessor(l_quantity_buf, h, read_only);
    accessor l_extendedprice_accessor(l_extendedprice_buf, h, read_only);
    accessor l_discount_accessor(l_discount_buf, h, read_only);

    // output accessors
    accessor sum_profit_accessor(sum_profit_buf, h, write_only, noinit);

    h.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      // the accumulators
      constexpr int ACCUM_CACHE_SIZE = 7;
      BRAMAccumulator<DBDecimal, (25 * 7), ACCUM_CACHE_SIZE, unsigned char>
          sum_profit_local[kFinalDataMaxSize];

      // initialize the accumulators
      UnrolledLoop<0, kFinalDataMaxSize>([&](auto j) {
        sum_profit_local[j].Init();
      });

      bool done = false;

      [[intel::ivdep(ACCUM_CACHE_SIZE)]]
      do {
        bool valid;
        FinalPipeData pipe_data = FinalPipe::read(valid);
        done = pipe_data.done && valid;

        const bool pipeDataValid = !pipe_data.done && valid && pipe_data.valid;

        UnrolledLoop<0, kFinalDataMaxSize>([&](auto j) {
          FinalData D = pipe_data.data.get<j>();

          bool D_valid = pipeDataValid && D.valid;
          unsigned int D_idx = D.lineitemIdx;

          // grab LINEITEM data from global memory and compute 'amount'
          DBDecimal quantity=0, extendedprice=0, discount=0, supplycost=0;
          if(D_valid) {
            quantity = l_quantity_accessor[D_idx];
            extendedprice = l_extendedprice_accessor[D_idx];
            discount = l_discount_accessor[D_idx];
            supplycost = D.supplycost;
          }

          // Why quantity x 100? So we can divide 'amount' by 100*100 later
          DBDecimal amount = (extendedprice * (100 - discount)) -
                              (supplycost * quantity * 100);

          // compute index based on order year and nation
          // See Date.hpp
          unsigned int orderyear = (D.orderdate >> 9) & 0x07FFFFF;
          unsigned int nation = D.nationkey;
          unsigned char idx = (orderyear - 1992) * 25 + nation;

          unsigned char idx_final = D_valid ? idx : 0;
          DBDecimal amount_final = D_valid ? amount : 0;

          sum_profit_local[j].Accumulate(idx_final, amount_final);
        });
      } while (!done);

      // push back the accumulated data to global memory
      for (size_t n = 0; n < 25; n++) {
        for (size_t y = 0; y < 7; y++) {
          size_t in_idx = y * 25 + n;
          size_t out_idx = (y + 1992) * 25 + n;

          DBDecimal amount = 0;

          UnrolledLoop<0, kFinalDataMaxSize>([&](auto j) {
            amount += sum_profit_local[j].Get(in_idx);
          });

          sum_profit_accessor[out_idx] = amount;
        }
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// FeedSort Kernel: kernel to filter out invalid data and feed the sorter
  auto feed_sort_event = q.submit([&](handler& h) {
    h.single_task<FeedSort>([=]() [[intel::kernel_args_restrict]] {
      bool done = false;
      size_t num_rows = 0;

      do {
        // get data from upstream
        bool valid;
        LineItemOrdersMinimalJoinedPipeData pipe_data = 
            LineItemOrdersPipe::read(valid);
        done = pipe_data.done && valid;

        if (!done && valid && pipe_data.valid) {
          NTuple<kLineItemOrdersJoinWinSize, LineItemOrdersMinimalJoined>
              shuffle_data;
          unsigned char valid_count = 0;
          char valid_bits = 0;

          // convert the 'valid' bits in the tuple to a bitset (valid_bits)
          UnrolledLoop<0, kLineItemOrdersJoinWinSize>([&](auto i) {
            constexpr char mask = 1 << i;
            valid_bits |= pipe_data.data.get<i>().valid ? mask : 0;
          });

          // full crossbar to do the shuffling from pipe_data to shuffle_data
          UnrolledLoop<0, Pow2(kLineItemOrdersJoinWinSize)>([&](auto i) {
            if (valid_bits == i) {
              Shuffle<i, kLineItemOrdersJoinWinSize,
                      LineItemOrdersMinimalJoined>(pipe_data.data,
                                                   shuffle_data);
              valid_count = CountOnes<char>(i);
            }
          });

          // Send the data to sorter.
          // The idea here is that this loop executes in the range
          // [0,kLineItemOrdersJoinWinSize] times.
          // However, we know that at most 6% of the data will match the filter
          // and go to the sorter. So, that means for every ~16 pieces of
          // data, we expect <1 will match the filter and go to the sorter.
          // Therefore, so long as kLineItemOrdersJoinWinSize <= 16
          // this loop will, on average, execute ONCE per outer loop iteration
          // (i.e. statistically, valid_count=1 for every 16 pieces of data).
          // NOTE: for this loop to get good throughput it is VERY important to:
          //    A) Apply the [[intel::speculated_iterations(0)]] attribute
          //    B) Explicitly bound the loop iterations
          // For an explanation why, see the optimize_inner_loops tutorial.
          [[intel::speculated_iterations(0)]]
          for (char i = 0; i < valid_count && 
                i < kLineItemOrdersJoinWinSize; i++) {
            UnrolledLoop<0, kLineItemOrdersJoinWinSize>([&](auto j) {
              if (j == i) {
                SortInPipe::write(SortData(shuffle_data.get<j>()));
              }
            });
          }
          
          num_rows += valid_count;
        }
      } while (!done);

      // send in pad data to ensure we send in exactly kSortSize elements
      ShannonIterator<int, 3> i(num_rows, kSortSize);

      while (i.InRange()) {
        SortInPipe::write(
            SortData(0, std::numeric_limits<DBIdentifier>::max(), 0, 0));

        i.Step();
      }

      // drain the input pipe
      while (!done) {
        bool valid;
        LineItemOrdersMinimalJoinedPipeData pipe_data = 
            LineItemOrdersPipe::read(valid);
        done = pipe_data.done && valid;
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// ConsumeSort Kernel: consume the output of the sorter
  auto consume_sort_event = q.submit([&](handler& h) {
    // this kernel doesn't depend on any memory copies; all data is fed
    // to/from it via SYCL pipes (see the README).

    h.single_task<ConsumeSort>([=]() [[intel::kernel_args_restrict]] {
      bool done = false;
      size_t num_rows = 0;

      // read out data from the sorter until 'done' signal from upstream
      do {
        bool valid;
        SortData in_data = SortOutPipe::read(valid);
        done = (in_data.partkey == std::numeric_limits<DBIdentifier>::max()) &&
               valid;
        num_rows += valid ? 1 : 0;

        if (!done && valid) {
          NTuple<1, LineItemOrdersMinimalJoined> out_data;
          out_data.get<0>() = LineItemOrdersMinimalJoined(
              true, in_data.lineitemIdx, in_data.partkey, in_data.suppkey,
              in_data.orderdate);

          LineItemOrdersSortedPipe::write(
              LineItemOrdersMinimalSortedPipeData(false, true, out_data));
        }
      } while (!done);

      // tell downstream kernel that the sort is done
      LineItemOrdersSortedPipe::write(
          LineItemOrdersMinimalSortedPipeData(true, false));

      // drain the data we don't care about from the sorter
      ShannonIterator<int, 3> i(num_rows, kSortSize);
      while (i.InRange()) {
        bool valid;
        (void)SortOutPipe::read(valid);

        if (valid) {
          i.Step();
        }
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// FifoSort Kernel: the sorter
  auto sort_event = q.submit([&](handler& h) {
    // this kernel doesn't depend on any memory copies; all data is fed
    // to/from it via SYCL pipes (see the README).
    
    h.single_task<FifoSort>([=]() [[intel::kernel_args_restrict]] {
      ihc::sort<SortType, kSortSize, SortInPipe, SortOutPipe>(ihc::LessThan());
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// JoinEverything Kernel: join the sorted
  ////    LINEITEM+ORDERS with SUPPLIER+PARTSUPPLIER
  auto join_li_o_s_ps_event = q.submit([&](handler& h) {
    // this kernel doesn't depend on any memory copies; all data is fed
    // to/from it via SYCL pipes (see the README).

    // PARTSUPPLIER table accessors
    size_t ps_rows = dbinfo.ps.rows;

    // LINEITEM table accessors
    size_t l_rows = dbinfo.l.rows;

    h.single_task<JoinEverything>([=]() [[intel::kernel_args_restrict]] {
      // callbacks for reading and writing data
      // readers
      GenericPipeReader<PartSupplierPartsPipe,
                        SupplierPartSupplierJoinedPipeData> s_ps_reader;
      GenericPipeReader<LineItemOrdersSortedPipe,
                        LineItemOrdersMinimalSortedPipeData> li_o_parts_reader;

      // writer
      GenericPipeWriter<FinalPipe,
                        FinalPipeData> joined_writer;

      // declare the joiner
      // The PartSupplier table has is sorted by partkey and has EXACTLY
      // 4 entries per part key
      DuplicateMergeJoiner<
          SupplierPartSupplierJoined, kPartSupplierDuplicatePartkeys,
          LineItemOrdersMinimalJoined, 1, FinalData, true>
          joiner(ps_rows, l_rows);

      // do join
      joiner.Go(s_ps_reader, li_o_parts_reader, joined_writer);

      // join is done, tell downstream
      FinalPipe::write(FinalPipeData(true, false));
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  // wait for kernel to finish
  filter_parts_event.wait();
  computation_kernel_event.wait();

  high_resolution_clock::time_point host_end = high_resolution_clock::now();
  duration<double, std::milli> diff = host_end - host_start;

  // gather profiling info
  auto filter_parts_start =
      filter_parts_event
          .get_profiling_info<info::event_profiling::command_start>();
  auto computation_end =
      computation_kernel_event
          .get_profiling_info<info::event_profiling::command_end>();

  // calculating the kernel execution time in ms
  auto kernel_execution_time = (computation_end - filter_parts_start) * 1e-6;

  kernel_latency = kernel_execution_time;
  total_latency = diff.count();

  return true;
}
