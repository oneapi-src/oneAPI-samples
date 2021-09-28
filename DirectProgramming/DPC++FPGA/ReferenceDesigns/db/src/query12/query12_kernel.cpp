#include <array>
#include <limits>
#include <stdio.h>

#include "query12_kernel.hpp"
#include "pipe_types.hpp"

#include "../db_utils/MergeJoin.hpp"
#include "../db_utils/Unroller.hpp"
#include "../db_utils/Tuple.hpp"

using namespace std::chrono;

// kernel class names
class LineItemProducer;
class OrdersProducer;
class Join;
class Compute;

bool SubmitQuery12(queue& q, Database& dbinfo, DBDate low_date,
                    DBDate high_date, int shipmode1, int shipmode2,
                    std::array<DBDecimal, 2>& high_line_count,
                    std::array<DBDecimal, 2>& low_line_count,
                    double& kernel_latency, double& total_latency) {
  // create space for the input buffers
  // LINEITEM table
  buffer l_orderkey_buf(dbinfo.l.orderkey);
  buffer l_shipmode_buf(dbinfo.l.shipmode);
  buffer l_commitdate_buf(dbinfo.l.commitdate);
  buffer l_shipdate_buf(dbinfo.l.shipdate);
  buffer l_receiptdate_buf(dbinfo.l.receiptdate);

  // ORDERS table
  buffer o_orderkey_buf(dbinfo.o.orderkey);
  buffer o_orderpriority_buf(dbinfo.o.orderpriority);

  // setup the output buffers
  buffer high_line_count_buf(high_line_count);
  buffer low_line_count_buf(low_line_count);

  // number of producing iterations depends on the number of elements per cycle
  const size_t l_rows = dbinfo.l.rows;
  const size_t l_iters =
      (l_rows + kLineItemJoinWindowSize - 1) / kLineItemJoinWindowSize;
  const size_t o_rows = dbinfo.o.rows;
  const size_t o_iters =
      (o_rows + kOrderJoinWindowSize - 1) / kOrderJoinWindowSize;

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  /////////////////////////////////////////////////////////////////////////////
  //// LineItemProducer Kernel: produce the LINEITEM table
  auto produce_lineitem_event = q.submit([&](handler& h) {
    size_t l_rows = dbinfo.l.rows;
    accessor l_orderkey_accessor(l_orderkey_buf, h, read_only);
    accessor l_shipmode_accessor(l_shipmode_buf, h, read_only);
    accessor l_commitdate_accessor(l_commitdate_buf, h, read_only);
    accessor l_shipdate_accessor(l_shipdate_buf, h, read_only);
    accessor l_receiptdate_accessor(l_receiptdate_buf, h, read_only);

    h.single_task<LineItemProducer>([=]() [[intel::kernel_args_restrict]] {
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < l_iters + 1; i++) {
        bool done = (i == l_iters);
        bool valid = (i != l_iters);

        // bulk read of data from global memory
        NTuple<kLineItemJoinWindowSize, LineItemRow> data;

        UnrolledLoop<0, kLineItemJoinWindowSize>([&](auto j) {
          size_t idx = (i*kLineItemJoinWindowSize + j);
          bool in_range = idx < l_rows;
          DBIdentifier key_tmp = l_orderkey_accessor[idx];
          int shipmode = l_shipmode_accessor[idx];
          DBDate commitdate = l_commitdate_accessor[idx];
          DBDate shipdate = l_shipdate_accessor[idx];
          DBDate receiptdate = l_receiptdate_accessor[idx];

          DBIdentifier key =
              in_range ? key_tmp : std::numeric_limits<DBIdentifier>::max();

          data.get<j>() = LineItemRow(in_range, key, shipmode, commitdate,
                                      shipdate, receiptdate);
        });

        // write to pipe
        LineItemProducerPipe::write(LineItemRowPipeData(done, valid, data));
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// OrdersProducer Kernel: produce the ORDERS table
  auto produce_orders_event = q.submit([&](handler& h) {
    size_t o_rows = dbinfo.o.rows;
    accessor o_orderkey_accessor(o_orderkey_buf, h, read_only);
    accessor o_orderpriority_accessor(o_orderpriority_buf, h, read_only);

    h.single_task<OrdersProducer>([=]() [[intel::kernel_args_restrict]] {
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < o_iters + 1; i++) {
        bool done = (i == o_iters);
        bool valid = (i != o_iters);

        // bulk read of data from global memory
        NTuple<kOrderJoinWindowSize, OrdersRow> data;

        UnrolledLoop<0, kOrderJoinWindowSize>([&](auto j) {
          size_t idx = (i*kOrderJoinWindowSize + j);
          bool in_range = idx < o_rows;
          
          DBIdentifier key_tmp = o_orderkey_accessor[idx];
          int orderpriority = o_orderpriority_accessor[idx];

          DBIdentifier key =
              in_range ? key_tmp : std::numeric_limits<DBIdentifier>::max();

          data.get<j>() = OrdersRow(in_range, key, orderpriority);
        });

        // write to pipe
        OrdersProducerPipe::write(OrdersRowPipeData(done, valid, data));
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// Join kernel
  auto join_event = q.submit([&](handler& h) {
    // streaming query12 computation
    h.single_task<Join>([=]() [[intel::kernel_args_restrict]] {
      MergeJoin<OrdersProducerPipe, OrdersRow, kOrderJoinWindowSize,
                LineItemProducerPipe, LineItemRow, kLineItemJoinWindowSize,
                JoinedProducerPipe, JoinedRow>();

      // join is done, tell downstream
      JoinedProducerPipe::write(JoinedRowPipeData(true, false));
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// Compute Kernel
  auto compute_event = q.submit([&](handler& h) {
    // output write accessors
    accessor high_line_count_accessor(high_line_count_buf, h, write_only, no_init);
    accessor low_line_count_accessor(low_line_count_buf, h, write_only, no_init);

    h.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      // local accumulators
      DBDecimal high_line_count1_local = 0, high_line_count2_local = 0;
      DBDecimal low_line_count1_local = 0, low_line_count2_local = 0;
      bool done;

      [[intel::initiation_interval(1)]]
      do {
        // get joined row from pipe
        JoinedRowPipeData joined_data = JoinedProducerPipe::read();

        // upstream kernel tells this kernel when it is done
        done = joined_data.done;

        if (!done && joined_data.valid) {
          DBDecimal high_line_count1_local_tmp[kLineItemJoinWindowSize];
          DBDecimal low_line_count1_local_tmp[kLineItemJoinWindowSize];
          DBDecimal high_line_count2_local_tmp[kLineItemJoinWindowSize];
          DBDecimal low_line_count2_local_tmp[kLineItemJoinWindowSize];

          UnrolledLoop<0, kLineItemJoinWindowSize>([&](auto i) {
            // determine 'where' criteria of query
            const bool is_shipmode1 =
                (joined_data.data.get<i>().shipmode == shipmode1);
                
            const bool is_shipmode2 =
                (joined_data.data.get<i>().shipmode == shipmode2);

            const bool valid_shipmode = (is_shipmode1 || is_shipmode2);

            const bool valid_commitdate =
                (joined_data.data.get<i>().commitdate <
                 joined_data.data.get<i>().receiptdate);

            const bool valid_shipdate = (joined_data.data.get<i>().shipdate <
                                         joined_data.data.get<i>().commitdate);

            const bool receipt_within_year_of_date =
                ((joined_data.data.get<i>().receiptdate >= low_date) &&
                 (joined_data.data.get<i>().receiptdate < high_date));

            const bool urgent_or_high =
                (joined_data.data.get<i>().orderpriority == 1 ||
                 joined_data.data.get<i>().orderpriority == 2);

            const bool do_computation = joined_data.data.get<i>().valid &&
                valid_shipmode && valid_commitdate && valid_shipdate &&
                receipt_within_year_of_date;

            if (do_computation) {
              // is this order priority urgent or high
              const DBDecimal high_line_val = urgent_or_high ? 1 : 0;
              const DBDecimal low_line_val = urgent_or_high ? 0 : 1;

              high_line_count1_local_tmp[i] = is_shipmode1 ? high_line_val : 0;
              low_line_count1_local_tmp[i] = is_shipmode1 ? low_line_val : 0;

              high_line_count2_local_tmp[i] = is_shipmode2 ? high_line_val : 0;
              low_line_count2_local_tmp[i] = is_shipmode2 ? low_line_val : 0;
            } else {
              high_line_count1_local_tmp[i] = 0;
              low_line_count1_local_tmp[i] = 0;

              high_line_count2_local_tmp[i] = 0;
              low_line_count2_local_tmp[i] = 0;
            }
          });

          // this creates an adder reduction tree from *_local_tmp to *_local
          UnrolledLoop<0, kLineItemJoinWindowSize>([&](auto i) {
            high_line_count1_local += high_line_count1_local_tmp[i];
            low_line_count1_local += low_line_count1_local_tmp[i];
            high_line_count2_local += high_line_count2_local_tmp[i];
            low_line_count2_local += low_line_count2_local_tmp[i];
          });
        }
      } while (!done);

      // write back the local data to global memory
      high_line_count_accessor[0] = high_line_count1_local;
      high_line_count_accessor[1] = high_line_count2_local;
      low_line_count_accessor[0] = low_line_count1_local;
      low_line_count_accessor[1] = low_line_count2_local;
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  // wait for the Compute kernel to finish
  produce_orders_event.wait();
  produce_lineitem_event.wait();
  join_event.wait();
  compute_event.wait();

  // stop timer
  high_resolution_clock::time_point host_end = high_resolution_clock::now();
  duration<double, std::milli> diff = host_end - host_start;

  //// gather profiling info
  auto start_time =
      compute_event.get_profiling_info<info::event_profiling::command_start>();
  auto end_time =
      compute_event.get_profiling_info<info::event_profiling::command_end>();

  // calculating the kernel execution time in ms
  auto kernel_execution_time = (end_time - start_time) * 1e-6;

  kernel_latency = kernel_execution_time;
  total_latency = diff.count();

  return true;
}
