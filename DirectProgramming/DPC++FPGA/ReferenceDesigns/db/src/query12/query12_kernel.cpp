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
  buffer<DBIdentifier,1> l_orderkey_buf(dbinfo.l.orderkey.size());
  buffer<int,1> l_shipmode_buf(dbinfo.l.shipmode.size());
  buffer<DBDate,1> l_commitdate_buf(dbinfo.l.commitdate.size());
  buffer<DBDate,1> l_shipdate_buf(dbinfo.l.shipdate.size());
  buffer<DBDate,1> l_receiptdate_buf(dbinfo.l.receiptdate.size());

  // ORDERS table
  buffer<DBIdentifier,1> o_orderkey_buf(dbinfo.o.orderkey.size());
  buffer<int,1> o_orderpriority_buf(dbinfo.o.orderpriority.size());

  // a convenient lamda to make the explicit copy code less verbose
  auto submit_copy = [&](auto& buf, const auto& host_data) {
    return q.submit([&](handler &h) {
      accessor accessor(buf, h, write_only, noinit);
      h.copy(host_data, accessor);
    });
  };

  // start the transers of the input buffers
  event copy_l_orderkey = submit_copy(l_orderkey_buf, dbinfo.l.orderkey.data());
  event copy_l_shipmode = submit_copy(l_shipmode_buf, dbinfo.l.shipmode.data());
  event copy_l_commitdate = 
    submit_copy(l_commitdate_buf, dbinfo.l.commitdate.data());
  event copy_l_shipdate = submit_copy(l_shipdate_buf, dbinfo.l.shipdate.data());
  event copy_l_receiptdate = 
    submit_copy(l_receiptdate_buf, dbinfo.l.receiptdate.data());

  event copy_o_orderkey =
    submit_copy(o_orderkey_buf, dbinfo.o.orderkey.data());
  event copy_o_orderpriority = 
    submit_copy(o_orderpriority_buf, dbinfo.o.orderpriority.data());

  // setup the output buffers
  buffer high_line_count_buf(high_line_count);
  buffer low_line_count_buf(low_line_count);

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  /////////////////////////////////////////////////////////////////////////////
  //// LineItemProducer Kernel: produce the LINEITEM table
  auto produce_lineitem_event = q.submit([&](handler& h) {
    // this kernel depends on the memory transfer for the LINEITEM table
    h.depends_on({copy_l_orderkey, copy_l_shipmode, copy_l_commitdate,
                  copy_l_shipdate, copy_l_receiptdate});

    size_t l_rows = dbinfo.l.rows;
    accessor l_orderkey_accessor(l_orderkey_buf, h, read_only);
    accessor l_shipmode_accessor(l_shipmode_buf, h, read_only);
    accessor l_commitdate_accessor(l_commitdate_buf, h, read_only);
    accessor l_shipdate_accessor(l_shipdate_buf, h, read_only);
    accessor l_receiptdate_accessor(l_receiptdate_buf, h, read_only);

    h.single_task<LineItemProducer>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < l_rows; i += kLineItemJoinWindowSize) {
        // bulk read of data from global memory
        NTuple<kLineItemJoinWindowSize, LineItemRow> data;

        UnrolledLoop<0, kLineItemJoinWindowSize>([&](auto j) {
          bool in_range = (i + j) < l_rows;
          DBIdentifier key = in_range ? l_orderkey_accessor[i + j]
                              : std::numeric_limits<DBIdentifier>::max();
          int shipmode = in_range ? l_shipmode_accessor[i + j] : 0;
          DBDate commitdate = in_range ? l_commitdate_accessor[i + j] : 0;
          DBDate shipdate = in_range ? l_shipdate_accessor[i + j] : 0;
          DBDate receiptdate = in_range ? l_receiptdate_accessor[i + j] : 0;

          data.get<j>() = LineItemRow(in_range, key, shipmode, commitdate,
                                      shipdate, receiptdate);
        });

        // write to pipe
        LineItemProducerPipe::write(LineItemRowPipeData(false, true, data));
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// OrdersProducer Kernel: produce the ORDERS table
  auto produce_orders_event = q.submit([&](handler& h) {
    // this kernel depends on the memory transfer for the ORDERS table
    h.depends_on({copy_o_orderkey, copy_o_orderpriority});

    size_t o_rows = dbinfo.o.rows;
    accessor o_orderkey_accessor(o_orderkey_buf, h, read_only);
    accessor o_orderpriority_accessor(o_orderpriority_buf, h, read_only);

    h.single_task<OrdersProducer>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < o_rows; i += kOrderJoinWindowSize) {
        // bulk read of data from global memory
        NTuple<kOrderJoinWindowSize, OrdersRow> data;

        UnrolledLoop<0, kOrderJoinWindowSize>([&](auto j) {
          bool in_range = (i + j) < o_rows;
          DBIdentifier key = in_range ? o_orderkey_accessor[i + j]
                              : std::numeric_limits<DBIdentifier>::max();
          int orderpriority = in_range ? o_orderpriority_accessor[i + j] : 0;

          data.get<j>() = OrdersRow(in_range, key, orderpriority);
        });

        // write to pipe
        OrdersProducerPipe::write(OrdersRowPipeData(false, true, data));
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// Join kernel
  auto join_event = q.submit([&](handler& h) {
    // this kernel doesn't depend on any memory copies; all data is fed
    // to/from it via SYCL pipes (see the README)

    int o_rows = dbinfo.o.rows;
    int l_rows = dbinfo.l.rows;

    // streaming query12 computation
    h.single_task<Join>([=]() [[intel::kernel_args_restrict]] {
      //// callbacks for reading and writing data
      // reader callback for the Orders table (table 1 for MergeJoiner)
      GenericPipeReader<OrdersProducerPipe,
                        OrdersRowPipeData> orders_reader;

      // reader callback for the LineItem table (table 2 for MergeJoiner)
      GenericPipeReader<LineItemProducerPipe,
                        LineItemRowPipeData> lineitem_reader;


      // the writer callback function
      GenericPipeWriter<JoinedProducerPipe,
                        JoinedRowPipeData> joined_writer;

      // declare the joiner
      MergeJoiner<OrdersRow, kOrderJoinWindowSize, LineItemRow,
                  kLineItemJoinWindowSize, JoinedRow>
          joiner(o_rows, l_rows);

      // do the join
      joiner.Go(orders_reader, lineitem_reader, joined_writer);

      // join is done, tell downstream
      JoinedProducerPipe::write(JoinedRowPipeData(true, false));
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //// Compute Kernel
  auto compute_event = q.submit([&](handler& h) {
    // output write accessors
    accessor high_line_count_accessor(high_line_count_buf, h, write_only, noinit);
    accessor low_line_count_accessor(low_line_count_buf, h, write_only, noinit);

    h.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      // local accumulators
      DBDecimal high_line_count1_local = 0, high_line_count2_local = 0;
      DBDecimal low_line_count1_local = 0, low_line_count2_local = 0;
      bool done;

      do {
        // get joined row from pipe
        bool pipe_valid;
        JoinedRowPipeData joined_data = JoinedProducerPipe::read(pipe_valid);

        // upstream kernel tells this kernel when it is done
        done = joined_data.done && pipe_valid;

        if (!done && pipe_valid) {
          DBDecimal high_line_count1_local_tmp[kLineItemJoinWindowSize];
          DBDecimal high_line_count2_local_tmp[kLineItemJoinWindowSize];
          DBDecimal low_line_count1_local_tmp[kLineItemJoinWindowSize];
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

            const bool do_computation = !done && joined_data.data.get<i>().valid
                && valid_shipmode && valid_commitdate && valid_shipdate
                && receipt_within_year_of_date;

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
