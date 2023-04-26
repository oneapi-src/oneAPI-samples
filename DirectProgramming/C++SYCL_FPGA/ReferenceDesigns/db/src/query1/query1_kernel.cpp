#include <stdio.h>

#include "query1_kernel.hpp"

#include "../db_utils/Accumulator.hpp"
#include "../db_utils/Tuple.hpp"
#include "../db_utils/Unroller.hpp"

using namespace std::chrono;

// how many elements to compute per cycle
#if defined(FPGA_SIMULATOR)
constexpr int kElementsPerCycle = 2;
#else
constexpr int kElementsPerCycle = 12;
#endif

// the kernel name
class Query1;

bool SubmitQuery1(queue& q, Database& dbinfo, DBDate low_date,
                  std::array<DBDecimal, kQuery1OutSize>& sum_qty,
                  std::array<DBDecimal, kQuery1OutSize>& sum_base_price,
                  std::array<DBDecimal, kQuery1OutSize>& sum_disc_price,
                  std::array<DBDecimal, kQuery1OutSize>& sum_charge,
                  std::array<DBDecimal, kQuery1OutSize>& avg_qty,
                  std::array<DBDecimal, kQuery1OutSize>& avg_price,
                  std::array<DBDecimal, kQuery1OutSize>& avg_discount,
                  std::array<DBDecimal, kQuery1OutSize>& count,
                  double& kernel_latency, double& total_latency) {
  // create space for input buffers
  buffer quantity_buf(dbinfo.l.quantity);
  buffer extendedprice_buf(dbinfo.l.extendedprice);
  buffer discount_buf(dbinfo.l.discount);
  buffer tax_buf(dbinfo.l.tax);
  buffer returnflag_buf(dbinfo.l.returnflag);
  buffer linestatus_buf(dbinfo.l.linestatus);
  buffer shipdate_buf(dbinfo.l.shipdate);

  // setup the output buffers
  buffer sum_qty_buf(sum_qty);
  buffer sum_base_price_buf(sum_base_price);
  buffer sum_disc_price_buf(sum_disc_price);
  buffer sum_charge_buf(sum_charge);
  buffer avg_qty_buf(avg_qty);
  buffer avg_price_buf(avg_price);
  buffer avg_discount_buf(avg_discount);
  buffer count_buf(count);

  const int rows = dbinfo.l.rows; 
  const size_t iters = (rows + kElementsPerCycle - 1) / kElementsPerCycle;

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  /////////////////////////////////////////////////////////////////////////////
  //// Query1 Kernel
  auto event = q.submit([&](handler& h) {
    // read accessors
    accessor quantity_accessor(quantity_buf, h, read_only);
    accessor extendedprice_accessor(extendedprice_buf, h, read_only);
    accessor discount_accessor(discount_buf, h, read_only);
    accessor tax_accessor(tax_buf, h, read_only);
    accessor returnflag_accessor(returnflag_buf, h, read_only);
    accessor linestatus_accessor(linestatus_buf, h, read_only);
    accessor shipdate_accessor(shipdate_buf, h, read_only);

    // write accessors
    accessor sum_qty_accessor(sum_qty_buf, h, write_only, no_init);
    accessor sum_base_price_accessor(sum_base_price_buf, h, write_only, no_init);
    accessor sum_disc_price_accessor(sum_disc_price_buf, h, write_only, no_init);
    accessor sum_charge_accessor(sum_charge_buf, h, write_only, no_init);
    accessor avg_qty_accessor(avg_qty_buf, h, write_only, no_init);
    accessor avg_price_accessor(avg_price_buf, h, write_only, no_init);
    accessor avg_discount_accessor(avg_discount_buf, h, write_only, no_init);
    accessor count_accessor(count_buf, h, write_only, no_init);

    h.single_task<Query1>([=]() [[intel::kernel_args_restrict]] {
      // local accumulation buffers
      RegisterAccumulator<DBDecimal, 6, unsigned char> sum_qty_local;
      RegisterAccumulator<DBDecimal, 6, unsigned char> sum_base_price_local;
      RegisterAccumulator<DBDecimal, 6, unsigned char> sum_disc_price_local;
      RegisterAccumulator<DBDecimal, 6, unsigned char> sum_charge_local;
      RegisterAccumulator<DBDecimal, 6, unsigned char> avg_discount_local;
      RegisterAccumulator<DBDecimal, 6, unsigned char> count_local;

      // initialize the accumulators
      sum_qty_local.Init();
      sum_base_price_local.Init();
      sum_disc_price_local.Init();
      sum_charge_local.Init();
      avg_discount_local.Init();
      count_local.Init();

      // stream each row in the DB (kElementsPerCycle rows at a time)
      [[intel::initiation_interval(1)]]
      for (size_t r = 0; r < iters; r++) {
        // locals
        DBDecimal qty[kElementsPerCycle];
        DBDecimal extendedprice[kElementsPerCycle];
        DBDecimal discount[kElementsPerCycle];
        DBDecimal tax[kElementsPerCycle];
        DBDecimal disc_price_tmp[kElementsPerCycle];
        DBDecimal charge_tmp[kElementsPerCycle];
        DBDecimal count_tmp[kElementsPerCycle];
        unsigned char out_idx[kElementsPerCycle];
        bool row_valid[kElementsPerCycle];

        // multiple elements per cycle
        UnrolledLoop<0, kElementsPerCycle>([&](auto p) {
          // is data in range of the table
          // (data size may not be divisible by kElementsPerCycle)
          size_t idx = r * kElementsPerCycle + p;
          bool in_range = idx < rows;

          // get this rows shipdate
          DBDate shipdate = shipdate_accessor[idx];

          // determine if the row is valid
          row_valid[p] = in_range && (shipdate <= low_date);

          // read or set values based on the validity of the data
          qty[p] = quantity_accessor[idx];
          extendedprice[p] = extendedprice_accessor[idx];
          discount[p] = discount_accessor[idx];
          tax[p] = tax_accessor[idx];
          char rf = returnflag_accessor[idx];
          char ls = linestatus_accessor[idx];
          count_tmp[p] = 1;

          // convert returnflag and linestatus into an index
          unsigned char rf_idx;
          if (rf == 'R') {
            rf_idx = 0;
          } else if (rf == 'A') {
            rf_idx = 1;
          } else {  // == 'N'
            rf_idx = 2;
          }
          unsigned char ls_idx;
          if (ls == 'O') {
            ls_idx = 0;
          } else {  // == 'F'
            ls_idx = 1;
          }
          out_idx[p] = ls_idx * kReturnFlagSize + rf_idx;

          // intermediate calculations
          disc_price_tmp[p] = extendedprice[p] * (100 - discount[p]);
          charge_tmp[p] =
              extendedprice[p] * (100 - discount[p]) * (100 + tax[p]);
        });

        // reduction accumulation
        UnrolledLoop<0, kElementsPerCycle>([&](auto p) {
          sum_qty_local.Accumulate(out_idx[p],
                                   row_valid[p] ? qty[p] : 0);
          sum_base_price_local.Accumulate(out_idx[p],
                                          row_valid[p] ? extendedprice[p] : 0);
          sum_disc_price_local.Accumulate(out_idx[p],
                                          row_valid[p] ? disc_price_tmp[p] : 0);
          sum_charge_local.Accumulate(out_idx[p],
                                      row_valid[p] ? charge_tmp[p] : 0);
          count_local.Accumulate(out_idx[p],
                                 row_valid[p] ? count_tmp[p] : 0);
          avg_discount_local.Accumulate(out_idx[p],
                                        row_valid[p] ? discount[p] : 0);
        });
      }

// perform averages and push back to global memory
#pragma unroll
      for (size_t i = 0; i < kQuery1OutSize; i++) {
        DBDecimal count = count_local.Get(i);

        sum_qty_accessor[i] = sum_qty_local.Get(i);
        sum_base_price_accessor[i] = sum_base_price_local.Get(i);
        sum_disc_price_accessor[i] = sum_disc_price_local.Get(i);
        sum_charge_accessor[i] = sum_charge_local.Get(i);

        avg_qty_accessor[i] = (count == 0) ? 0 : (sum_qty_local.Get(i) / count);
        avg_price_accessor[i] =
            (count == 0) ? 0 : (sum_base_price_local.Get(i) / count);
        avg_discount_accessor[i] =
            (count == 0) ? 0 : (avg_discount_local.Get(i) / count);

        count_accessor[i] = count;
      }
    });
  });
  /////////////////////////////////////////////////////////////////////////////

  // wait for kernel to finish
  event.wait();

  high_resolution_clock::time_point host_end = high_resolution_clock::now();
  duration<double, std::milli> diff = host_end - host_start;

  // gather profiling info
  auto kernel_start_time =
      event.get_profiling_info<info::event_profiling::command_start>();
  auto kernel_end_time =
      event.get_profiling_info<info::event_profiling::command_end>();

  // calculating the kernel execution time in ms
  auto kernel_execution_time = (kernel_end_time - kernel_start_time) * 1e-6;

  kernel_latency = kernel_execution_time;
  total_latency = diff.count();

  return true;
}
