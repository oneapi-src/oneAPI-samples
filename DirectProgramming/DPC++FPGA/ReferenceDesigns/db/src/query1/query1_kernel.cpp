#include <stdio.h>

#include "query1_kernel.hpp"

#include "../db_utils/Accumulator.hpp"
#include "../db_utils/Tuple.hpp"
#include "../db_utils/Unroller.hpp"

using namespace std::chrono;

// how many elements to compute per cycle
constexpr int kElementsPerCycle = 12;

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
  buffer<DBDecimal,1> quantity_buf(dbinfo.l.quantity.size());
  buffer<DBDecimal,1> extendedprice_buf(dbinfo.l.extendedprice.size());
  buffer<DBDecimal,1> discount_buf(dbinfo.l.discount.size());
  buffer<DBDecimal,1> tax_buf(dbinfo.l.tax.size());
  buffer<char,1> returnflag_buf(dbinfo.l.returnflag.size());
  buffer<char,1> linestatus_buf(dbinfo.l.linestatus.size());
  buffer<DBDate,1> shipdate_buf(dbinfo.l.shipdate.size());

  // a convenient lamda to make the explicit copy code less verbose
  auto submit_copy = [&](auto& buf, const auto& host_data) {
    return q.submit([&](handler &h) {
      accessor accessor(buf, h, write_only, noinit);
      h.copy(host_data, accessor);
    });
  };

  // start the transers of the input buffers
  event copy_quantity = submit_copy(quantity_buf, dbinfo.l.quantity.data());
  event copy_extendedprice = 
    submit_copy(extendedprice_buf, dbinfo.l.extendedprice.data());
  event copy_discount = submit_copy(discount_buf, dbinfo.l.discount.data());
  event copy_tax = submit_copy(tax_buf, dbinfo.l.tax.data());
  event copy_returnflag = 
    submit_copy(returnflag_buf, dbinfo.l.returnflag.data());
  event copy_linestatus = 
    submit_copy(linestatus_buf, dbinfo.l.linestatus.data());
  event copy_shipdate = 
    submit_copy(shipdate_buf, dbinfo.l.shipdate.data());

  // setup the output buffers
  buffer sum_qty_buf(sum_qty);
  buffer sum_base_price_buf(sum_base_price);
  buffer sum_disc_price_buf(sum_disc_price);
  buffer sum_charge_buf(sum_charge);
  buffer avg_qty_buf(avg_qty);
  buffer avg_price_buf(avg_price);
  buffer avg_discount_buf(avg_discount);
  buffer count_buf(count);

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  /////////////////////////////////////////////////////////////////////////////
  //// Query1 Kernel
  auto event = q.submit([&](handler& h) {
    // this kernel depends on all the memory transfers from earlier
    h.depends_on({copy_quantity, copy_extendedprice, copy_discount, copy_tax,
                 copy_returnflag, copy_linestatus, copy_shipdate});

    // read accessors
    int rows = dbinfo.l.rows;
    accessor quantity_accessor(quantity_buf, h, read_only);
    accessor extendedprice_accessor(extendedprice_buf, h, read_only);
    accessor discount_accessor(discount_buf, h, read_only);
    accessor tax_accessor(tax_buf, h, read_only);
    accessor returnflag_accessor(returnflag_buf, h, read_only);
    accessor linestatus_accessor(linestatus_buf, h, read_only);
    accessor shipdate_accessor(shipdate_buf, h, read_only);

    // write accessors
    accessor sum_qty_accessor(sum_qty_buf, h, write_only, noinit);
    accessor sum_base_price_accessor(sum_base_price_buf, h, write_only, noinit);
    accessor sum_disc_price_accessor(sum_disc_price_buf, h, write_only, noinit);
    accessor sum_charge_accessor(sum_charge_buf, h, write_only, noinit);
    accessor avg_qty_accessor(avg_qty_buf, h, write_only, noinit);
    accessor avg_price_accessor(avg_price_buf, h, write_only, noinit);
    accessor avg_discount_accessor(avg_discount_buf, h, write_only, noinit);
    accessor count_accessor(count_buf, h, write_only, noinit);

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
      for (size_t r = 0; r < rows; r += kElementsPerCycle) {
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
          bool in_range = (r + p) < rows;

          // get this rows shipdate
          DBDate shipdate = in_range ? shipdate_accessor[r + p] : 0;

          // determine if the row is valid
          row_valid[p] = in_range && (shipdate <= low_date);

          // read or set values based on the validity of the data
          qty[p] = in_range ? quantity_accessor[r + p] : 0;
          extendedprice[p] = in_range ? extendedprice_accessor[r + p] : 0;
          discount[p] = in_range ? discount_accessor[r + p] : 0;
          tax[p] = in_range ? tax_accessor[r + p] : 0;
          char rf = in_range ? returnflag_accessor[r + p] : 0;
          char ls = in_range ? linestatus_accessor[r + p] : 0;
          count_tmp[p] = in_range ? 1 : 0;

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
