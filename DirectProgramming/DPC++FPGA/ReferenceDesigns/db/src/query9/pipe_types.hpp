#ifndef __PIPE_TYPES_H__
#define __PIPE_TYPES_H__
#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "../db_utils/StreamingData.hpp"
#include "../dbdata.hpp"

using namespace sycl;

//
// A single row of the SUPPLIER table
// with a subset of the columns (needed for this query)
//
class SupplierRow {
 public:
  SupplierRow() {}
  // SupplierRow()
  //    : valid(false), suppkey(0), nationkey(0) {}
  SupplierRow(bool v_valid, DBIdentifier v_suppkey, unsigned char v_nationkey)
      : valid(v_valid), suppkey(v_suppkey), nationkey(v_nationkey) {}

  DBIdentifier PrimaryKey() const { return suppkey; }

  bool valid;
  DBIdentifier suppkey;
  unsigned char nationkey;
};

//
// A single row of the PARTSUPPLIER table
// with a subset of the columns (needed for this query)
//
class PartSupplierRow {
 public:
  PartSupplierRow() : valid(false), partkey(0), suppkey(0), supplycost(0) {}
  PartSupplierRow(bool v_valid, DBIdentifier v_partkey, DBIdentifier v_suppkey,
                  DBDecimal v_supplycost)
      : valid(v_valid),
        partkey(v_partkey),
        suppkey(v_suppkey),
        supplycost(v_supplycost) {}

  // NOTE: this is not true, but is key to be used by MapJoin
  DBIdentifier PrimaryKey() const { return suppkey; }

  bool valid;
  DBIdentifier partkey;
  DBIdentifier suppkey;
  DBDecimal supplycost;
};

//
// A row of the join SUPPLIER and PARTSUPPLIER table
//
class SupplierPartSupplierJoined {
 public:
  SupplierPartSupplierJoined()
      : valid(false), partkey(0), suppkey(0), supplycost(0), nationkey(0) {}
  SupplierPartSupplierJoined(bool v_valid, DBIdentifier v_partkey,
                             DBIdentifier v_suppkey, DBDecimal v_supplycost,
                             unsigned char v_nationkey)
      : valid(v_valid),
        partkey(v_partkey),
        suppkey(v_suppkey),
        supplycost(v_supplycost),
        nationkey(v_nationkey) {}

  DBIdentifier PrimaryKey() const { return partkey; }

  void Join(const SupplierRow& s_row, const PartSupplierRow& ps_row) {
    partkey = ps_row.partkey;
    suppkey = s_row.suppkey;
    supplycost = ps_row.supplycost;
    nationkey = s_row.nationkey;
  }

  bool valid;
  DBIdentifier partkey;
  DBIdentifier suppkey;
  DBDecimal supplycost;
  unsigned char nationkey;
};

//
// A single row of the ORDERS table
// with a subset of the columns (needed for this query)
//
class OrdersRow {
 public:
  OrdersRow() : valid(false), orderkey(0), orderdate(0) {}
  OrdersRow(bool v_valid, DBIdentifier v_orderkey, DBDate v_orderdate)
      : valid(v_valid), orderkey(v_orderkey), orderdate(v_orderdate) {}

  DBIdentifier PrimaryKey() const { return orderkey; }

  bool valid;
  DBIdentifier orderkey;
  DBDate orderdate;
};

//
// A single row of the LINEITEM table
// with a subset of the columns (needed for this query)
//
class LineItemMinimalRow {
 public:
  LineItemMinimalRow()
      : valid(false), idx(0), orderkey(0), partkey(0), suppkey(0) {}
  LineItemMinimalRow(bool v_valid, unsigned int v_idx, DBIdentifier v_orderkey,
                     DBIdentifier v_partkey, DBIdentifier v_suppkey)
      : valid(v_valid),
        idx(v_idx),
        orderkey(v_orderkey),
        partkey(v_partkey),
        suppkey(v_suppkey) {}

  DBIdentifier PrimaryKey() const { return orderkey; }

  bool valid;
  unsigned int idx;
  DBIdentifier orderkey, partkey, suppkey;
};

//
// A row of the join LINEITEM and ORDERS table
//
class LineItemOrdersMinimalJoined {
 public:
  LineItemOrdersMinimalJoined()
      : valid(false), lineitemIdx(0), partkey(0), suppkey(0), orderdate(0) {}
  LineItemOrdersMinimalJoined(bool v_valid, unsigned int v_lineitem_idx,
                              DBIdentifier v_partkey, DBIdentifier v_suppkey,
                              DBDate v_orderdate)
      : valid(v_valid),
        lineitemIdx(v_lineitem_idx),
        partkey(v_partkey),
        suppkey(v_suppkey),
        orderdate(v_orderdate) {}

  DBIdentifier PrimaryKey() { return partkey; }

  void Join(const OrdersRow& o_row, const LineItemMinimalRow& li_row) {
    lineitemIdx = li_row.idx;
    partkey = li_row.partkey;
    suppkey = li_row.suppkey;
    orderdate = o_row.orderdate;
  }

  bool valid;
  unsigned int lineitemIdx;
  DBIdentifier partkey;
  DBIdentifier suppkey;
  DBDate orderdate;
};

//
// Datatype to be sent to be sorted by the FifoSorter
//
class SortData {
 public:
  SortData() {}
  SortData(unsigned int v_lineitem_idx, DBIdentifier v_partkey,
           DBIdentifier v_suppkey, DBDate v_orderdate)
      : lineitemIdx(v_lineitem_idx),
        partkey(v_partkey),
        suppkey(v_suppkey),
        orderdate(v_orderdate) {}
  SortData(const LineItemOrdersMinimalJoined& d)
      : lineitemIdx(d.lineitemIdx),
        partkey(d.partkey),
        suppkey(d.suppkey),
        orderdate(d.orderdate) {}

  bool operator<(const SortData& t) const { return partkey < t.partkey; }
  bool operator>(const SortData& t) const { return partkey > t.partkey; }
  bool operator<=(const SortData& t) const { return partkey <= t.partkey; }
  bool operator>=(const SortData& t) const { return partkey >= t.partkey; }
  bool operator==(const SortData& t) const { return partkey == t.partkey; }
  bool operator!=(const SortData& t) const { return partkey != t.partkey; }

  unsigned int lineitemIdx;
  DBIdentifier partkey;
  DBIdentifier suppkey;
  DBDate orderdate;
};

//
// The final data used to compute the 'amount'
//
class FinalData {
 public:
  FinalData()
      : valid(false),
        partkey(0),
        lineitemIdx(0),
        orderdate(0),
        supplycost(0),
        nationkey(0) {}
  
  FinalData(bool v_valid, DBIdentifier v_partkey, unsigned int v_lineitem_idx,
            DBDate v_orderdate, DBDecimal v_supplycost,
            unsigned char v_nationkey)
      : valid(v_valid),
        partkey(v_partkey),
        lineitemIdx(v_lineitem_idx),
        orderdate(v_orderdate),
        supplycost(v_supplycost),
        nationkey(v_nationkey) {}

  DBIdentifier PrimaryKey() { return partkey; }

  void Join(const SupplierPartSupplierJoined& s_ps_row,
            const LineItemOrdersMinimalJoined& li_o_row) {
    valid = s_ps_row.suppkey == li_o_row.suppkey;

    partkey = s_ps_row.partkey;
    lineitemIdx = li_o_row.lineitemIdx;
    orderdate = li_o_row.orderdate;
    supplycost = s_ps_row.supplycost;
    nationkey = s_ps_row.nationkey;
  }

  bool valid;
  DBIdentifier partkey;
  unsigned int lineitemIdx;
  DBDate orderdate;
  DBDecimal supplycost;
  unsigned char nationkey;
};

// joining window sizes
constexpr int kRegexFilterElementsPerCycle = 1;
constexpr int kOrdersJoinWinSize = 1;
constexpr int kLineItemJoinWinSize = 2;
constexpr int kLineItemOrdersJoinWinSize = kLineItemJoinWinSize;
constexpr int kLineItemOrdersSortedWinSize = 1;
constexpr int kPartSupplierDuplicatePartkeys = 4;
constexpr int kFinalDataMaxSize =
    kPartSupplierDuplicatePartkeys * kLineItemOrdersSortedWinSize;

// pipe data
using LineItemMinimalRowPipeData =
    StreamingData<LineItemMinimalRow, kLineItemJoinWinSize>;

using OrdersRowPipeData = 
    StreamingData<OrdersRow, kOrdersJoinWinSize>;

using LineItemOrdersMinimalJoinedPipeData =
    StreamingData<LineItemOrdersMinimalJoined, kLineItemOrdersJoinWinSize>;

using LineItemOrdersMinimalSortedPipeData =
    StreamingData<LineItemOrdersMinimalJoined, 1>;

using PartSupplierRowPipeData =
    StreamingData<PartSupplierRow, kPartSupplierDuplicatePartkeys>;

using SupplierPartSupplierJoinedPipeData =
    StreamingData<SupplierPartSupplierJoined, kPartSupplierDuplicatePartkeys>;

using FinalPipeData =
    StreamingData<FinalData, kFinalDataMaxSize>;

// pipes
using LineItemPipe =
    sycl::pipe<class LineItemPipeClass, LineItemMinimalRowPipeData>;

using OrdersPipe = 
    sycl::pipe<class OrdersPipeClass, OrdersRowPipeData>;

using LineItemOrdersPipe =
    sycl::pipe<class LineItemOrdersPipeClass, 
              LineItemOrdersMinimalJoinedPipeData>;

using LineItemOrdersSortedPipe = 
  sycl::pipe<class LineItemOrdersSortedPipeClass,
              LineItemOrdersMinimalSortedPipeData>;

using PartSupplierPartsPipe =
  sycl::pipe<class PartSupplierPartsPipeClass,
              SupplierPartSupplierJoinedPipeData>;

using PartSupplierPipe =
    sycl::pipe<class PartSupplierPipeClass, PartSupplierRowPipeData>;

using FinalPipe =
    sycl::pipe<class FinalPipeClass, FinalPipeData>;

#endif /* __PIPE_TYPES_H__ */
