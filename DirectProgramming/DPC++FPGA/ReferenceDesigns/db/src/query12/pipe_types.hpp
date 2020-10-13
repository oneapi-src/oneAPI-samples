#ifndef __PIPE_TYPES_H__
#define __PIPE_TYPES_H__
#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "../db_utils/StreamingData.hpp"
#include "../dbdata.hpp"

using namespace sycl;

//
// A single row of the ORDERS table
// with a subset of the columns (needed for this query)
//
class OrdersRow {
 public:
  OrdersRow() : valid(false), orderkey(0), orderpriority(0) {}
  OrdersRow(bool _valid, DBIdentifier _key, int _orderpriority)
      : valid(_valid), orderkey(_key), orderpriority(_orderpriority) {}

  DBIdentifier PrimaryKey() const { return orderkey; }

  bool valid;
  DBIdentifier orderkey;
  int orderpriority;
};

//
// A single row of the LINEITEM table
// with a subset of the columns (needed for this query)
//
class LineItemRow {
 public:
  LineItemRow()
      : valid(false),
        orderkey(0),
        shipmode(0),
        commitdate(0),
        shipdate(0),
        receiptdate(0) {}

  LineItemRow(bool _valid, DBIdentifier _key, int _shipmode,
              DBDate _commitdate, DBDate _shipdate, DBDate _receiptdate)
      : valid(_valid),
        orderkey(_key),
        shipmode(_shipmode),
        commitdate(_commitdate),
        shipdate(_shipdate),
        receiptdate(_receiptdate) {}

  DBIdentifier PrimaryKey() const { return orderkey; }

  bool valid;
  DBIdentifier orderkey;
  int shipmode;
  DBDate commitdate;
  DBDate shipdate;
  DBDate receiptdate;
};

//
// A row of the join LINEITEM and ORDERS table
//
class JoinedRow {
 public:
  JoinedRow()
      : valid(false),
        orderpriority(0),
        shipmode(0),
        commitdate(0),
        shipdate(0),
        receiptdate(0) {}
    
  JoinedRow(bool _valid, DBIdentifier _key, int _orderpriority, int _shipmode,
            DBDate _commitdate, DBDate _shipdate, DBDate _receiptdate)
      : valid(_valid),
        orderpriority(_orderpriority),
        shipmode(_shipmode),
        commitdate(_commitdate),
        shipdate(_shipdate),
        receiptdate(_receiptdate) {}

  void Join(const OrdersRow& o_row, const LineItemRow& l_row) {
    orderpriority = o_row.orderpriority;
    shipmode = l_row.shipmode;
    commitdate = l_row.commitdate;
    shipdate = l_row.shipdate;
    receiptdate = l_row.receiptdate;
  }

  bool valid;
  int orderpriority;
  int shipmode;
  DBDate commitdate;
  DBDate shipdate;
  DBDate receiptdate;
};

// JOIN window sizes
constexpr int kOrderJoinWindowSize = 4;
constexpr int kLineItemJoinWindowSize = 16;

// pipe data types
using OrdersRowPipeData = StreamingData<OrdersRow, kOrderJoinWindowSize>;
using LineItemRowPipeData = StreamingData<LineItemRow, kLineItemJoinWindowSize>;
using JoinedRowPipeData = StreamingData<JoinedRow, kLineItemJoinWindowSize>;

// the pipes
using OrdersProducerPipe =
  pipe<class OrdersProducerPipeClass, OrdersRowPipeData>;

using LineItemProducerPipe =
  pipe<class LineItemProducerPipeClass, LineItemRowPipeData>;

using JoinedProducerPipe =
  pipe<class JoinedProducerPipeClass, JoinedRowPipeData>;

#endif /* __PIPE_TYPES_H__ */
