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
  // SupplierRow() : valid(false), suppkey(0), nationkey(0) {}
  SupplierRow() {}
  SupplierRow(bool _valid, DBIdentifier _suppkey, unsigned char _nationkey)
      : valid(_valid), suppkey(_suppkey), nationkey(_nationkey) {}

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
  PartSupplierRow()
      : valid(false), partkey(0), suppkey(0), availqty(0), supplycost(0) {}
  PartSupplierRow(bool _valid, DBIdentifier _partkey, DBIdentifier _suppkey,
                  int _availqty, DBDecimal _supplycost)
      : valid(_valid),
        partkey(_partkey),
        suppkey(_suppkey),
        availqty(_availqty),
        supplycost(_supplycost) {}

  // NOTE: this is not true, but is key to be used by MapJoin
  DBIdentifier PrimaryKey() const { return suppkey; }

  bool valid;
  DBIdentifier partkey;
  DBIdentifier suppkey;
  int availqty;
  DBDecimal supplycost;
};

//
// A row of the join SUPPLIER and PARTSUPPLIER table
//
class SupplierPartSupplierJoined {
 public:
  SupplierPartSupplierJoined()
      : valid(false), partkey(0), supplycost(0), nationkey(0) {}
  SupplierPartSupplierJoined(bool _valid, DBIdentifier _partkey, int _availqty,
                             DBDecimal _supplycost, unsigned char _nationkey)
      : valid(_valid),
        partkey(_partkey),
        availqty(_availqty),
        supplycost(_supplycost),
        nationkey(_nationkey) {}

  DBIdentifier PrimaryKey() const { return partkey; }

  void Join(const SupplierRow& s_row, const PartSupplierRow& ps_row) {
    partkey = ps_row.partkey;
    availqty = ps_row.availqty;
    supplycost = ps_row.supplycost;
    nationkey = s_row.nationkey;
  }

  bool valid;
  DBIdentifier partkey;
  int availqty;
  DBDecimal supplycost;
  unsigned char nationkey;
};

//
// The output data for this kernel (the {partkey,partvalue} pair)
// this is the datatype that is sorted by the FifoSorter
//
class OutputData {
 public:
  OutputData() {}
  // OutputData() : partkey(0), partvalue(0) {}
  OutputData(DBIdentifier _partkey, DBDecimal _partvalue)
      : partkey(_partkey), partvalue(_partvalue) {}

  bool operator<(const OutputData& t) const { return partvalue < t.partvalue; }
  bool operator>(const OutputData& t) const { return partvalue > t.partvalue; }
  bool operator==(const OutputData& t) const {
    return partvalue == t.partvalue;
  }
  bool operator!=(const OutputData& t) const {
    return partvalue != t.partvalue;
  }

  DBIdentifier partkey;
  DBDecimal partvalue;
};

constexpr int kJoinWinSize = 1;

// pipe types
using PartSupplierRowPipeData =
  StreamingData<PartSupplierRow, kJoinWinSize>;

using SupplierPartSupplierJoinedPipeData =
  StreamingData<SupplierPartSupplierJoined, kJoinWinSize>;


// pipes
using ProducePartSupplierPipe =
  pipe<class ProducePartSupplierPipeClass, PartSupplierRowPipeData>;
  
using PartSupplierPartsPipe =
  pipe<class PartSupplierPartsPipeClass, SupplierPartSupplierJoinedPipeData>;

#endif /* __PIPE_TYPES_H__ */
