#ifndef __DBDATA_HPP__
#define __DBDATA_HPP__
#pragma once

#include <array>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// database types
typedef unsigned int DBIdentifier;
typedef unsigned int DBUint;
typedef unsigned int DBInt;
typedef long long DBDecimal;
typedef unsigned int DBDate;

// Set the scale factor
//
// The default (and only option) scale factor for
// emulation is 0.01; a scale factor of 1 for emulation
// takes far too long.
//
// The default scale factor for hardware is 1. However,
// the SF_SMALL flag allows the hardware design to be compiled
// with a scale factor of 0.01
#if defined(FPGA_EMULATOR) || defined(SF_SMALL)
constexpr float kSF = 0.01f;
#else
constexpr float kSF = 1.0f;
#endif

// ensure the selected scale factor is supported
static_assert((kSF == 0.01f || kSF == 1.0f), "Unsupported Scale Factor (kSF)");

// table sizes based on the Scale Factor (kSF)
constexpr int kPartTableSize = kSF * 200000;
constexpr int kPartSupplierTableSize = kSF * 800000;
constexpr int kOrdersTableSize = kSF * 1500000;
constexpr int kSupplierTableSize = kSF * 10000;
constexpr int kCustomerTableSize = kSF * 150000;

// LINEITEM table is not a strict multiple of kSF
constexpr int LineItemTableSizeFnc() {
  if (kSF == 0.01f) {
    return 60175;
  } else if (kSF == 1.0f) {
    return 6001215;
  } else {
    return 0;  // error, should be caught by kSF static_assert
  }
}

constexpr int kLineItemTableSize = LineItemTableSizeFnc();
constexpr int kNationTableSize = 25;
constexpr int kRegionTableSize = 5;

constexpr int kLineStatusSize = 2;
constexpr int kReturnFlagSize = 3;
constexpr int kQuery1OutSize = kReturnFlagSize * kLineStatusSize;

// helpers
DBDate DateFromString(std::string& date_str);
int ShipmodeStrToInt(std::string& shipmode_str);

// LINEITEM table
struct LineItemTable {
  std::vector<DBIdentifier> orderkey;
  std::vector<DBIdentifier> partkey;
  std::vector<DBIdentifier> suppkey;
  std::vector<DBInt> linenumber;
  std::vector<DBDecimal> quantity;
  std::vector<DBDecimal> extendedprice;
  std::vector<DBDecimal> discount;
  std::vector<DBDecimal> tax;
  std::vector<char> returnflag;
  std::vector<char> linestatus;
  std::vector<DBDate> shipdate;
  std::vector<DBDate> commitdate;
  std::vector<DBDate> receiptdate;
  std::vector<char> shipinstruct;
  std::vector<int> shipmode;
  std::vector<char> comment;

  size_t rows;
};

// ORDERS table
struct OrdersTable {
  std::vector<DBIdentifier> orderkey;
  std::vector<DBIdentifier> custkey;
  std::vector<char> orderstatus;
  std::vector<DBDecimal> totalprice;
  std::vector<DBDate> orderdate;
  std::vector<int> orderpriority;
  std::vector<char> clerk;
  std::vector<int> shippriority;
  std::vector<char> comment;

  size_t rows;
};

// PARTS table
struct PartsTable {
  std::vector<DBIdentifier> partkey;
  std::vector<char> name;
  std::vector<char> mfgr;
  std::vector<char> brand;
  std::vector<char> type;
  std::vector<int> size;
  std::vector<char> container;
  std::vector<DBDecimal> retailprice;
  std::vector<char> comment;

  size_t rows;
};

// SUPPLIER table
struct SupplierTable {
  std::vector<DBIdentifier> suppkey;
  std::vector<char> name;
  std::vector<char> address;
  std::vector<unsigned char> nationkey;
  std::vector<char> phone;
  std::vector<DBDecimal> acctbal;
  std::vector<char> comment;

  size_t rows;
};

// PARTSUPP table
struct PartSupplierTable {
  std::vector<DBIdentifier> partkey;
  std::vector<DBIdentifier> suppkey;
  std::vector<int> availqty;
  std::vector<DBDecimal> supplycost;
  std::vector<char> comment;

  size_t rows;
};

// NATION table
struct NationTable {
  std::vector<DBIdentifier> nationkey;
  std::vector<char> name;
  std::vector<DBIdentifier> regionkey;
  std::vector<char> comment;

  std::unordered_map<std::string, unsigned char> name_key_map;
  std::array<std::string, kNationTableSize> key_name_map;
  size_t rows;
};

// the database
struct Database {
  LineItemTable l;
  OrdersTable o;
  PartsTable p;
  SupplierTable s;
  PartSupplierTable ps;
  NationTable n;

  bool Parse(std::string db_root_dir);

  // validation functions
  bool ValidateSF();

  bool ValidateQ1(std::string db_root_dir,
                  std::array<DBDecimal, 3 * 2>& sum_qty,
                  std::array<DBDecimal, 3 * 2>& sum_base_price,
                  std::array<DBDecimal, 3 * 2>& sum_disc_price,
                  std::array<DBDecimal, 3 * 2>& sum_charge,
                  std::array<DBDecimal, 3 * 2>& avg_qty,
                  std::array<DBDecimal, 3 * 2>& avg_price,
                  std::array<DBDecimal, 3 * 2>& avg_discount,
                  std::array<DBDecimal, 3 * 2>& count);

  bool ValidateQ9(std::string db_root_dir,
                  std::array<DBDecimal, 25 * 2020>& sum_profit);

  bool ValidateQ11(std::string db_root_dir, std::vector<DBIdentifier>& partkeys,
                   std::vector<DBDecimal>& partkey_values);

  bool ValidateQ12(std::string db_root_dir,
                   std::array<DBDecimal, 2> high_line_count,
                   std::array<DBDecimal, 2> low_line_count);

  // print functions
  void PrintQ1(std::array<DBDecimal, 3 * 2>& sum_qty,
               std::array<DBDecimal, 3 * 2>& sum_base_price,
               std::array<DBDecimal, 3 * 2>& sum_disc_price,
               std::array<DBDecimal, 3 * 2>& sum_charge,
               std::array<DBDecimal, 3 * 2>& avg_qty,
               std::array<DBDecimal, 3 * 2>& avg_price,
               std::array<DBDecimal, 3 * 2>& avg_discount,
               std::array<DBDecimal, 3 * 2>& count);

  void PrintQ9(std::array<DBDecimal, 25 * 2020>& sum_profit);

  void PrintQ11(std::vector<DBIdentifier>& partkeys,
                std::vector<DBDecimal>& partkey_values);

  void PrintQ12(std::string& SM1, std::string& SM2,
                std::array<DBDecimal, 2> high_line_count,
                std::array<DBDecimal, 2> low_line_count);

 private:
  bool ParseLineItemTable(std::string f, LineItemTable& tbl);
  bool ParseOrdersTable(std::string f, OrdersTable& tbl);
  bool ParsePartsTable(std::string f, PartsTable& tbl);
  bool ParseSupplierTable(std::string f, SupplierTable& tbl);
  bool ParsePartSupplierTable(std::string f, PartSupplierTable& tbl);
  bool ParseNationTable(std::string f, NationTable& tbl);
};

#endif /* __DBDATA_HPP__ */
