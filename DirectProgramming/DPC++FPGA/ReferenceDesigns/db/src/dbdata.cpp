#include <assert.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>

#include "dbdata.hpp"
#include "db_utils/Date.hpp"

// choose a file separator based on the platform (Windows or Linux)
#if defined(WIN32) || defined(_WIN32) || defined(_MSC_VER)
constexpr char kSeparator = '\\';
#else 
constexpr char kSeparator = '/';
#endif

//
// split a row in the database file on its separator ('|')
//
std::vector<std::string> SplitRowStr(const std::string& row) {
  std::stringstream ss(row);
  std::string segment;
  std::vector<std::string> columns;

  while (std::getline(ss, segment, '|')) {
    columns.push_back(segment);
  }

  return columns;
}

//
// appends 'n' characters of a string to a vector
//
void AppendStringToCharVec(std::vector<char>& v, const std::string& str,
                           const unsigned int n) {
  // append the 'n' characters, padding with null terminators ('\0')
  const size_t str_n = str.size();
  for (size_t i = 0; i < n; i++) {
    v.push_back(i < str_n ? str[i] : '\0');
  }
}

//
// convert a money string (i.e. '1209.12' dollars) to
// the internal 'cents' representation (i.e. '120912' cents)
//
DBDecimal MoneyFloatToCents(const std::string& money_float_str) {
  // parse money string
  std::istringstream ss(money_float_str);
  std::string dollars_str, cents_str;

  std::getline(ss, dollars_str, '.');
  std::getline(ss, cents_str, '.');
  
  DBDecimal dollars = atoi(dollars_str.c_str());
  DBDecimal cents = atoi(cents_str.c_str());

  // convert to fixed point format (i.e. in cents)
  return dollars * 100 + cents;
}

//
// convert a SHIPMODE string to the internal representation (integer)
//
int ShipmodeStrToInt(std::string& shipmode_str) {
  if (shipmode_str == "REG AIR") {
    return 0;
  } else if (shipmode_str == "AIR") {
    return 1;
  } else if (shipmode_str == "RAIL") {
    return 2;
  } else if (shipmode_str == "SHIP") {
    return 3;
  } else if (shipmode_str == "TRUCK") {
    return 4;
  } else if (shipmode_str == "MAIL") {
    return 5;
  } else if (shipmode_str == "FOB") {
    return 6;
  } else {
    std::cerr << "WARNING: Found unknown SHIPMODE '" << shipmode_str
              << " defaulting to REG AIR\n";
    return 0;
  }
}

//
// check if two decimal values are within an epsilon of each other
//
bool AlmostEqual(double x, double y, double epsilon = 0.01f) {
  return std::fabs(x - y) < epsilon;
}

//
// trim the leading whitespace of a std::string
//
void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { 
    return !std::isspace(ch);
  }));
}

//
// trim the trailing whitespace of a std::string
//
void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { 
    return !std::isspace(ch);
  }).base(), s.end());
}

//
// trim leading and trailing whitespace of a string
//
void trim(std::string& s) {
  ltrim(s);
  rtrim(s);
}

//
// the main parsing function
// parses '*.tbl' files location in directory 'db_root_dir'
//
bool Database::Parse(std::string db_root_dir) {
  std::cout << "Parsing database files in: " << db_root_dir << std::endl;

  bool success = true;

  // parse each table
  success &= ParseLineItemTable(db_root_dir + kSeparator + "lineitem.tbl", l);
  success &= ParseOrdersTable(db_root_dir + kSeparator +"orders.tbl", o);
  success &= ParsePartsTable(db_root_dir + kSeparator + "part.tbl", p);
  success &= ParseSupplierTable(db_root_dir + kSeparator + "supplier.tbl", s);
  success &= ParsePartSupplierTable(db_root_dir + kSeparator + "partsupp.tbl", ps);
  success &= ParseNationTable(db_root_dir + kSeparator + "nation.tbl", n);

  return success;
}

//
// parse the LINEITEM table
//
bool Database::ParseLineItemTable(std::string f, LineItemTable& tbl) {
  std::cout << "Parsing LINEITEM table from: " << f << "\n";

  // populate data row by row (as presented in the file)
  std::ifstream ifs(f);
  std::string line;
  tbl.rows = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to parse LINEITEM table\n";
    return false;
  }

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 16);

    tbl.orderkey.push_back(std::stoll(column_data[0]));
    tbl.partkey.push_back(std::stoll(column_data[1]));
    tbl.suppkey.push_back(std::stoll(column_data[2]));
    tbl.linenumber.push_back(std::stoll(column_data[3]));
    tbl.quantity.push_back(std::stoll(column_data[4]));

    tbl.extendedprice.push_back(MoneyFloatToCents(column_data[5]));
    tbl.discount.push_back(MoneyFloatToCents(column_data[6]));
    tbl.tax.push_back(MoneyFloatToCents(column_data[7]));

    tbl.returnflag.push_back(column_data[8].at(0));
    tbl.linestatus.push_back(column_data[9].at(0));

    tbl.shipdate.push_back(Date(column_data[10]).ToCompact());
    tbl.commitdate.push_back(Date(column_data[11]).ToCompact());
    tbl.receiptdate.push_back(Date(column_data[12]).ToCompact());

    AppendStringToCharVec(tbl.shipinstruct, column_data[13], 25);
    tbl.shipmode.push_back(ShipmodeStrToInt(column_data[14]));
    AppendStringToCharVec(tbl.comment, column_data[15], 44);

    tbl.rows++;
  }

  std::cout << "Finished parsing LINEITEM table with " << tbl.rows << " rows\n";

  return true;
}

//
// parse the ORDERS table
//
bool Database::ParseOrdersTable(std::string f, OrdersTable& tbl) {
  std::cout << "Parsing ORDERS table from: " << f << "\n";

  // populate data row by row (as presented in the file)
  std::ifstream ifs(f);
  std::string line;
  tbl.rows = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to parse ORDERS table\n";
    return false;
  }

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 9);

    // store date for this row
    tbl.orderkey.push_back(std::stoll(column_data[0]));
    tbl.custkey.push_back(std::stoll(column_data[1]));
    tbl.orderstatus.push_back(column_data[2][0]);
    tbl.totalprice.push_back(MoneyFloatToCents(column_data[3]));
    tbl.orderdate.push_back(Date(column_data[4]).ToCompact());
    tbl.orderpriority.push_back((int)(column_data[5][0] - '0'));
    AppendStringToCharVec(tbl.clerk, column_data[6], 15);
    tbl.shippriority.push_back(std::stoi(column_data[7]));
    AppendStringToCharVec(tbl.comment, column_data[8], 80);

    tbl.rows++;
  }

  std::cout << "Finished parsing ORDERS table with " << tbl.rows << " rows\n";

  return true;
}

//
// parse the PARTS table
//
bool Database::ParsePartsTable(std::string f, PartsTable& tbl) {
  std::cout << "Parsing PARTS table from: " << f << "\n";

  // populate data row by row (as presented in the file)
  std::ifstream ifs(f);
  std::string line;
  tbl.rows = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to parse PARTS table\n";
    return false;
  }

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 9);

    tbl.partkey.push_back(std::stoll(column_data[0]));

    // formatting part name: no spaces, all uppercase
    transform(column_data[1].begin(), column_data[1].end(),
              column_data[1].begin(), ::toupper);
    AppendStringToCharVec(tbl.name, column_data[1], 55);

    AppendStringToCharVec(tbl.mfgr, column_data[2], 25);
    AppendStringToCharVec(tbl.brand, column_data[3], 10);
    AppendStringToCharVec(tbl.type, column_data[4], 25);
    tbl.size.push_back(std::stoi(column_data[5]));
    AppendStringToCharVec(tbl.container, column_data[6], 10);
    tbl.retailprice.push_back(MoneyFloatToCents(column_data[7]));
    AppendStringToCharVec(tbl.comment, column_data[8], 23);

    tbl.rows++;
  }

  std::cout << "Finished parsing PARTS table with " << tbl.rows << " rows\n";

  return true;
}

//
// parse the SUPPLIER table
//
bool Database::ParseSupplierTable(std::string f, SupplierTable& tbl) {
  std::cout << "Parsing SUPPLIER table from: " << f << "\n";

  // populate data row by row (as presented in the file)
  std::ifstream ifs(f);
  std::string line;
  tbl.rows = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to parse SUPPLIER table\n";
    return false;
  }

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 7);

    tbl.suppkey.push_back(std::stoll(column_data[0]));
    AppendStringToCharVec(tbl.name, column_data[1], 25);
    AppendStringToCharVec(tbl.address, column_data[2], 40);
    tbl.nationkey.push_back((unsigned char)(std::stoul(column_data[3])));
    AppendStringToCharVec(tbl.phone, column_data[4], 15);
    tbl.acctbal.push_back(MoneyFloatToCents(column_data[5]));
    AppendStringToCharVec(tbl.comment, column_data[6], 101);

    tbl.rows++;
  }

  std::cout << "Finished parsing SUPPLIER table with " << tbl.rows << " rows\n";

  return true;
}

//
// parse the PARTSUPPLIER table
//
bool Database::ParsePartSupplierTable(std::string f, PartSupplierTable& tbl) {
  std::cout << "Parsing PARTSUPPLIER table from: " << f << "\n";

  // populate data row by row (as presented in the file)
  std::ifstream ifs(f);
  std::string line;
  tbl.rows = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to parse PARTSUPPLIER table\n";
    return false;
  }

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 5);

    tbl.partkey.push_back(std::stoll(column_data[0]));
    tbl.suppkey.push_back(std::stoll(column_data[1]));
    tbl.availqty.push_back(std::stoi(column_data[2]));
    tbl.supplycost.push_back(MoneyFloatToCents(column_data[3]));
    AppendStringToCharVec(tbl.comment, column_data[4], 199);

    tbl.rows++;
  }

  std::cout << "Finished parsing PARTSUPPLIER table with " << tbl.rows
            << " rows\n";

  return true;
}

//
// parse the NATION table
//
bool Database::ParseNationTable(std::string f, NationTable& tbl) {
  std::cout << "Parsing NATION table from: " << f << "\n";

  // populate data row by row (as presented in the file)
  std::ifstream ifs(f);
  std::string line;
  tbl.rows = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to parse NATION table\n";
    return false;
  }

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 4);

    DBIdentifier nationkey = std::stoll(column_data[0]);
    std::string nationname = column_data[1];

    tbl.nationkey.push_back(nationkey);

    // convention: all upper case
    trim(nationname);
    transform(nationname.begin(), nationname.end(), nationname.begin(),
              ::toupper);
    AppendStringToCharVec(tbl.name, nationname, 25);

    tbl.regionkey.push_back(std::stoll(column_data[2]));
    AppendStringToCharVec(tbl.comment, column_data[3], 152);

    // add entry into map
    tbl.name_key_map[nationname] = nationkey;
    tbl.key_name_map[nationkey] = nationname;

    tbl.rows++;
  }

  std::cout << "Finished parsing NATION table with " << tbl.rows << " rows\n";

  return true;
}

//
// Checks the size of each parsed table against the expected size based
// on the selected scale factor
//
bool Database::ValidateSF() {
  bool ret = true;

  if (l.rows != kLineItemTableSize) {
    std::cerr << "LineItem table size has " << l.rows << " rows"
              << " when it should have " << kLineItemTableSize << "\n";
    ret = false;
  }

  if (o.rows != kOrdersTableSize) {
    std::cerr << "Orders table size has " << o.rows << " rows"
              << " when it should have " << kOrdersTableSize << "\n";
    ret = false;
  }

  if (p.rows != kPartTableSize) {
    std::cerr << "Parts table size has " << p.rows << " rows"
              << " when it should have " << kPartTableSize << "\n";
    ret = false;
  }

  if (s.rows != kSupplierTableSize) {
    std::cerr << "Supplier table size has " << s.rows << " rows"
              << " when it should have " << kSupplierTableSize << "\n";
    ret = false;
  }

  if (ps.rows != kPartSupplierTableSize) {
    std::cerr << "PartSupplier table size has " << ps.rows << " rows"
              << " when it should have " << kPartSupplierTableSize << "\n";
    ret = false;
  }

  if (n.rows != kNationTableSize) {
    std::cerr << "Nation table size has " << n.rows << " rows"
              << " when it should have " << kNationTableSize << "\n";
    ret = false;
  }

  return ret;
}

//
// validate the results of Query 1
//
bool Database::ValidateQ1(std::string db_root_dir,
                          std::array<DBDecimal, 3 * 2>& sum_qty,
                          std::array<DBDecimal, 3 * 2>& sum_base_price,
                          std::array<DBDecimal, 3 * 2>& sum_disc_price,
                          std::array<DBDecimal, 3 * 2>& sum_charge,
                          std::array<DBDecimal, 3 * 2>& avg_qty,
                          std::array<DBDecimal, 3 * 2>& avg_price,
                          std::array<DBDecimal, 3 * 2>& avg_discount,
                          std::array<DBDecimal, 3 * 2>& count) {
  std::cout << "Validating query 1 test results\n";

  // populate date row by row (as presented in the file)
  std::string path(db_root_dir + kSeparator + "answers" + kSeparator + "q1.out");
  std::ifstream ifs(path);

  bool valid = true;

  if (!ifs.is_open()) {
    std::cout << "Failed to open " << path << "\n";
    return false;
  }

  // this will hold the line read from the input file
  std::string line;

  // do nothing with the first line, it is a header line
  std::getline(ifs, line);

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 10);

    char rf = column_data[0][0];
    char ls = column_data[1][0];
    double sum_qty_gold = std::stod(column_data[2]);
    double sum_base_price_gold = std::stod(column_data[3]);
    double sum_disc_price_gold = std::stod(column_data[4]);
    double sum_charge_gold = std::stod(column_data[5]);
    double avg_qty_gold = std::stod(column_data[6]);
    double avg_price_gold = std::stod(column_data[7]);
    double avg_disc_gold = std::stod(column_data[8]);
    unsigned int count_order_gold = std::stoll(column_data[9]);

    unsigned int rf_idx;
    if (rf == 'R') {
      rf_idx = 0;
    } else if (rf == 'A') {
      rf_idx = 1;
    } else {  // == 'N'
      rf_idx = 2;
    }
    unsigned int ls_idx;
    if (ls == 'O') {
      ls_idx = 0;
    } else {  // == 'F'
      ls_idx = 1;
    }
    unsigned int idx = ls_idx * 3 + rf_idx;
    assert(idx < 6);

    double sum_qty_res = (double)(sum_qty[idx]);
    double sum_base_price_res = (double)(sum_base_price[idx]) / (100.00);
    double sum_disc_price_res =
        (double)(sum_disc_price[idx]) / (100.00 * 100.00);
    double sum_charge_res =
        (double)(sum_charge[idx]) / (100.00 * 100.00 * 100.00);
    double avg_qty_res = (double)(avg_qty[idx]);
    double avg_price_res = (double)(avg_price[idx]) / (100.00);
    double avg_disc_res = (double)(avg_discount[idx]) / (100.00);
    unsigned int count_order_res = count[idx];

    if (!AlmostEqual(sum_qty_gold, sum_qty_res, 0.01f)) {
      std::cerr << "ERROR: sum_qty for returnflag=" << rf
                << " and linestatus=" << ls << " (Expected=" << sum_qty_gold
                << ", Result=" << sum_qty_res << ")\n";
      valid = false;
    }
    if (!AlmostEqual(sum_base_price_gold, sum_base_price_res, 0.01f)) {
      std::cerr << "ERROR: sum_base_price for returnflag=" << rf
                << " and linestatus=" << ls
                << " (Expected=" << sum_base_price_gold
                << ", Result=" << sum_base_price_res << ")\n";
      valid = false;
    }
    if (!AlmostEqual(sum_disc_price_gold, sum_disc_price_res, 0.01f)) {
      std::cerr << "ERROR: sum_disc_price for returnflag=" << rf
                << " and linestatus=" << ls
                << " (Expected=" << sum_disc_price_gold
                << ", Result=" << sum_disc_price_res << ")\n";
      valid = false;
    }
    if (!AlmostEqual(sum_charge_gold, sum_charge_res, 0.01f)) {
      std::cerr << "ERROR: sum_charge for returnflag=" << rf
                << " and linestatus=" << ls << " (Expected=" << sum_charge_gold
                << ", Result=" << sum_charge_res << ")\n";
      valid = false;
    }
    if (!AlmostEqual(avg_qty_gold, avg_qty_res, 1.0f)) {
      std::cerr << "ERROR: avg_qty for returnflag=" << rf
                << " and linestatus=" << ls << " (Expected=" << avg_qty_gold
                << ", Result=" << avg_qty_res << ")\n";
      valid = false;
    }
    if (!AlmostEqual(avg_price_gold, avg_price_res, 1.0f)) {
      std::cerr << "ERROR: avg_price for returnflag=" << rf
                << " and linestatus=" << ls << " (Expected=" << avg_price_gold
                << ", Result=" << avg_price_res << ")\n";
      valid = false;
    }
    if (!AlmostEqual(avg_disc_gold, avg_disc_res, 1.0f)) {
      std::cerr << "ERROR: avg_disc for returnflag=" << rf
                << " and linestatus=" << ls << " (Expected=" << avg_disc_gold
                << ", Result=" << avg_disc_res << ")\n";
      valid = false;
    }
    if (count_order_gold != count_order_res) {
      std::cerr << "ERROR: count for returnflag=" << rf
                << " and linestatus=" << ls << " (Expected=" << count_order_gold
                << ", Result=" << count_order_res << ")\n";
      valid = false;
    }
  }

  return valid;
}

//
// validate the results of Query 9
//
bool Database::ValidateQ9(std::string db_root_dir,
                          std::array<DBDecimal, 25 * 2020>& sum_profit) {
  std::cout << "Validating query 9 test results\n";

  // populate date row by row (as presented in the file)
  std::string path(db_root_dir + kSeparator + "answers" + kSeparator + "q9.out");
  std::ifstream ifs(path);
  std::string line;

  bool valid = true;

  if (!ifs.is_open()) {
    std::cout << "Failed to open " << path << "\n";
    return false;
  }

  // do nothing with the first line, it is a header line
  std::getline(ifs, line);

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 3);

    std::string nationname_gold = column_data[0];
    trim(nationname_gold);
    transform(nationname_gold.begin(), nationname_gold.end(),
              nationname_gold.begin(), ::toupper);

    assert(n.name_key_map.find(nationname_gold) != n.name_key_map.end());

    unsigned char nationkey_gold = n.name_key_map[nationname_gold];

    unsigned int year_gold = std::stoi(column_data[1]);
    double sum_profit_gold = std::stod(column_data[2]);

    double sum_profit_res =
        (double)(sum_profit[year_gold * 25 + nationkey_gold]) / (100.0 * 100.0);

    if (!AlmostEqual(sum_profit_gold, sum_profit_res, 0.01f)) {
      std::cerr << "ERROR: sum_profit for " << nationname_gold << " in "
                << year_gold << " did not match (Expected=" << sum_profit_gold
                << ", Result=" << sum_profit_res << ")\n";
      valid = false;
    }
  }

  return valid;
}

//
// validate the results of Query 11
//
bool Database::ValidateQ11(std::string db_root_dir,
                           std::vector<DBIdentifier>& partkeys,
                           std::vector<DBDecimal>& partkey_values) {
  std::cout << "Validating query 11 test results\n";

  // populate date row by row (as presented in the file)
  std::string path(db_root_dir + kSeparator + "answers" + kSeparator + "q11.out");
  std::ifstream ifs(path);
  std::string line;
  size_t i = 0;

  if (!ifs.is_open()) {
    std::cout << "Failed to open " << path << "\n";
    return false;
  }

  bool valid = true;

  // do nothing with the first line, it is a header line
  std::getline(ifs, line);

  while (std::getline(ifs, line)) {
    // split row into column strings by separator ('|')
    std::vector<std::string> column_data = SplitRowStr(line);
    assert(column_data.size() == 2);
    assert(i < kPartTableSize);

    DBIdentifier partkey_gold = std::stoll(column_data[0]);
    double value_gold = std::stod(column_data[1]);

    DBIdentifier partkey_res = partkeys[i];
    double value_res = (double)(partkey_values[i]) / (100.00);

    if (partkey_gold != partkey_res) {
      std::cerr << "ERROR: partkeys at index " << i << " do not match "
                << "(Expected=" << partkey_gold << ", Result=" << partkey_res
                << ")\n";
      valid = false;
    }

    if (!AlmostEqual(value_gold, value_res, 0.01f)) {
      std::cerr << "ERROR: value at index " << i << " do not match "
                << "(Expected=" << value_gold << ", Result=" << value_res
                << ")\n";
      valid = false;
    }

    i++;
  }

  assert(i > 0);

  return valid;
}

//
// validate the results of Query 12
//
bool Database::ValidateQ12(std::string db_root_dir,
                           std::array<DBDecimal, 2> high_line_count,
                           std::array<DBDecimal, 2> low_line_count) {
  std::cout << "Validating query 12 test results" << std::endl;

  // populate date row by row (as presented in the file)
  std::string path(db_root_dir + kSeparator + "answers" + kSeparator + "q12.out");
  std::ifstream ifs(path);
  std::vector<std::string> column_data;
  std::string line;

  bool valid = true;

  if (!ifs.is_open()) {
    std::cout << "Failed to open " << path << "\n";
    return false;
  }

  // do nothing with the first line, it is a header line
  std::getline(ifs, line);

  // MAIL shipmode
  std::getline(ifs, line);
  column_data = SplitRowStr(line);
  assert(column_data.size() == 3);

  DBDecimal mail_high_count_gold = std::stoll(column_data[1]);
  DBDecimal mail_low_count_gold = std::stoll(column_data[2]);

  if (mail_high_count_gold != high_line_count[0]) {
    std::cerr << "ERROR: MAIL high_count (Expected=" << mail_high_count_gold
              << ", Result=" << high_line_count[0] << ")\n";
    valid = false;
  }
  if (mail_low_count_gold != low_line_count[0]) {
    std::cerr << "ERROR: MAIL low_count (Expected=" << mail_low_count_gold
              << ", Result=" << low_line_count[0] << ")\n";
    valid = false;
  }

  // SHIP shipmode
  std::getline(ifs, line);
  column_data = SplitRowStr(line);
  assert(column_data.size() == 3);

  DBDecimal ship_high_count_gold = std::stoll(column_data[1]);
  DBDecimal ship_low_count_gold = std::stoll(column_data[2]);

  if (ship_high_count_gold != high_line_count[1]) {
    std::cerr << "ERROR: SHIP high_count (Expected=" << ship_high_count_gold
              << ", Result=" << high_line_count[1] << ")\n";
    valid = false;
  }
  if (ship_low_count_gold != low_line_count[1]) {
    std::cerr << "ERROR: SHIP low_count (Expected=" << ship_low_count_gold
              << ", Result=" << low_line_count[1] << ")\n";
    valid = false;
  }

  return valid;
}

//
// print the results of Query 1
//
void Database::PrintQ1(std::array<DBDecimal, 3 * 2>& sum_qty,
                       std::array<DBDecimal, 3 * 2>& sum_base_price,
                       std::array<DBDecimal, 3 * 2>& sum_disc_price,
                       std::array<DBDecimal, 3 * 2>& sum_charge,
                       std::array<DBDecimal, 3 * 2>& avg_qty,
                       std::array<DBDecimal, 3 * 2>& avg_price,
                       std::array<DBDecimal, 3 * 2>& avg_discount,
                       std::array<DBDecimal, 3 * 2>& count) {
  // line status (ls) and return flag (rf)
  const std::array<char, 2> ls = {'O', 'F'};
  const std::array<char, 3> rf = {'R', 'A', 'N'};

  // print the header
  std::cout << "l|l|sum_qty|sum_base_price|sum_disc_price|"
               "sum_charge|avg_qty|avg_price|avg_disc|count_order\n";

  // print the results
  std::cout << std::fixed << std::setprecision(2);
  for (int ls_idx = 0; ls_idx < 2; ls_idx++) {
    for (int rf_idx = 0; rf_idx < 3; rf_idx++) {
      int i = ls_idx * 3 + rf_idx;
      std::cout << rf[rf_idx] << "|" << ls[ls_idx] << "|" << sum_qty[i] << "|"
                << (double)(sum_base_price[i]) / 100.0 << "|"
                << (double)(sum_disc_price[i]) / (100.00 * 100.00) << "|"
                << (double)(sum_charge[i]) / (100.00 * 100.00 * 100.00) << "|"
                << avg_qty[i] << "|" << (double)(avg_price[i]) / 100.0 << "|"
                << (double)(avg_discount[i]) / 100.0 << "|" << count[i] << "\n";
    }
  }
}

//
// print the results of Query 9
//
void Database::PrintQ9(std::array<DBDecimal, 25 * 2020>& sum_profit) {
  // row of Q9 output for local sorting
  struct Row {
    Row(std::string& nation, int year, DBDecimal sum_profit)
        : nation(nation), year(year), sum_profit(sum_profit) {}
    std::string nation;
    int year;
    DBDecimal sum_profit;

    void print() {
      std::cout << nation << "|" << year << "|"
                << (double)(sum_profit) / (100.0 * 100.0) << "\n";
    }
  };

  // create the rows
  std::vector<Row> outrows;
  for (unsigned char nat = 0; nat < kNationTableSize; nat++) {
    std::string nation_name = n.key_name_map[nat];
    for (int y = 1992; y <= 1998; y++) {
      outrows.push_back(Row(nation_name, y, sum_profit[y * 25 + nat]));
    }
  }

  // sort rows by year
  std::sort(outrows.begin(), outrows.end(),
    [](const Row& a, const Row& b) -> bool {
      return a.year > b.year;
  });

  // sort rows by nation
  // stable_sort() preserves the order of the previous sort
  std::stable_sort(outrows.begin(), outrows.end(),
    [](const Row& a, const Row& b) -> bool {
      return a.nation < b.nation;
  });

  // print the header
  std::cout << "nation|o_year|sum_profit\n";

  // print the results
  std::cout << std::fixed << std::setprecision(2);
  for (int i = 0; i < outrows.size(); i++) {
    outrows[i].print();
  }
}

//
// print the results of Query 11
//
void Database::PrintQ11(std::vector<DBIdentifier>& partkeys,
                        std::vector<DBDecimal>& partkey_values) {
  // print the header
  std::cout << "ps_partkey|value\n";

  // print the results
  std::cout << std::fixed << std::setprecision(2);
  for (int i = 0; i < partkeys.size(); i++) {
    std::cout << partkeys[i] << "|" << (double)(partkey_values[i]) / (100.00)
              << "\n";
  }
}

//
// print the results of Query 12
//
void Database::PrintQ12(std::string& SM1, std::string& SM2,
                        std::array<DBDecimal, 2> high_line_count,
                        std::array<DBDecimal, 2> low_line_count) {
  // print the header
  std::cout << "l_shipmode|high_line_count|low_line_count\n";

  // print the results
  std::cout << std::fixed;
  std::cout << SM1 << "|" << high_line_count[0] << "|" << low_line_count[0]
            << "\n";
  std::cout << SM2 << "|" << high_line_count[1] << "|" << low_line_count[1]
            << "\n";
}
