//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "db_utils/Date.hpp"
#include "db_utils/LikeRegex.hpp"
#include "dbdata.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// include files depending on the query selected
#if (QUERY == 1)
#include "query1/query1_kernel.hpp"
bool DoQuery1(queue& q, Database& dbinfo, std::string& db_root_dir,
              std::string& args, bool test, bool print, double& kernel_latency,
              double& total_latency);
#elif (QUERY == 9)
#include "query9/query9_kernel.hpp"
bool DoQuery9(queue& q, Database& dbinfo, std::string& db_root_dir,
              std::string& args, bool test, bool print, double& kernel_latency,
              double& total_latency);
#elif (QUERY == 11)
#include "query11/query11_kernel.hpp"
bool DoQuery11(queue& q, Database& dbinfo, std::string& db_root_dir,
               std::string& args, bool test, bool print, double& kernel_latency,
               double& total_latency);
#elif (QUERY == 12)
#include "query12/query12_kernel.hpp"
bool DoQuery12(queue& q, Database& dbinfo, std::string& db_root_dir,
               std::string& args, bool test, bool print, double& kernel_latency,
               double& total_latency);
#endif

//
// print help for the program
//
void Help() {
  std::cout << "USAGE:\n";
  std::cout << "\t./db --dbroot=<database root directory> [...]\n";
  std::cout << "\n";

  std::cout << "Optional Arguments:\n";
  std::cout << "\t--args=<comma separated arguments for query>"
               "   see the examples below\n";
  std::cout << "\t--test    enables testing of the query."
               " This overrides the arguments to the query (--args) "
               "and uses default input from TPCH documents\n";
  std::cout << "\t--print   print the query results to stdout\n";
  std::cout << "\t--runs    how many iterations of the query to run\n";
  std::cout << "\t--help    print this help message\n";
  std::cout << "\n";

  std::cout << "Examples:\n";
  std::cout << "./db --dbroot=/path/to/database/files "
            << "[--test] [--args=<DATE,DELTA>]\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files --test\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files "
            << "--args=1998-12-01,90\n";
  std::cout << "\n";

  std::cout << "./db --dbroot=/path/to/database/files "
            << "[--test] [--args=<COLOUR>]\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files --test\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files --args=GREEN\n";
  std::cout << "\n";

  std::cout << "./db --dbroot=/path/to/database/files "
            << "[--test] [--args=<NATION>]\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files --test\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files "
            << "--args=GERMANY\n";
  std::cout << "\n";

  std::cout << "./db --dbroot=/path/to/database/files [--test] "
            << "[--args=<SHIPMODE1,SHIPMODE2,DATE>]\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files --test\n";
  std::cout << "\t ./db --dbroot=/path/to/database/files "
            << "--args=MAIL,SHIP,1994-01-10\n";
  std::cout << "\n";
}

//
// determine if a string starts with a prefix
//
bool StrStartsWith(std::string& str, std::string prefix) {
  return str.find(prefix) == 0;
}

//
// main
//
int main(int argc, char* argv[]) {
  // argument defaults
  Database dbinfo;
  std::string db_root_dir = ".";
  std::string args = "";
  unsigned int query = QUERY;
  bool test_query = false;
#ifndef FPGA_EMULATOR
  unsigned int runs = 5;
#else
  unsigned int runs = 1;
#endif
  bool print_result = false;
  bool need_help = false;

  // parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      need_help = true;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (StrStartsWith(arg, "--dbroot=")) {
        db_root_dir = str_after_equals;
      } else if (StrStartsWith(arg, "--query=")) {
        query = atoi(str_after_equals.c_str());
      } else if (StrStartsWith(arg, "--args=")) {
        args = str_after_equals;
      } else if (StrStartsWith(arg, "--test")) {
        test_query = true;
      } else if (StrStartsWith(arg, "--print")) {
        print_result = true;
      } else if (StrStartsWith(arg, "--runs")) {
#ifndef FPGA_EMULATOR
        // for hardware, ensure at least two iterations to ensure we can run
        // a 'warmup' iteration
        runs = std::max(2, atoi(str_after_equals.c_str()) + 1);
#else
        // for emulation, allow a single iteration and don't add a 'warmup' run
        runs = std::max(1, atoi(str_after_equals.c_str()));
#endif
      } else {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  // print help if needed or asked
  if (need_help) {
    Help();
    return 0;
  }

  // make sure the query is supported
  if (!(query == 1 || query == 9 || query == 11 || query == 12)) {
    std::cerr << "ERROR: unsupported query (" << query << "). "
              << "Only queries 1, 9, 11 and 12 are supported\n";
    return 1;
  }

  if (query != QUERY) {
    std::cerr << "ERROR: project not currently configured for query " << query
              << "\n";
    std::cerr << "\trerun CMake using the command: 'cmake .. -DQUERY=" << query
              << "'\n";
    return 1;
  }

  try {
    // queue properties to enable profiling
    auto props = property_list{property::queue::enable_profiling()};

    // the device selector
#ifdef FPGA_EMULATOR
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // create the device queue
    queue q(selector, dpc_common::exception_handler, props);

    // parse the database files located in the 'db_root_dir' directory
    bool success = dbinfo.Parse(db_root_dir);
    if (!success) {
      std::cerr << "ERROR: couldn't read the DB files\n";
      return 1;
    }

    std::cout << "Database SF = " << kSF << "\n";

    // make sure the parsed database files match the set scale factor
    if (!dbinfo.ValidateSF()) {
      std::cerr << "ERROR: could not validate the "
                << "scale factor of the parsed database files\n";
      return 1;
    }

    // track timing information for each run
    std::vector<double> total_latency(runs);
    std::vector<double> kernel_latency(runs);

    // run 'runs' iterations of the query
    for (unsigned int run = 0; run < runs && success; run++) {
      // run the selected query
      if (query == 1) {
#if (QUERY == 1)
        success = DoQuery1(q, dbinfo, db_root_dir, args,
                           test_query, print_result,
                           kernel_latency[run], total_latency[run]);
#endif
      } else if (query == 9) {
        // query9
#if (QUERY == 9)
        success = DoQuery9(q, dbinfo, db_root_dir, args,
                           test_query, print_result,
                           kernel_latency[run], total_latency[run]);
#endif
      } else if (query == 11) {
        // query11
#if (QUERY == 11)
        success = DoQuery11(q, dbinfo, db_root_dir, args,
                            test_query, print_result,
                            kernel_latency[run], total_latency[run]);
#endif
      } else if (query == 12) {
        // query12
#if (QUERY == 12)
        success = DoQuery12(q, dbinfo, db_root_dir, args,
                            test_query, print_result,
                            kernel_latency[run], total_latency[run]);
#endif
      } else {
        std::cerr << "ERROR: unsupported query (" << query << ")\n";
        return 1;
      }
    }

    if (success) {
      // don't analyze the runtime in emulation
#ifndef FPGA_EMULATOR
      // compute the average total latency across all iterations,
      // excluding the first 'warmup' iteration
      double total_latency_avg =
          std::accumulate(total_latency.begin() + 1, total_latency.end(), 0.0) /
          (double)(runs - 1);

      // print the performance results
      std::cout << "Processing time: " << total_latency_avg << " ms\n";
#endif

      std::cout << "PASSED\n";
    } else {
      std::cout << "FAILED\n";
      return 1;
    }

  } catch (exception const& e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  return 0;
}

#if (QUERY == 1)
bool DoQuery1(queue& q, Database& dbinfo, std::string& db_root_dir,
              std::string& args, bool test, bool print, double& kernel_latency,
              double& total_latency) {
  // NOTE: this is fixed based on the TPCH docs
  Date date = Date("1998-12-01");
  unsigned int DELTA = 90;

  // parse the query arguments
  if (!test && !args.empty()) {
    std::stringstream ss(args);
    std::string tmp;
    std::getline(ss, tmp, ',');
    DELTA = atoi(tmp.c_str());
  } else {
    if (!args.empty()) {
      std::cout << "Testing query 1, therefore ignoring the '--args' flag\n";
    }
  }

  // check query arguments
  if (!(DELTA <= 120 && DELTA >= 60)) {
    std::cerr << "ERROR: DELTA must be in the range [60,120]\n";
    return false;
  }

  // compute query interval
  Date low_date = date.PreviousDate(DELTA);
  unsigned int low_date_compact = low_date.ToCompact();

  std::cout << "Running Q1 within " << DELTA << " days of " << date.year << "-"
            << date.month << "-" << date.day << "\n";

  // the query output data
  std::array<DBDecimal, kQuery1OutSize> sum_qty = {0}, sum_base_price = {0},
                                       sum_disc_price = {0}, sum_charge = {0},
                                       avg_qty = {0}, avg_price = {0},
                                       avg_discount = {0}, count = {0};

  // perform the query
  bool success =
      SubmitQuery1(q, dbinfo, low_date_compact, sum_qty, sum_base_price,
                   sum_disc_price, sum_charge, avg_qty, avg_price, avg_discount,
                   count, kernel_latency, total_latency);

  if (success) {
    // validate the results of the query, if requested
    if (test) {
      success = dbinfo.ValidateQ1(db_root_dir, sum_qty, sum_base_price,
                                  sum_disc_price, sum_charge, avg_qty,
                                  avg_price, avg_discount, count);
    }

    // print the results of the query, if requested
    if (print) {
      dbinfo.PrintQ1(sum_qty, sum_base_price, sum_disc_price, sum_charge,
                     avg_qty, avg_price, avg_discount, count);
    }
  }

  return success;
}
#endif

#if (QUERY == 9)
bool DoQuery9(queue& q, Database& dbinfo, std::string& db_root_dir,
              std::string& args, bool test, bool print, double& kernel_latency,
              double& total_latency) {
  // the default colour regex based on the TPCH documents
  std::string colour = "GREEN";

  // parse the query arguments
  if (!test && !args.empty()) {
    std::stringstream ss(args);
    std::getline(ss, colour, ',');
  } else {
    if (!args.empty()) {
      std::cout << "Testing query 9, therefore ignoring the '--args' flag\n";
    }
  }

  // convert the colour regex to uppercase characters (convention)
  transform(colour.begin(), colour.end(), colour.begin(), ::toupper);

  std::cout << "Running Q9 with colour regex: " << colour << "\n";

  // the output of the query
  std::array<DBDecimal, 25 * 2020> sum_profit;

  // perform the query
  bool success = SubmitQuery9(q, dbinfo, colour, sum_profit, kernel_latency,
                              total_latency);

  if (success) {
    // validate the results of the query, if requested
    if (test) {
      success = dbinfo.ValidateQ9(db_root_dir, sum_profit);
    }

    // print the results of the query, if requested
    if (print) {
      dbinfo.PrintQ9(sum_profit);
    }
  }

  return success;
}
#endif

#if (QUERY == 11)
bool DoQuery11(queue& q, Database& dbinfo, std::string& db_root_dir,
               std::string& args, bool test, bool print, double& kernel_latency,
               double& total_latency) {
  // the default nation, based on the TPCH documents
  std::string nation = "GERMANY";

  // parse the query arguments
  if (!test && !args.empty()) {
    std::stringstream ss(args);
    std::getline(ss, nation, ',');
  } else {
    if (!args.empty()) {
      std::cout << "Testing query 11, therefore ignoring the '--args' flag\n";
    }
  }

  // convert the nation name to uppercase characters (convention)
  transform(nation.begin(), nation.end(), nation.begin(), ::toupper);

  std::cout << "Running Q11 for nation " << nation.c_str()
            << " (key=" << (int)(dbinfo.n.name_key_map[nation]) << ")\n";

  // the query output
  std::vector<DBIdentifier> partkeys(kPartTableSize);
  std::vector<DBDecimal> partkey_values(kPartTableSize);

  // perform the query
  bool success = SubmitQuery11(q, dbinfo, nation, partkeys, partkey_values,
                               kernel_latency, total_latency);

  if (success) {
    // validate the results of the query, if requested
    if (test) {
      success = dbinfo.ValidateQ11(db_root_dir, partkeys, partkey_values);
    }

    // print the results of the query, if requested
    if (print) {
      dbinfo.PrintQ11(partkeys, partkey_values);
    }
  }

  return success;
}
#endif

#if (QUERY == 12)
bool DoQuery12(queue& q, Database& dbinfo, std::string& db_root_dir,
               std::string& args, bool test, bool print, double& kernel_latency,
               double& total_latency) {
  // the default query date and shipmodes, based on the TPCH documents
  Date date = Date("1994-01-01");
  std::string shipmode1 = "MAIL", shipmode2 = "SHIP";

  // parse the query arguments
  if (!test && !args.empty()) {
    std::stringstream ss(args);
    std::string tmp;

    std::getline(ss, tmp, ',');
    date = Date(tmp);

    if (ss.good()) {
      std::getline(ss, shipmode1, ',');
    }

    if (ss.good()) {
      std::getline(ss, shipmode2, ',');
    }
  } else {
    if (!args.empty()) {
      std::cout << "Testing query 12, therefore ignoring the '--args' flag\n";
    }
  }

  // check the arguments, date must be January 1 of some year
  if (date.month != 1 || date.day != 1) {
    std::cerr << "ERROR: Date must be first of January "
              << "in the given year (e.g. 1994-01-01)\n";
    return false;
  }

  // compute the interval for the query
  Date low_date = date;
  Date high_date = Date(low_date.year + 1, low_date.month, low_date.day);

  std::cout << "Running Q12 between years " << low_date.year << " and "
            << high_date.year << " for SHIPMODES " << shipmode1 << " and "
            << shipmode2 << "\n";

  // the output of the query
  std::array<DBDecimal, 2> high_line_count, low_line_count;

  // perform the query
  bool success = SubmitQuery12(
      q, dbinfo, low_date.ToCompact(), high_date.ToCompact(),
      ShipmodeStrToInt(shipmode1), ShipmodeStrToInt(shipmode2), high_line_count,
      low_line_count, kernel_latency, total_latency);

  if (success) {
    // validate the results of the query, if requested
    if (test) {
      success =
          dbinfo.ValidateQ12(db_root_dir, high_line_count, low_line_count);
    }

    // print the results of the query, if requested
    if (print) {
      dbinfo.PrintQ12(shipmode1, shipmode2, high_line_count, low_line_count);
    }
  }

  return success;
}
#endif
