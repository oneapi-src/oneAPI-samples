#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "constexpr_math.hpp"  // included from ../../../include

#include "common/common.hpp"
#include "exception_handler.hpp"

using namespace sycl;

// ensure only one of GZIP and SNAPPY is defined
#if defined(GZIP) and defined(SNAPPY)
static_assert(false, "Only one of GZIP and SNAPPY can be defined!");
#endif

// if neither of GZIP and SNAPPY is defined, default to SNAPPY
#if not defined(GZIP) and not defined(SNAPPY)
#define SNAPPY
#endif

// the number of literals to process at once can be set from the command line
// use the macro -DLITERALS_PER_CYCLE=<literals_per_cycle>
// This is sent to the LZ77 decoder to read multiple elements at once from
// the history buffer.
#if not defined(LITERALS_PER_CYCLE)
// default LITERALS_PER_CYCLE for GZIP
#if defined(GZIP)
#define LITERALS_PER_CYCLE 4
#endif
// default LITERALS_PER_CYCLE for SNAPPY
#if defined(SNAPPY)
#define LITERALS_PER_CYCLE 8
#endif
#endif
constexpr unsigned kLiteralsPerCycle = LITERALS_PER_CYCLE;
static_assert(kLiteralsPerCycle > 0);
static_assert(fpga_tools::IsPow2(kLiteralsPerCycle));

// include files and aliases specific to GZIP and SNAPPY decompression
#if defined(GZIP)
#include "gzip/gzip_decompressor.hpp"
#else
#include "snappy/snappy_data_gen.hpp"
#include "snappy/snappy_decompressor.hpp"
#endif

// aliases and testing functions specific to GZIP and SNAPPY decompression
#if defined(GZIP)
using GzipDecompressorT = GzipDecompressor<kLiteralsPerCycle>;
bool RunGzipTest(sycl::queue& q, GzipDecompressorT decompressor,
                 const std::string test_dir);
std::string decompressor_name = "GZIP";
#else
using SnappyDecompressorT = SnappyDecompressor<kLiteralsPerCycle>;
bool RunSnappyTest(sycl::queue& q, SnappyDecompressorT decompressor,
                   const std::string test_dir);
std::string decompressor_name = "SNAPPY";
#endif

// Prints the usage for the executable command line args
void PrintUsage(std::string exe_name) {
  std::cerr << "USAGE: \n"
            << exe_name << " <input filename> <output filename> [runs]\n"
            << exe_name << " <test directory>" << std::endl;
}

int main(int argc, char* argv[]) {
  // reading and validating the command line arguments
  // if no arguments are given, we will run the default tests for uncompressed,
  // statically compressed, and dynamically compressed blocks
  // if arguments are given, we will assume the user wants to decompress a
  // specific file
#if defined(GZIP)
  std::string test_dir = "../data/gzip";
#else
  std::string test_dir = "../data/snappy";
#endif

  std::string in_filename;
  std::string out_filename;
  int runs;
  bool default_test_mode = false;

  if (argc == 1 || argc == 2) {
    default_test_mode = true;
  } else if (argc > 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (default_test_mode) {
    if (argc > 1) test_dir = argv[1];
  } else {
    // default the number of runs based on emulation, simulation, or hardware
#if defined(FPGA_EMULATOR)
    runs = 2;
#elif defined(FPGA_SIMULATOR)
    runs = 1;
#else
    runs = 9;
#endif

    in_filename = argv[1];
    out_filename = argv[2];
    if (argc > 3) runs = atoi(argv[3]);
    if (runs < 1) {
      std::cerr << "ERROR: 'runs' must be greater than 0\n";
      std::terminate();
    }
  }

  std::cout << "Using " << decompressor_name << " decompression\n";
  std::cout << std::endl;

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  // create the device queue
  queue q(selector, fpga_tools::exception_handler);

  device device = q.get_device();

  std::cout << "Running on device: "
            << device.get_info<info::device::name>().c_str() 
            << std::endl;

  // create the decompressor based on which decompression version we are using
#if defined(GZIP)
  GzipDecompressorT decompressor;
#else
  SnappyDecompressorT decompressor;
#endif

  // perform the test or single file decompression
  bool passed;
  if (default_test_mode) {
#if defined(GZIP)
    passed = RunGzipTest(q, decompressor, test_dir);
#else
    passed = RunSnappyTest(q, decompressor, test_dir);
#endif
  } else {
    // decompress a specific file specified at the command line
    passed = decompressor.DecompressFile(q, in_filename, out_filename, runs,
                                         true, true);
  }

  if (passed) {
    std::cout << "PASSED" << std::endl;
    return 0;
  } else {
    std::cout << "FAILED" << std::endl;
    return 1;
  }
}

//
// Pretty formatting for printing the result of a test
//
void PrintTestResults(std::string test_name, bool passed) {
  if (passed)
    std::cout << ">>>>> " << test_name << ": PASSED <<<<<\n";
  else
    std::cerr << ">>>>> " << test_name << ": FAILED <<<<<\n";
}

#if defined(GZIP)
bool RunGzipTest(sycl::queue& q, GzipDecompressorT decompressor,
                 const std::string test_dir) {


#ifdef FPGA_SIMULATOR
  // the name of the file for the simulator is fixed
  std::string small_filename = test_dir + "/small.gz";
  
  std::cout << ">>>>> Small File Test <<<<<" << std::endl;
  bool small_test_pass = decompressor.DecompressFile(
      q, small_filename, "", 1, false, false);
  PrintTestResults("Small File Test", small_test_pass);
  std::cout << std::endl;

  return small_test_pass;
#else
  // the name of the files for the default test are fixed
  std::string uncompressed_filename = test_dir + "/uncompressed.gz";
  std::string static_compress_filename = test_dir + "/static_compressed.gz";
  std::string dynamic_compress_filename = test_dir + "/dynamic_compressed.gz";
  std::string tp_test_filename = test_dir + "/tp_test.gz";

  std::cout << ">>>>> Uncompressed File Test <<<<<" << std::endl;
  bool uncompressed_test_pass = decompressor.DecompressFile(
      q, uncompressed_filename, "", 1, false, false);
  PrintTestResults("Uncompressed File Test", uncompressed_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Statically Compressed File Test <<<<<" << std::endl;
  bool static_test_pass = decompressor.DecompressFile(
      q, static_compress_filename, "", 1, false, false);
  PrintTestResults("Statically Compressed File Test", static_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Dynamically Compressed File Test <<<<<" << std::endl;
  bool dynamic_test_pass = decompressor.DecompressFile(
      q, dynamic_compress_filename, "", 1, false, false);
  PrintTestResults("Dynamically Compressed File Test", dynamic_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Throughput Test <<<<<" << std::endl;
  constexpr int kTPTestRuns = 5;
  bool tp_test_pass = decompressor.DecompressFile(q, tp_test_filename, "",
                                                  kTPTestRuns, true, false);
  PrintTestResults("Throughput Test", tp_test_pass);
  std::cout << std::endl;

  return uncompressed_test_pass && static_test_pass && dynamic_test_pass &&
         tp_test_pass;    
#endif

}
#endif

#if defined(SNAPPY)
bool RunSnappyTest(sycl::queue& q, SnappyDecompressorT decompressor,
                   const std::string test_dir) {


#ifdef FPGA_SIMULATOR
  std::cout << ">>>>> Alice In Wonderland Test <<<<<" << std::endl;
  std::string alice_in_file = test_dir + "/alice29_small.txt.sz";
  auto in_bytes = ReadInputFile(alice_in_file);
  auto result = decompressor.DecompressBytes(q, in_bytes, 1, false);

  std::string alice_ref_file = test_dir + "/alice29_small_ref.txt";
  auto ref_bytes = ReadInputFile(alice_ref_file);
  bool alice_test_pass =
      (result != std::nullopt) && (result.value() == ref_bytes);

  PrintTestResults("Alice In Wonderland Test", alice_test_pass);
  std::cout << std::endl;

  return alice_test_pass;
#else

  std::cout << ">>>>> Alice In Wonderland Test <<<<<" << std::endl;
  std::string alice_in_file = test_dir + "/alice29.txt.sz";
  auto in_bytes = ReadInputFile(alice_in_file);
  auto result = decompressor.DecompressBytes(q, in_bytes, 1, false);

  std::string alice_ref_file = test_dir + "/alice29.ref.txt";
  auto ref_bytes = ReadInputFile(alice_ref_file);
  bool alice_test_pass =
      (result != std::nullopt) && (result.value() == ref_bytes);

  PrintTestResults("Alice In Wonderland Test", alice_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Only Literal Strings Test <<<<<" << std::endl;
  auto test1_bytes = GenerateSnappyCompressedData(333, 3, 0, 0, 3);
  auto test1_ret = decompressor.DecompressBytes(q, test1_bytes, 1, false);
  bool test1_pass = test1_ret != std::nullopt;
  PrintTestResults("Only Literal Strings Test", test1_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Many Copies Test <<<<<" << std::endl;
  auto test2_bytes = GenerateSnappyCompressedData(65535, 1, 64, 13, 9);
  auto test2_ret = decompressor.DecompressBytes(q, test2_bytes, 1, false);
  bool test2_pass = test2_ret != std::nullopt;
  PrintTestResults("Many Copies Test", test2_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Mixed Literal Strings and Copies Test <<<<<" << std::endl;
  auto test3_bytes = GenerateSnappyCompressedData(16065, 7, 13, 5, 3);
  auto test3_ret = decompressor.DecompressBytes(q, test3_bytes, 1, false);
  bool test3_pass = test3_ret != std::nullopt;
  PrintTestResults("Mixed Literal Strings and Copies Test", test3_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Throughput Test <<<<<" << std::endl;
  constexpr int kTPTestRuns = 5;
#ifndef FPGA_EMULATOR
  auto test_tp_bytes = GenerateSnappyCompressedData(65536, 2, 0, 0, 128);
#else
  auto test_tp_bytes = GenerateSnappyCompressedData(65536, 2, 0, 0, 2);
#endif
  auto test_tp_ret =
      decompressor.DecompressBytes(q, test_tp_bytes, kTPTestRuns, true);
  bool test_tp_pass = test_tp_ret != std::nullopt;
  PrintTestResults("Throughput Test", test_tp_pass);
  std::cout << std::endl;

  return alice_test_pass && test1_pass && test2_pass && test3_pass &&
         test_tp_pass;
#endif

}
#endif
