#ifndef __COMMON_HPP__
#define __COMMON_HPP__

// clang-format off
#include <iostream>
#include <functional>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Included from DirectProgramming/DPC++FPGA/include/
#include "constexpr_math.hpp"
#include "memory_utils.hpp"
// clang-format on

// we only use unsigned ac_ints, so this alias lets us not write the 'false'
// template argument everytime
template <int bits>
using ac_uint = ac_int<bits, false>;

//
// Extend a type 'T' with a boolean flag
//
template <typename T>
struct FlagBundle {
  using value_type = T;
  static_assert(fpga_tools::has_subscript_v<T>);
  FlagBundle() : data(T()), flag(false) {}
  FlagBundle(T d_in) : data(d_in), flag(false) {}
  FlagBundle(T d_in, bool f_in) : data(d_in), flag(f_in) {}
  FlagBundle(bool f_in) : data(T()), flag(f_in) {}

  // this is used by the functions in memory_utils.hpp to ensure the size of
  // the type in the SYCL pipe matches the memory width
  static constexpr size_t size = T::size;

  unsigned char& operator[](int i) { return data[i]; }
  const unsigned char& operator[](int i) const { return data[i]; }

  T data;
  bool flag;
};

//
// The data that goes into the LZ77 decoder from the Huffman decoder
//
template <unsigned n, unsigned max_length, unsigned max_distance>
struct LZ77InputData {
  static_assert(n > 0);
  static_assert(max_length > 0);
  static_assert(max_distance > 0);
  
  static constexpr unsigned size = n;
  static constexpr unsigned max_literals = n;
  static constexpr unsigned valid_count_bits =
      fpga_tools::Log2(max_literals) + 1;
  static constexpr unsigned length_bits = fpga_tools::Log2(max_length) + 1;
  static constexpr unsigned distance_bits = fpga_tools::Log2(max_distance) + 1;

  static_assert(valid_count_bits > 0);
  static_assert(n < fpga_tools::Pow2(valid_count_bits));
  static_assert(length_bits > 0);
  static_assert(max_length < fpga_tools::Pow2(length_bits));
  static_assert(distance_bits > 0);
  static_assert(max_distance < fpga_tools::Pow2(distance_bits));

  LZ77InputData() {}

  // indicates whether this is a literal or {length, distance} pair
  bool is_literal;

  // either the literals, or the length from the {length, distance} pair
  union {
    ac_uint<length_bits> length;
    unsigned char literal[n];
  };

  // either the number of valid literals, or the distance from the
  // {length, distance} pair
  union {
    ac_uint<distance_bits> distance;
    ac_uint<valid_count_bits> valid_count;
  };

  unsigned char& operator[](int i) { return literal[i]; }
  const unsigned char& operator[](int i) const { return literal[i]; }
};

// The LZ77 datastructure specific for the GZIP decompressor
constexpr unsigned kGzipMaxLZ77Length = 32768;
constexpr unsigned kGzipMaxLZ77Distance = 32768;
template <unsigned n>
using GzipLZ77InputData =
    LZ77InputData<n, kGzipMaxLZ77Length, kGzipMaxLZ77Distance>;

// The LZ77 datastructure specific for the Snappy decompressor
// Snappy V1.1 format sets the maximum history to 65K
// At the time of writing this, the maximum history distance will be 32K, but
// the specification claims support for 65K, so we will be safe.
constexpr unsigned kSnappyMaxLZ77Length = 64;
constexpr unsigned kSnappyMaxLZ77Distance = 1 << 16;
template <unsigned n>
using SnappyLZ77InputData =
    LZ77InputData<n, kSnappyMaxLZ77Length, kSnappyMaxLZ77Distance>;

//
// Holds an array of bytes, where valid_count indicates how many of the 'n'
// bytes are valid. The valid bytes must be sequential and start at index 0.
// E.g., if valid_count = 2, then byte[0] and byte[1] are valid, while byte[2],
// byte[3], ..., byte[n-1] are not.
//
template <unsigned n>
struct BytePack {
  static constexpr unsigned count_bits = fpga_tools::Log2(n) + 1;
  static_assert(count_bits > 0);
  static_assert(n < fpga_tools::Pow2(count_bits));
  static const size_t size = n;

  unsigned char byte[n];
  ac_uint<count_bits> valid_count;

  unsigned char& operator[](int i) { return byte[i]; }
  const unsigned char& operator[](int i) const { return byte[i]; }
};

//
// Similar to a BytePack, but all of the bytes are valid.
//
template <int n>
struct ByteSet {
  static const size_t size = n;

  unsigned char byte[n];

  unsigned char& operator[](int i) { return byte[i]; }
  const unsigned char& operator[](int i) const { return byte[i]; }
};

//
// returns the number of trailing zeros in an ac_int
// E.g. 0b011101000 has 3 trailing zeros
//
template <int bits, bool is_signed>
auto CTZ(const ac_int<bits, is_signed>& in) {
  static_assert(bits > 0);
  constexpr int out_bits = fpga_tools::Log2(bits) + 1;
  ac_uint<out_bits> ret(bits);
#pragma unroll
  for (int i = bits - 1; i >= 0; i--) {
    if (in[i]) {
      ret = i;
    }
  }
  return ret;
}

//
// Selects a SYCL device using a string. This is typically used to select
// the FPGA simulator device
//
class select_by_string : public sycl::default_selector {
 public:
  select_by_string(std::string s) : target_name(s) {}
  virtual int operator()(const sycl::device& device) const {
    std::string name = device.get_info<sycl::info::device::name>();
    if (name.find(target_name) != std::string::npos) {
      // The returned value represents a priority, this number is chosen to be
      // large to ensure high priority
      return 10000;
    }
    return -1;
  }

 private:
  std::string target_name;
};

//
// Reads 'filename' and returns an array of chars (the bytes of the file)
//
std::vector<unsigned char> ReadInputFile(std::string filename) {
  // open file stream
  std::ifstream fin(filename);

  // make sure it opened
  if (!fin.good() || !fin.is_open()) {
    std::cerr << "ERROR: could not open " << filename << " for reading\n";
    std::terminate();
  }

  // read in bytes
  std::vector<unsigned char> result;
  char tmp;
  while (fin.get(tmp)) {
    result.push_back(tmp);
  }
  fin.close();

  return result;
}

//
// Writes the chars (bytes) from 'data' to 'filename'
//
void WriteOutputFile(std::string filename, std::vector<unsigned char>& data) {
  // open file stream
  std::ofstream fout(filename.c_str());

  // make sure it opened
  if (!fout.good() || !fout.is_open()) {
    std::cerr << "ERROR: could not open " << filename << " for writing\n";
    std::terminate();
  }

  // write out bytes
  for (auto& c : data) {
    fout << c;
  }
  fout.close();
}

//
// A base class for a decmompressor
// This class is purely virtual, i.e. another class must inherit from it and
// override the 'DecompressBytes' function. This is done in
// ../gzip/gzip_decompress.hpp and ../src/snappy/snappy_decompressor.hpp for the
// GZIP and SNAPPY decompressors, respectively.
//
class DecompressorBase {
 public:
  virtual std::optional<std::vector<unsigned char>> DecompressBytes(
      sycl::queue&, std::vector<unsigned char>, int, bool) = 0;

  bool DecompressFile(sycl::queue& q, std::string in_filename,
                      std::string out_filename, int runs, bool print_stats,
                      bool write_output) {
    std::cout << "Decompressing '" << in_filename << "' " << runs
              << ((runs == 1) ? " time" : " times") << std::endl;

    std::vector<unsigned char> in_bytes = ReadInputFile(in_filename);
    auto result = DecompressBytes(q, in_bytes, runs, print_stats);

    if (result != std::nullopt) {
      if (write_output) {
        std::cout << std::endl;
        std::cout << "Writing output data to '" << out_filename << "'"
                  << std::endl;
        std::cout << std::endl;
        WriteOutputFile(out_filename, result.value());
      }
      return true;
    } else {
      return false;
    }
  }
};

//
// The Producer kernel reads 'literals_per_cycle' elements at a time from
// memory (in_ptr) and writes them into InPipe. We use the utilities from
// DirectProgramming/DPC++FPGA/include/memory_utils.hpp to do this.
// In this design, the host code in main.cpp guarantees that
// literals_per_cycle is a multiple of in_count_padded and therefore we
// don't need to account for remainders.
//
template <typename Id, typename InPipe, unsigned literals_per_cycle>
sycl::event SubmitProducer(sycl::queue& q, unsigned in_count_padded,
                           unsigned char* in_ptr) {
  assert(in_count_padded % literals_per_cycle == 0);
  auto iteration_count = in_count_padded / literals_per_cycle;
  return q.single_task<Id>([=] {
    // use the MemoryToPipe utility to read from in_ptr 'literals_per_cycle'
    // elements per cycle and write them to 'InPipe'
    sycl::device_ptr<unsigned char> in(in_ptr);
    fpga_tools::MemoryToPipe<InPipe, literals_per_cycle, true>(in,
                                                               iteration_count);
  });
}

//
// Same idea as SubmitProducer but in the opposite direction. Data is streamed
// from the SYCL pipe (OutPipe) and written to memory (out_ptr).
//
template <typename Id, typename OutPipe, unsigned literals_per_cycle>
sycl::event SubmitConsumer(sycl::queue& q, unsigned out_count_padded,
                           unsigned char* out_ptr) {
  assert(out_count_padded % literals_per_cycle == 0);
  auto iteration_count = out_count_padded / literals_per_cycle;
  return q.single_task<Id>([=] {
    // use the PipeToMemory utility to read 'literals_per_cycle'
    // elements per cycle from 'OutPipe' and write them to 'out_ptr'
    sycl::device_ptr<unsigned char> out(out_ptr);
    fpga_tools::PipeToMemory<OutPipe, literals_per_cycle, true>(
        out, iteration_count);

    // read the last 'done' signal
    bool done = false;
    while (!done) {
      bool valid;
      auto d = OutPipe::read(valid);
      done = d.flag && valid;
    }
  });
}

#endif /* __COMMON_HPP__ */