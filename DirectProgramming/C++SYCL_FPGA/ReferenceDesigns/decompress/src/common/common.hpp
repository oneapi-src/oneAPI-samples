#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <sycl/sycl.hpp>
#include <functional>
#include <iostream>
#include <optional>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "common_metaprogramming.hpp"
#include "constexpr_math.hpp"  // included from ../../../../include
#include "memory_utils.hpp"    // included from ../../../../include

// we only use unsigned ac_ints in this design, so this alias lets us not write
// the 'false' template argument every where
template <int bits>
using ac_uint = ac_int<bits, false>;

//
// Extend a type 'T' with a boolean flag
//
template <typename T>
struct FlagBundle {
  using value_type = T;

  // ensure the type carried in this class has a subscript operator and that
  // it has a static integer member named 'size'
  static_assert(fpga_tools::has_subscript_v<T>);

  // this is used by the functions in memory_utils.hpp to ensure the size of
  // the type in the SYCL pipe matches the memory width
  static constexpr size_t size = T::size;

  FlagBundle() : data(T()), flag(false) {}
  FlagBundle(T d_in) : data(d_in), flag(false) {}
  FlagBundle(T d_in, bool f_in) : data(d_in), flag(f_in) {}
  FlagBundle(bool f_in) : data(T()), flag(f_in) {}

  unsigned char& operator[](int i) { return data[i]; }
  const unsigned char& operator[](int i) const { return data[i]; }

  T data;
  bool flag;
};

//
// The generic data that goes into the LZ77 decoder from the Huffman decoder
//
//  Template parameters:
//    literals_per_cycle_: the number of literals coming into the LZ77 decoder
//      at once LZ77 decoder. This is NOT the literals_per_cycle that sets how
//      many literals the LZ77 decoder reads from the history buffer at once.
//    max_distance_: The maximum distance value in a {length, distance} pair.
//      This sets how may bits are required for the variable used to index into
//      the LZ77 decoder's history buffer.
//    max_length_: the maximum length value in a {lengt, distance} pair.
//
template <size_t literals_per_cycle_, size_t max_distance_, size_t max_length_>
struct LZ77InputData {
  static constexpr auto literals_per_cycle = literals_per_cycle_;
  static constexpr auto max_distance = max_distance_;
  static constexpr auto max_length = max_length_;

  static_assert(literals_per_cycle_ > 0);
  static_assert(max_length > 0);
  static_assert(max_distance > 0);

  static constexpr size_t size = literals_per_cycle_;
  static constexpr size_t max_literals = literals_per_cycle_;
  static constexpr size_t valid_count_bits = fpga_tools::Log2(max_literals) + 1;
  static constexpr size_t length_bits = fpga_tools::Log2(max_length) + 1;
  static constexpr size_t distance_bits = fpga_tools::Log2(max_distance) + 1;

  static_assert(valid_count_bits > 0);
  static_assert(literals_per_cycle < fpga_tools::Pow2(valid_count_bits));
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
    unsigned char literal[literals_per_cycle_];
  };

  // either the number of valid literals, or the distance in the
  // {length, distance} pair
  union {
    ac_uint<distance_bits> distance;
    ac_uint<valid_count_bits> valid_count;
  };

  unsigned char& operator[](int i) { return literal[i]; }
  const unsigned char& operator[](int i) const { return literal[i]; }
};

//
// Metaprogramming utils to check if a type is any instance of LZ77InputData
//
namespace detail {
template <typename T>
struct is_lz77_input_data_impl : std::false_type {};

template <unsigned a, unsigned b, unsigned c>
struct is_lz77_input_data_impl<LZ77InputData<a, b, c>> : std::true_type {};
}  // namespace detail

template <class T>
struct is_lz77_input_data {
  static constexpr bool value = detail::is_lz77_input_data_impl<T>{};
};

template <class T>
inline constexpr bool is_lz77_input_data_v = is_lz77_input_data<T>::value;

// The LZ77 datastructure specific for the GZIP decompressor
constexpr size_t kGzipMaxLZ77Length = 32768;
constexpr size_t kGzipMaxLZ77Distance = 32768;
using GzipLZ77InputData =
    LZ77InputData<1, kGzipMaxLZ77Distance, kGzipMaxLZ77Length>;

// The LZ77 datastructure specific for the Snappy decompressor
// Snappy V1.1 format sets the maximum history to 65K
// At the time of writing this, the maximum history distance will be 32K, but
// the specification claims support for 65K, so we will be safe.
constexpr size_t kSnappyMaxLZ77Length = 64;
constexpr size_t kSnappyMaxLZ77Distance = 1 << 16;
template <size_t n>
using SnappyLZ77InputData =
    LZ77InputData<n, kSnappyMaxLZ77Distance, kSnappyMaxLZ77Length>;

//
// Holds an array of bytes, where valid_count indicates how many of the 'n'
// bytes are valid. The valid bytes must be sequential and start at index 0.
// E.g., if valid_count = 2, then byte[0] and byte[1] are valid, while byte[2],
// byte[3], ..., byte[n-1] are not.
//
template <size_t num_bytes>
struct BytePack {
  static constexpr unsigned count_bits = fpga_tools::Log2(num_bytes) + 1;
  static_assert(count_bits > 0);
  static_assert(num_bytes < fpga_tools::Pow2(count_bits));
  static constexpr size_t size = num_bytes;

  unsigned char byte[num_bytes];
  ac_uint<count_bits> valid_count;

  unsigned char& operator[](int i) { return byte[i]; }
  const unsigned char& operator[](int i) const { return byte[i]; }
};

//
// Similar to a BytePack, but all of the bytes are valid.
//
template <size_t num_bytes>
struct ByteSet {
  static constexpr size_t size = num_bytes;

  unsigned char byte[num_bytes];

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
// Reads 'filename' and returns an array of chars (the bytes of the file)
//
std::vector<unsigned char> ReadInputFile(const std::string& filename) {
  // open file stream
  std::ifstream fin(filename, std::ios::binary);

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
void WriteOutputFile(const std::string& filename,
                     std::vector<unsigned char>& data) {
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
  //
  // A virtual function that must be overriden by a deriving class.
  // The overriding function performs the actual decompression using the FPGA.
  // See ../gzip/gzip_decompressor.hpp and ../snappy/snappy_decompressor.hpp
  // for the GZIP and SNAPPY versions of these, respectively
  //
  virtual std::optional<std::vector<unsigned char>> DecompressBytes(
      sycl::queue&, std::vector<unsigned char>&, int, bool) = 0;

  //
  // Reads the bytes in 'in_filename', decompresses them, and writes the
  // output to 'out_filename' (if write_output == true). This function uses
  // the DecompressBytes virtual function above to do the actual decompression.
  //
  // Arguments:
  //    q: the SYCL queue
  //    in_filename: the file path to the compressed input file
  //    out_filename: the file path where to write the output file
  //    runs: the number of times to call decompress the same file. This is for
  //      throughput testing purposes.
  //    print_stats: whether to print the execution time and throughput
  //      statistics to stdout
  //    write_output: whether to write the decompressed output to 'out_filename'
  //
  bool DecompressFile(sycl::queue& q, const std::string& in_filename,
                      const std::string& out_filename, int runs,
                      bool print_stats, bool write_output) {
    std::cout << "Decompressing '" << in_filename << "' " << runs
              << ((runs == 1) ? " time" : " times") << std::endl;

    auto in_bytes = ReadInputFile(in_filename);
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
// DirectProgramming/C++SYCL_FPGA/include/memory_utils.hpp to do this.
//
//  Template parameters:
//    Id: the type to use for the kernel ID
//    InPipe: a SYCL pipe that streams bytes into the decompression engine,
//      'literals_per_cycle' at a time
//    literals_per_cycle: the number of bytes to read from the pointer and
//      write to the pipe at once.
//
//  Arguments:
//    q: the SYCL queue
//    in_count_padded: the total number of bytes to read from in_ptr and write
//      to the input pipe. In this design, we pad the size to be a multiple of
//      literals_per_cycle.
//    in_ptr: a pointer to the input data
//
template <typename Id, typename InPipe, unsigned literals_per_cycle>
sycl::event SubmitProducer(sycl::queue& q, unsigned in_count_padded,
                           unsigned char* in_ptr) {
  assert(in_count_padded % literals_per_cycle == 0);
  auto iteration_count = in_count_padded / literals_per_cycle;
  return q.single_task<Id>([=] {
    // Use the MemoryToPipe utility to read from in_ptr 'literals_per_cycle'
    // elements at once and write them to 'InPipe'.
    // The 'false' template argument is our way of guaranteeing to the library
    // that 'literals_per_cycle is a multiple 'iteration_count'. In both the
    // GZIP and SNAPPY designs, we guarantee this in the DecompressBytes
    // functions in ../gzip/gzip_decompressor.hpp and
    // ../snappy/snappy_decompressor.hpp respectively.
#if defined (IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer
    // lives on the device.
    // Knowing this, the compiler won't generate hardware to
    // potentially get data from the host.
    sycl::device_ptr<unsigned char> in(in_ptr);
#else
    // Device pointers are not supported when targeting an FPGA 
    // family/part
    unsigned char* in(in_ptr);
#endif
    fpga_tools::MemoryToPipe<InPipe, literals_per_cycle, false>(
        in, iteration_count);
  });
}

//
// Same idea as SubmitProducer but in the opposite direction. Data is streamed
// from the SYCL pipe (OutPipe) and written to memory (out_ptr).
//
//  Template parameters:
//    Id: the type to use for the kernel ID
//    InPipe: a SYCL pipe that streams bytes from the decompression engine,
//      'literals_per_cycle' at a time
//    literals_per_cycle: the number of bytes to read from pipe and write to the
//      pointer at once.
//
//  Arguments:
//    q: the SYCL queue
//    out_count_padded: the total number of bytes to read from input pipe and
//      write to out_ptr. In this design, we pad the size to be a multiple of
//      literals_per_cycle.
//    out_ptr: a pointer to the output data
//
template <typename Id, typename OutPipe, unsigned literals_per_cycle>
sycl::event SubmitConsumer(sycl::queue& q, unsigned out_count_padded,
                           unsigned char* out_ptr) {
  assert(out_count_padded % literals_per_cycle == 0);
  auto iteration_count = out_count_padded / literals_per_cycle;
  return q.single_task<Id>([=] {
    // Use the PipeToMemory utility to read 'literals_per_cycle'
    // elements at once from 'OutPipe' and write them to 'out_ptr'.
    // For details about the 'false' template parameter, see the SubmitProducer
    // function above.

#if defined (IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer
    // lives on the device.
    // Knowing this, the compiler won't generate hardware to
    // potentially get data from the host.
    sycl::device_ptr<unsigned char> out(out_ptr);
#else
    // Device pointers are not supported when targeting an FPGA 
    // family/part
    unsigned char* out(out_ptr);
#endif
    
    fpga_tools::PipeToMemory<OutPipe, literals_per_cycle, false>(
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
