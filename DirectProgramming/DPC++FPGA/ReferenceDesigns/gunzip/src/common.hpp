#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "constexpr_math.hpp"

// we only use unsigned ac_ints, so use this alias to avoid having to type
// 'false' all the time
template<int bits>
using ac_uint = ac_int<bits, false>;

//
// Append a flag to a type 'T'
//
template<typename T>
struct FlagBundle {
  using value_type = T;
  FlagBundle() : data(T()), flag(false) {}
  FlagBundle(T d_in) : data(d_in), flag(false) {}
  FlagBundle(T d_in, bool f_in) : data(d_in), flag(f_in) {}
  FlagBundle(bool f_in) : data(T()), flag(f_in) {}

  T data;
  bool flag;
};

//
// The data that comes out of the huffman decoder
//
template<unsigned n, unsigned max_length, unsigned max_distance>
struct LZ77InputData {
  static_assert(n > 0);
  static_assert(max_length > 0);
  static_assert(max_distance > 0);

  static constexpr unsigned max_symbols = n;
  // TODO: store value - 1 of valid_count, length, and offset to save 1 bit each
  static constexpr unsigned valid_count_bits =
    fpga_tools::Log2(max_symbols) + 1;
  static constexpr unsigned length_bits =
    fpga_tools::Log2(max_length) + 1;
  static constexpr unsigned distance_bits =
    fpga_tools::Log2(max_distance) + 1;

  LZ77InputData() {}
  
  bool is_copy;
  union {
    ac_uint<length_bits> length;
    unsigned char symbol[n];
  };
  union {
    ac_uint<distance_bits> distance;
    ac_uint<valid_count_bits> valid_count;
  };
};

//
// Holds a pack of bytes. valid_count indicates how many of the 'n' bytes are
// valid. The valid bytes must be sequential. E.g., if valid_count = 2, then
// byte[0] and byte[1] are valid, while byte[2], byte[3], ..., byte[n-1] are
// not
//
template<int n>
struct BytePack {
  static constexpr int count_bits = fpga_tools::Log2(n) + 1;
  unsigned char byte[n];
  ac_uint<count_bits> valid_count;
};

//
// Similar to a BytePack, but all of the bytes are valid.
//
template<int n>
struct ByteSet {
  unsigned char byte[n];
};

//
// returns the number of trailing zeros
//
template<int bits>
auto CTZ(const ac_uint<bits>& in) {
  constexpr int out_bits = fpga_tools::Log2(bits) + 1;
  //ac_uint<out_bits> ret(bits);
  ac_uint<out_bits> ret;
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
  virtual int operator()(const sycl::device &device) const {
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

constexpr unsigned kMaxLZ77Length = 32768;
constexpr unsigned kMaxLZ77Distance = 32768;
template<unsigned n>
using GzipLZ77InputData = LZ77InputData<n, kMaxLZ77Length, kMaxLZ77Distance>;

#endif /* __COMMON_HPP__ */