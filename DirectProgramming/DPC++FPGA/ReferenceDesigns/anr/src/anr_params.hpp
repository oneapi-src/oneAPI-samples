#ifndef __ANR_PARAMS_HPP__
#define __ANR_PARAMS_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_fixed.hpp>
#include <iostream>
#include <string>
#include <utility>

//
// A struct to hold the ANR configuration paremeters
//
struct ANRParams {
  // the floating point format
  using FloatT = float;

  // the alpha blending computation uses fixed point format, these constants
  // hold the total number of bits and the number of integer bits (the number
  // of fractional bits is the difference between the two)
  static constexpr int kAlphaTotalBits = 9;
  static constexpr int kAlphaIntegerBits = 1;

  // the ac_fixed type for the alpha value
  using AlphaFixedT = ac_fixed<kAlphaTotalBits, kAlphaIntegerBits, false>;

  // default constructor
  ANRParams() {}

  // static method to parse the ANRParams from a file
  static ANRParams FromFile(std::string filename) {
    // the return object
    ANRParams ret;

    // create the file stream to parse
    std::ifstream is(filename);

    // make sure we opened the file fine
    if (!is.is_open() || is.fail()) {
      std::cerr << "ERROR: failed to open " << filename << " for reading\n";
      std::terminate();
    }

    // parse the lines
    std::string line;
    while (std::getline(is, line)) {
      size_t colon_pos = line.find(':');
      auto name = line.substr(0, colon_pos);
      auto val = std::stod(line.substr(colon_pos + 1));

      if (name == "sig_shot") {
        ret.sig_shot = val;
        ret.sig_shot_2 = val * val;
      } else if (name == "k") {
        ret.k = val;
      } else if (name == "sig_i_coeff") {
        ret.sig_i_coeff = val;
      } else if (name == "sig_s") {
        ret.sig_s = val;
      } else if (name == "alpha") {
        ret.alpha = val;
        ret.one_minus_alpha = 1 - val;
      } else if (name == "filter_size") {
        ret.filter_size = val;
      } else if (name == "pixel_bits") {
        ret.pixel_bits = val;
      } else {
        std::cerr << "WARNING: unknown name " << name
                  << " in ANRParams constructor\n";
      }
    }

    return ret;
  }

  int filter_size;     // filter size
  FloatT sig_shot;     // shot noise
  FloatT k;            // total gain
  FloatT sig_i_coeff;  // intensity sigma coefficient
  FloatT sig_s;        // spatial sigma
  FloatT alpha;        // alpha value for alpha blending
  int pixel_bits;      // the number of bits for each pixel

  // precomputed values
  FloatT sig_shot_2;       // shot noise squared
  FloatT one_minus_alpha;  // 1 - alpha
};

// convenience method for printing the ANRParams
std::ostream& operator<<(std::ostream& os, const ANRParams& params) {
  os << "sig_shot: " << params.sig_shot << "\n";
  os << "k: " << params.k << "\n";
  os << "sig_i_coeff: " << params.sig_i_coeff << "\n";
  os << "sig_s: " << params.sig_s << "\n";
  os << "alpha: " << params.alpha << "\n";
  return os;
}

#endif /* __ANR_PARAMS_HPP__ */