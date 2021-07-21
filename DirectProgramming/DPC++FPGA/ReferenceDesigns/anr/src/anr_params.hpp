#ifndef __ANR_PARAMS_HPP__
#define __ANR_PARAMS_HPP__

#include <iostream>
#include <string> 

//
// A struct to hold the ANR configuration paremeters
//
struct ANRParams {
  using FloatT = float;

  ANRParams() {}
  ANRParams(FloatT _sig_shot, FloatT _k, FloatT _sig_i_coeff, FloatT _sig_s,
            FloatT _alpha) : sig_shot(_sig_shot), k(_k),
            sig_i_coeff(_sig_i_coeff), sig_s(_sig_s), alpha(_alpha),
            sig_shot_2(_sig_shot*_sig_shot), one_minus_alpha(1 - _alpha) {}
  
  
  static ANRParams FromFile(std::string filename) {
    // the return object
    ANRParams ret;

    // create the file stream to parse
    std::ifstream is(filename);

    std::string line;
    while (std::getline(is, line)) {
      size_t colon_pos = line.find(':');
      auto name = line.substr(0, colon_pos);
      auto val = std::stod(line.substr(colon_pos+1));

      if (name == "sig_shot") {
        ret.sig_shot = val;
        ret.sig_shot_2 = val*val;
      } else if (name == "k") {
        ret.k = val;
      } else if (name == "sig_i_coeff") {
        ret.sig_i_coeff = val;
      } else if (name == "sig_s") {
        ret.sig_s = val;
      } else if (name == "alpha") {
        ret.alpha = val;
        ret.one_minus_alpha = FloatT(1) - val;
      } else if (name == "filter_size") {
        ret.filter_size = val;
      } else {
        std::cerr << "WARNING: unknown name " << name
                  << " in ANRParams constructor\n";
      }
    }

    return ret;
  }

  int filter_size;  // filter size
  FloatT sig_shot;  // shot noise
  FloatT k;  // total gain
  FloatT sig_i_coeff;  // intensity sigma coefficient
  FloatT sig_s;  // spatial sigma
  FloatT alpha;  // alpha value for alpha blending

  // precomputed values
  FloatT sig_shot_2;  // shot noise squared
  FloatT one_minus_alpha;  // 1 - alpha
};

std::ostream& operator<<(std::ostream& os, const ANRParams& params) {
  os << "sig_shot: " << params.sig_shot << "\n";
  os << "k: " << params.k << "\n";
  os << "sig_i_coeff: " << params.sig_i_coeff << "\n";
  os << "sig_s: " << params.sig_s << "\n";
  os << "alpha: " << params.alpha << "\n";
  return os;
}

#endif /* __ANR_PARAMS_HPP__ */