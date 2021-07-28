#ifndef __QFP_EXP_LUT_HPP__
#define __QFP_EXP_LUT_HPP__

#include "qfp.hpp"
#include "rom_base.hpp"

// the QFP bits for the ExpLUT
constexpr unsigned exp_qfp_total_bits = 10;
constexpr unsigned exp_qfp_exponent_bits = 6;
constexpr unsigned exp_lut_depth=(1 << exp_qfp_total_bits);
static_assert(exp_qfp_total_bits >= exp_qfp_exponent_bits);

//
// A LUT for computing exp(-x)
// Uses ROMBase to create a ROM initialized with the values of exp(-x)
// using quantized floating point (QFP) numbers for indices.
//
struct ExpLUT : ROMBase<float, exp_lut_depth> {
  // the QFP format
  using QFP = QFP<exp_qfp_total_bits, exp_qfp_exponent_bits, false>;

  // the functor used to initialize the ROM
  // NOTE: anything called from within the functor's operator() MUST be
  // constexpr or else you won't get a ROM
  struct InitFunctor {
    constexpr float operator () (int x) const {
      // treat the ROM index as a QFP number and convert to a float (f) and use
      // the float to compute exp(-f) (== 1 / exp(f)) and initialize that entry
      // of the ROM
      float f = QFP::ToFP32(x);
      return (1.0f / hldutils::Exp(f, 70));
    }
    constexpr InitFunctor() = default;
  };

  // constexpr constructor using the initializer above
  constexpr ExpLUT() : ROMBase<float, exp_lut_depth>(InitFunctor()) {}
};

#endif /*__QFP_EXP_LUT_HPP__*/
