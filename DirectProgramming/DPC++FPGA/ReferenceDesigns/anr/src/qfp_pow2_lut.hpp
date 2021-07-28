#ifndef __QFP_POW2_LUT_HPP__
#define __QFP_POW2_LUT_HPP__

#include "qfp.hpp"
#include "rom_base.hpp"

// the QFP bits for the Pow2LUT
constexpr unsigned pow2_qfp_total_bits = 10;
constexpr unsigned pow2_qfp_exponent_bits = 6;
constexpr unsigned pow2_lut_depth=(1 << exp_qfp_total_bits);
static_assert(pow2_qfp_total_bits >= pow2_qfp_exponent_bits);

//
// A LUT for computing x^2
// Uses ROMBase to create a ROM initialized with the values of exp(-x)
// using quantized floating point (QFP) numbers for indices.
//
struct Pow2LUT : ROMBase<float, pow2_lut_depth> {
  // the QFP format
  using QFP = QFP<pow2_qfp_total_bits, pow2_qfp_exponent_bits, false>;

  // the functor used to initialize the ROM
  // NOTE: anything called from within the functor's operator() MUST be
  // constexpr or else you won't get a ROM
  struct InitFunctor {
    constexpr float operator () (int x) const {
      // treat the ROM index as a QFP number and convert to a float (f) and use
      // the float to compute f^2 and initialize that entry of the ROM
      float f = QFP::ToFP32(x);
      return f * f;
    }
    constexpr InitFunctor() = default;
  };

  // constexpr constructor using the initializer above
  constexpr Pow2LUT() : ROMBase<float, pow2_lut_depth>(InitFunctor()) {}
};

#endif /* __QFP_POW2_LUT_HPP__ */