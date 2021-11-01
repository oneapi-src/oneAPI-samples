#ifndef __QFP_EXP_LUT_HPP__
#define __QFP_EXP_LUT_HPP__

#include "qfp.hpp"
#include "rom_base.hpp"

// the QFP bits for the ExpLUT
constexpr unsigned kExpQFPTotalBits = 10;
constexpr unsigned kExpQFPExponentBits = 6;
constexpr unsigned kExpLUTDepth = (1 << kExpQFPTotalBits);
constexpr int kExpTaylorSeriesTerms = 70;

static_assert(kExpQFPTotalBits >= kExpQFPExponentBits);
static_assert(kExpTaylorSeriesTerms > 3);

//
// A LUT for computing exp(-x)
// Uses ROMBase to create a ROM initialized with the values of exp(-x)
// using quantized floating point (QFP) numbers for indices.
//
struct ExpLUT : ROMBase<unsigned short, kExpLUTDepth> {
  // the QFP format
  using QFP = QFP<kExpQFPTotalBits, kExpQFPExponentBits, false>;

  // the functor used to initialize the ROM
  // NOTE: anything called from within the functor's operator() MUST be
  // constexpr or else you won't get a ROM
  struct InitFunctor {
    constexpr unsigned short operator () (int x) const {
      // treat the ROM index as a QFP number and convert to a float (f) and use
      // the float to compute exp(-f) (== 1/exp(f)) and initialize that entry
      // of the ROM
      float f = QFP::ToFP32CE(x);
      float val = 1.0f / hldutils::Exp(f, kExpTaylorSeriesTerms);
      return QFP::FromFP32CE(val);
    }
    constexpr InitFunctor() = default;
  };

  // constexpr constructor using the initializer above
  constexpr ExpLUT() : ROMBase<unsigned short, kExpLUTDepth>(InitFunctor()) {}
};

#endif /*__QFP_EXP_LUT_HPP__*/
