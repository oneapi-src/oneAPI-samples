#ifndef __QFP_INV_LUT_HPP__
#define __QFP_INV_LUT_HPP__

#include "qfp.hpp"
#include "rom_base.hpp"

// the QFP bits for the Pow2LUT
constexpr unsigned kInvQFPTotalBits = 10;
constexpr unsigned kInvQFPExponentBits = 3;
constexpr unsigned kInvLutDepth = (1 << kInvQFPTotalBits);
static_assert(kInvQFPTotalBits >= kInvQFPExponentBits);

//
// A LUT for computing 1/x
//
struct InvLUT : ROMBase<unsigned short, kInvLutDepth> {
  // the QFP format
  using QFP = QFP<kInvQFPTotalBits, kInvQFPExponentBits, false>;

  // the functor used to initialize the ROM
  // NOTE: anything called from within the functor's operator() MUST be
  // constexpr or else you won't get a ROM
  struct InitFunctor {
    constexpr unsigned short operator () (int x) const {
      // treat the ROM index as a QFP number and convert to a float (f) and use
      // the float to compute 1/f and initialize that entry of the ROM
      float f = QFP::ToFP32CE(x);
      float val = 1.0f / f ;
      return QFP::FromFP32CE(val);
    }
    constexpr InitFunctor() = default;
  };

  // constexpr constructor using the initializer above
  constexpr InvLUT() : ROMBase<unsigned short, kInvLutDepth>(InitFunctor()) {}
};

#endif /* __QFP_INV_LUT_HPP__ */