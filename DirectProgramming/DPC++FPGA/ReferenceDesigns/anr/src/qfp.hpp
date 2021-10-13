#ifndef __QFP_HPP__
#define __QFP_HPP__

#include <array>
#include <limits>

#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "mp_math.hpp"

//
// A static class that is used to convert to/from 32-bit floating point
// and quantized floating point (QFP) format. Note that, when converting from
// FP32 to QFP, we truncate the mantissa, instead of rounding. This reduces
// the area required for the conversion at the expense of decreased accuracy.
//
template <unsigned qfp_total_bits, unsigned qfp_exponent_bits, bool is_signed>
struct QFP {
  QFP(const QFP&) = delete;
  QFP& operator=(const QFP&) = delete;

  // determine if the QFP can fit into a unsigned char or unsigned short (if
  // not, default to an unsigned int)
  static constexpr bool fits_in_uchar =
    qfp_total_bits <= sizeof(unsigned char)*8;
  static constexpr bool fits_in_ushort =
    qfp_total_bits <= sizeof(unsigned short)*8;
  
  using qfp_type =
    std::conditional_t<fits_in_uchar, unsigned char,
      std::conditional_t<fits_in_ushort, unsigned short, unsigned>>;

  // 32-bit floating point bits based on
  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  static constexpr unsigned kFP32SignBits = 1;
  static constexpr unsigned kFP32ExponentBits = 8;
  static constexpr unsigned kFP32MantissaBits = 23;
  static constexpr unsigned kFP32TotalBits =
      kFP32SignBits + kFP32ExponentBits + kFP32MantissaBits;
  static constexpr int kFP32ExponentOffset =
    (1 << (kFP32ExponentBits-1)) - 1;
  static constexpr unsigned kFP32ExponentMask = (1 << kFP32ExponentBits) - 1;
  static constexpr unsigned kFP32MantissaMask = (1 << kFP32MantissaBits) - 1;

  // A union for accesing the mantissa, exponent, and sign bits
  typedef union {
    float f;
    struct {
      unsigned mantissa   : kFP32MantissaBits;
      unsigned exponent   : kFP32ExponentBits;
      unsigned sign       : kFP32SignBits;
    } parts;
  } FloatCast;

  // the number of mantissa bits for the QFP
  static constexpr unsigned qfp_mantissa_bits =
      qfp_total_bits - qfp_exponent_bits - is_signed;

  // masks for the exponent and mantissa of the QFP
  static constexpr unsigned qfp_mask = (1 << qfp_total_bits) - 1;
  static constexpr unsigned qfp_exponent_mask = (1 << qfp_exponent_bits) - 1;
  static constexpr unsigned qfp_mantissa_mask = (1 << qfp_mantissa_bits) - 1;
  static constexpr int qfp_exponent_offset =
      (1 << (qfp_exponent_bits - 1)) - 1;

  // the difference in bits between the QFP and the 32-bit float
  static constexpr unsigned mantissa_bit_diff =
      kFP32MantissaBits - qfp_mantissa_bits;

  // static asserts
  static_assert(kFP32TotalBits == (sizeof(float) * 8));
  static_assert(qfp_mantissa_bits <= kFP32MantissaBits);
  static_assert(qfp_exponent_bits > 0);
  static_assert(qfp_mantissa_bits > 0);
  static_assert(qfp_total_bits > qfp_exponent_bits);
  
  //
  // convert from a 32-bit float to a QFP
  //
  static qfp_type FromFP32(float f) {
    // use the float cast to get the parts of the FP32
    FloatCast f_casted = {.f = f};
    int fp32_sign = f_casted.parts.sign;
    ac_int<kFP32ExponentBits + 1, true> fp32_exponent =
        f_casted.parts.exponent;
    ac_int<kFP32MantissaBits, false> fp32_mantissa = f_casted.parts.mantissa;

    // get the most significant qfp_mantissa_bits from the float's mantissa
    // NOTE: we are doing truncation here without rounding, which will further
    // reduce accuracy but require less area.
    auto qfp_mantissa =
        (fp32_mantissa >> mantissa_bit_diff) & qfp_mantissa_mask;

    // compute the QFP exponent. Subtract the FP32 offset (127) from the FP32
    // exponent and add back the QFP exponent offset.
    auto qfp_exponent =
        fp32_exponent - kFP32ExponentOffset + qfp_exponent_offset;

    // get the sign bit
    int qfp_sign = fp32_sign;

    // build the output ac_int
    if constexpr (is_signed) {
      return qfp_type((qfp_sign << (qfp_exponent_bits + qfp_mantissa_bits)) |
                      (qfp_exponent << qfp_mantissa_bits) | (qfp_mantissa)) &
                      qfp_mask;
    } else {
      return qfp_type((qfp_exponent << qfp_mantissa_bits) | (qfp_mantissa)) &
                      qfp_mask;
    }
  }

  //
  // CONSTEXPR
  // convert from a 32-bit float to a QFP
  //
  static constexpr qfp_type FromFP32CE(float f) {
    // get the sign, exponent, and mantissa from the float
    int fp32_sign = (f < 0) ? 1 : 0;
    int fp32_exponent = FP32ExtractExponent(f) + kFP32ExponentOffset;
    int fp32_mantissa = FP32ExtractMantissa(f);

    // get the most significant qfp_mantissa_bits from the float's mantissa
    // NOTE: we are doing truncation here, not rounding.
    int qfp_mantissa =
        (fp32_mantissa >> mantissa_bit_diff) & qfp_mantissa_mask;

    // compute the QFP exponent. Subtract the FP32 offset (127) from the FP32
    // exponent and add back the QFP exponent offset.
    const int qfp_exponent_tmp =
        (fp32_exponent == 0) ? 0 :
        (int(fp32_exponent) - kFP32ExponentOffset + qfp_exponent_offset);
    int qfp_exponent = (qfp_exponent_tmp < 0) ? 0 : qfp_exponent_tmp;

    // get the sign bit
    int qfp_sign = fp32_sign;

    // build the output ac_int
    if constexpr (is_signed) {
      return qfp_type((qfp_sign << (qfp_exponent_bits + qfp_mantissa_bits)) |
                      (qfp_exponent << qfp_mantissa_bits) | (qfp_mantissa)) &
                      qfp_mask;
    } else {
      return qfp_type((qfp_exponent << qfp_mantissa_bits) | (qfp_mantissa)) &
                      qfp_mask;
    }
  }

  //
  // convert a QFP to a 32-bit float
  //
  static float ToFP32(qfp_type i) {
    int sign_bit = 0;
    if constexpr (!is_signed) {
      sign_bit = (i >> (qfp_exponent_bits + qfp_mantissa_bits)) & 0x1;
    }

    ac_int<kFP32ExponentBits + 1, true> fp32_exponent_tmp =
        (i >> qfp_mantissa_bits) & qfp_exponent_mask;
    auto fp32_exponent =
        fp32_exponent_tmp - qfp_exponent_offset + kFP32ExponentOffset;
    ac_int<kFP32MantissaBits, true> fp32_mantissa =
        (i & qfp_mantissa_mask) << mantissa_bit_diff;

    FloatCast f_casted;
    f_casted.parts.sign = sign_bit;
    f_casted.parts.exponent = fp32_exponent;
    f_casted.parts.mantissa = fp32_mantissa;

    return f_casted.f;
  }

  //
  // CONSTEXPR
  // convert a QFP to a 32-bit float
  //
  static constexpr float ToFP32CE(qfp_type i) {
    int sign_bit = 0;
    if constexpr (!is_signed) {
      sign_bit = (i >> (qfp_exponent_bits + qfp_mantissa_bits)) & 0x1;
    }
    int fp32_exponent_tmp =
        int((i >> qfp_mantissa_bits) & qfp_exponent_mask);
    int fp32_exponent =
        fp32_exponent_tmp - qfp_exponent_offset + kFP32ExponentOffset;
    int fp32_mantissa = (i & qfp_mantissa_mask) << mantissa_bit_diff;

    int offset_exponent =
        (fp32_exponent == 0) ? 0 : (fp32_exponent - kFP32ExponentOffset);
    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    //compute the mantissa sum
    float mantissa_sum = (fp32_exponent == 0) ? 0.0 : 1.0;
    for (int i = 1; i <= kFP32MantissaBits; i++) {
      mantissa_sum +=
          ((fp32_mantissa >> (kFP32MantissaBits-i)) & 0x1) * Pow(2, -i);
    }

    if (sign_bit == 0) return Pow(2, offset_exponent) * mantissa_sum;
    else return -1.0f * Pow(2, offset_exponent) * mantissa_sum;
  }

 private:
  QFP();
};

#endif /* _QFP_HPP__ */