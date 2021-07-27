#ifndef __QFP_HPP__
#define __QFP_HPP__

#include <CL/sycl/INTEL/ac_types/ac_fixed.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#include <array>
#include <bitset>
#include <limits>

//
// QFP Type          | Exponent | Mantissa
//
// SquareQFP            5           5
// IntensitySigmaQFP    6           4
// ExpQFP               6           4
// InverseInQFP         3           7
// InverseOutQFP        4           6
//

//
// A static class that is used to convert to/from 32-bit floating point
// and quantized floating point (QFP) format.
//
template <unsigned qfp_total_bits, unsigned qfp_exponent_bits, bool is_signed>
struct QFP {
  QFP(const QFP&) = delete;
  QFP& operator=(const QFP&) = delete;

  // the ac_int type used to represent the QFP
  using ac_int_t = ac_int<qfp_total_bits, is_signed>;

  // 32-bit floating point bits based on
  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  static constexpr unsigned fp32_sign_bits = 1;
  static constexpr unsigned fp32_exponent_bits = 8;
  static constexpr unsigned fp32_mantissa_bits = 23;
  static constexpr unsigned fp32_total_bits =
      fp32_sign_bits + fp32_exponent_bits + fp32_mantissa_bits;

  // A union for accesing the mantissa, exponent, and sign bits
  typedef union {
    float f;
    struct {
      unsigned mantissa : fp32_mantissa_bits;
      unsigned exponent : fp32_exponent_bits;
      unsigned sign : fp32_sign_bits;
    } parts;
  } FloatCast;

  // the number of mantissa bits for the QFP
  static constexpr unsigned qfp_mantissa_bits =
      qfp_total_bits - qfp_exponent_bits - is_signed;

  // masks for the exponent and mantissa of the QFP
  static constexpr unsigned qfp_exponent_mask = (1 << qfp_exponent_bits) - 1;
  static constexpr unsigned qfp_mantissa_mask = (1 << qfp_mantissa_bits) - 1;

  // the difference in bits between the QFP and the 32-bit float
  static constexpr unsigned mantissa_bit_diff =
      fp32_mantissa_bits - qfp_mantissa_bits;

  // static asserts
  static_assert(fp32_total_bits == (sizeof(float) * 8));
  static_assert(qfp_mantissa_bits <= fp32_mantissa_bits);
  static_assert(qfp_total_bits > qfp_exponent_bits);

  //
  // convert from a 32-bit float to a QFP ac_int
  //
  static ac_int_t FromFP32(float f) {
    // use the float cast to get the parts of the FP32
    FloatCast f_casted = {.f = f};

    // get the most significant qfp_mantissa_bits from the float's mantissa
    unsigned qfp_mantissa =
        (f_casted.parts.mantissa >> mantissa_bit_diff) & qfp_mantissa_mask;

    // compute the QFP exponent
    const int qfp_exponent_tmp = (int(f_casted.parts.exponent) + 31 - 127);
    unsigned qfp_exponent = (qfp_exponent_tmp < 0) ? 0 : qfp_exponent_tmp;

    // get the sign bit
    unsigned qfp_sign = f_casted.parts.sign;

    // build the output ac_int
    if constexpr (is_signed) {
      return ac_int_t((qfp_sign << (qfp_exponent_bits + qfp_mantissa_bits)) |
                      (qfp_exponent << qfp_mantissa_bits) | (qfp_mantissa));
    } else {
      return ac_int_t((qfp_exponent << qfp_mantissa_bits) | (qfp_mantissa));
    }
  }

  //
  // convert a QFP ac_int to a 32-bit float
  //
  static float ToFP32(ac_int_t i) {
    unsigned sign_bit = 0;
    if constexpr (!is_signed) {
      sign_bit = (i >> (qfp_exponent_bits + qfp_mantissa_bits)) & 0x1;
    }
    unsigned fp32_exponent_tmp =
        int((i >> qfp_mantissa_bits) & qfp_exponent_mask);
    unsigned fp32_exponent =
        (fp32_exponent_tmp == 0) ? 0 : (fp32_exponent_tmp - 31 + 127);
    unsigned fp32_mantissa = (i & qfp_mantissa_mask) << mantissa_bit_diff;

    FloatCast f_casted;
    f_casted.parts.sign = sign_bit;
    f_casted.parts.exponent = fp32_exponent;
    f_casted.parts.mantissa = fp32_mantissa;

    return float(f_casted.f);
  }

 private:
  QFP();
};

#endif /* _QFP_HPP__ */