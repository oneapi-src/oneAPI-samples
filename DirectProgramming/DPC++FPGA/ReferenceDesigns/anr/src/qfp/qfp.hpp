#ifndef __QFP_HPP__
#define __QFP_HPP__

#include <array>
#include <bitset>
#include <limits>

#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#include <CL/sycl/INTEL/ac_types/ac_fixed.hpp>


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
// A union for getting the mantissa, exponent, and sign bits from a float
//
typedef union {
  float f;
  struct {
    unsigned mantissa  : 23;
    unsigned exponent : 8;
    unsigned sign     : 1;
  } parts;
} FloatCast;

//
// TODO
//
template<unsigned qfp_total_bits, unsigned qfp_exponent_bits, bool is_signed>
struct QFP {
  QFP(const QFP&) = delete;
  QFP& operator=(const QFP&) = delete;

  using ac_int_t = ac_int<qfp_total_bits, is_signed>;

  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  static constexpr unsigned fp32_sign_bits = 1;
  static constexpr unsigned fp32_exponent = 8;
  static constexpr unsigned fp32_mantissa_bits = 23;

  static constexpr unsigned qfp_mantissa_bits =
    qfp_total_bits - qfp_exponent_bits - is_signed;
  static constexpr unsigned qfp_exponent_mask = (1 << qfp_exponent_bits) - 1;
  static constexpr unsigned qfp_mantissa_mask = (1 << qfp_mantissa_bits) - 1;
  static constexpr unsigned mantissa_bit_diff =
    fp32_mantissa_bits - qfp_mantissa_bits;
  
  static_assert((fp32_sign_bits+fp32_exponent+fp32_mantissa_bits) == (sizeof(float)*8));
  static_assert(qfp_mantissa_bits <= fp32_mantissa_bits);
  static_assert(qfp_total_bits > qfp_exponent_bits);

  //
  // convert from a 32-bit float to a QFP ac_int
  //
  static ac_int_t FromFP32(float f) {
    // use the float cast to get the parts of the FP32
    FloatCast f_casted = { .f = f };
    
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
                      (qfp_exponent << qfp_mantissa_bits) |
                      (qfp_mantissa));
    } else {
      return ac_int_t((qfp_exponent << qfp_mantissa_bits) |
                      (qfp_mantissa));
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
    unsigned fp32_exponent_tmp = int((i >> qfp_mantissa_bits) & qfp_exponent_mask);
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