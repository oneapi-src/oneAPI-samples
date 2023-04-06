/**
   T - 32, 64, 128bits
   all inputs < modulus
   inv_modulus - (2^(sizeof(T)*8) / modulus)
*/

// Mod ops
//  value % modulus -> return
//  aka barret_reduce64
template <typename T>
T XeHE_barrett_reduce(const T value, const T modulus, const T inv_modulus[2])
{
  // Reduces input using base 2^(sizeof(T)*8) Barrett reduction
  // floor(2^64 / mod) == floor( floor(2^128 / mod) )
  T tmp[2], tmp;

  multiply_uint<T>(input, inv_modulus[1], tmp);

  // Barrett subtraction
  tmp = input - tmp[1] * modulus;

  // One more subtraction is enough
  return static_cast<T>(tmp) -
    (modulus & static_cast<T>(-static_cast<T>(tmp >= modulus)));
}

//  (value[2] % modulus) -> return
//  aka barret_reduce128
// input allocation size must be 128 bits
template <typename T>
T XeHE_barrett_reduce(const T * value, const T modulus, const T inv_modulus[2])
{

  // Reduces input using base 2^(sizeof(T)*8) Barrett reduction

  T tmp1, tmp2[2], tmp3 = 0, carry[2];

  // Multiply input and inv modulus
  // Round 1
  multiply_uint<T>(input[0], inv_modulus[0], carry);


  multiply_uint<T>(input[0], inv_modulus[1], tmp2);
  tmp3 = tmp2[1] + add_uint<T>(tmp2[0], carry[1]);

            // Round 2
  multiply_uint<T>(input[1], inv_modulus[0], tmp2);
  carry[0] = tmp2[1] + add_uint64<T>(tmp1, tmp2[0]);

  // This is all we care about
  tmp1 = input[1] * inv_modulus[1] + tmp3 + carry[0];

  // Barrett subtraction
  tmp3 = input[0] - tmp1 * modulus;
  // One more subtraction is enough
  return static_cast<T>(tmp3) - (modulus & static_cast<T>(-static_cast<T>(tmp3 >= modulus)));

}

// (operand1 + operand2) % modulus -> return
template <typename T>
T XeHE_add_mod(const T operand1, const T operand2, const T modulus) {
  T sum = operand1 + operand2;
  return (sum >= modulus) ? (sum - modulus) : sum;
}

// (operand1 - operand2) % modulus -> return
template <typename T>
T XeHE_sub_mod(const T operand1, const T operand2, const T modulus) {
  T diff = (operand1 + modulus) - operand2;
  return (diff >= modulus) ? (diff - modulus) : diff;
}

// (operand1 * operand2) % modulus -> return
template <typename T>
T XeHE_mul_mod(const T operand1, const T operand2, const T modulus,
               const T inv_modulus[2])
{
  T z[2];
  multiply_uint64<T>(operand1, operand2, z);
  return XeHE_barrett_reduce(z, modulus, inv_modulus);
}

// (operand1 * operand2 + operand3) % modulus -> return
template <typename T>
T XeHE_mad_mod(const T operand1, const T operand2, const T operand3,
               const T modulus, const T inv_modulus[2]);
