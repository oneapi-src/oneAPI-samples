// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "util/xe_uintarith.h"
#include "util/xe_uintcore.h"
#include <cstdint>
#include <type_traits>

namespace xehe
{
    namespace util
    {
        

#if 0
        /**
        Returns (operand++) mod modulus.
        Correctness: operand must be at most (2 * modulus -2) for correctness.
        */
        SEAL_NODISCARD inline std::uint64_t increment_uint_mod(std::uint64_t operand, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (operand > (modulus.value() - 1) << 1)
            {
                throw std::out_of_range("operand");
            }
#endif
            operand++;
            return operand - (modulus.value() &
                              static_cast<std::uint64_t>(-static_cast<std::int64_t>(operand >= modulus.value())));
        }

        /**
        Returns (operand--) mod modulus.
        @param[in] operand Must be at most (modulus - 1).
        */
        SEAL_NODISCARD inline std::uint64_t decrement_uint_mod(std::uint64_t operand, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (operand >= modulus.value())
            {
                throw std::out_of_range("operand");
            }
#endif
            std::int64_t carry = static_cast<std::int64_t>(operand == 0);
            return operand - 1 + (modulus.value() & static_cast<std::uint64_t>(-carry));
        }
#endif
        /**
        Returns (-operand) mod modulus.
        Correctness: operand must be at most modulus for correctness.
        */
        template <typename T>
        XeHE_NODISCARD T negate_uint_mod(T operand, const T modulus)
        {
#ifdef XeHE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand >= modulus)
            {
                throw std::out_of_range("operand");
            }
#endif
            auto non_zero = static_cast<int>(operand != 0);
            return (modulus - operand) & static_cast<T>(-non_zero);
        }

        /**
        Returns (operand * inv(2)) mod modulus.
        Correctness: operand must be even and at most (2 * modulus - 2) or odd and at most (modulus - 2).
        @param[in] operand Should be at most (modulus - 1).
        */
        template <typename T>
        XeHE_NODISCARD T div2_uint_mod(T operand, const T modulus)
        {
#ifdef XeHE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand >= modulus)
            {
                throw std::out_of_range("operand");
            }
#endif
            if (operand & 1)
            {
                T temp;
                auto carry = add_uint64(operand, modulus, 0, &temp);
                operand = temp >> 1;
                if (carry)
                {
                    return operand | (T(1) << ((sizeof(T) == 8) ? (bits_per_uint64 - 1) : (bits_per_uint32 - 1)));
                }
                return operand;
            }
            return operand >> 1;
        }

        /**
        Returns (operand1 + operand2) mod modulus.
        Correctness: (operand1 + operand2) must be at most (2 * modulus - 1).
        */
        template <typename T>
        XeHE_NODISCARD T add_uint_mod(
            T operand1, T operand2, const T modulus)
        {
#ifdef XeHE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1 + operand2 >= modulus << 1)
            {
                throw std::out_of_range("operands");
            }
#endif
            // Sum of operands modulo Modulus can never wrap around 2^64
            operand1 += operand2;
            return operand1 - (modulus &
                               static_cast<T>(-static_cast<int>(operand1 >= modulus)));
        }

        /**
        Returns (operand1 - operand2) mod modulus.
        Correctness: (operand1 - operand2) must be at most (modulus - 1) and at least (-modulus).
        @param[in] operand1 Should be at most (modulus - 1).
        @param[in] operand2 Should be at most (modulus - 1).
        */
        template <typename T>
        XeHE_NODISCARD T sub_uint_mod(
            T operand1, T operand2, const T modulus)
        {
#ifdef XeHE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }

            if (operand1 >= modulus)
            {
                throw std::out_of_range("operand1");
            }
            if (operand2 >= modulus)
            {
                throw std::out_of_range("operand2");
            }
#endif
            T temp;
            auto borrow = static_cast<T>(XeHE_SUB_BORROW_UINT64(operand1, operand2, 0, &temp));
            return static_cast<T>(temp) + (modulus & static_cast<T>(-borrow));
        }

        /**
        Returns input mod modulus. This is not standard Barrett reduction.
        Correctness: modulus must be at most 63-bit.
        @param[in] input Should be at most 128-bit.
        */


        template <typename T/*,
            typename = std::enable_if_t<is_uint64_v<T>>,
            typename = std::enable_if_t<is_uint32_v<T>>*/>
        T barrett_reduce_128(const T *input, const T modulus, const T* const_ratio)
        {
#ifdef XeHE_DEBUG
            if (!input)
            {
                throw std::invalid_argument("input");
            }
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
#endif
            // Reduces input using base 2^64 Barrett reduction
            // input allocation size must be 128 bits

            T tmp1, tmp2[2], tmp3 = 0, carry = 0;
            //const std::uint64_t *const_ratio = modulus.const_ratio().data();

            // Multiply input and const_ratio
            // Round 1
            multiply_uint64_hw64<T>(input[0], const_ratio[0], &carry);


            multiply_uint64<T>(input[0], const_ratio[1], tmp2);
            tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

            // Round 2
            multiply_uint64<T>(input[1], const_ratio[0], tmp2);
            carry = tmp2[1] + add_uint64<T>(tmp1, tmp2[0], &tmp1);

            // This is all we care about
            tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

            // Barrett subtraction
            tmp3 = input[0] - tmp1 * modulus;
            // One more subtraction is enough
            return static_cast<T>(tmp3) -
                   (modulus & static_cast<T>(-static_cast<T>(tmp3 >= modulus)));
        }


        /**
        Returns input mod modulus. This is not standard Barrett reduction.
        Correctness: modulus must be at most 63-bit.
        */
        template <typename T/*, typename = std::enable_if_t<is_uint64_v<T>>*/>
        T barrett_reduce_64(T input, const T modulus, const T* const_ratio)
        {
#ifdef Xe_HE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
#endif
#if 0
            return(input % modulus);
#else
            // Reduces input using base 2^64 Barrett reduction
            // floor(2^64 / mod) == floor( floor(2^128 / mod) )
            T tmp[2];
            //const std::uint64_t *const_ratio = modulus.const_ratio().data();
            multiply_uint64_hw64(input, const_ratio[1], tmp + 1);

            // Barrett subtraction
            tmp[0] = input - tmp[1] * modulus;

            // One more subtraction is enough
            return static_cast<T>(tmp[0]) -
                   (modulus &
                    static_cast<T>(-static_cast<T>(tmp[0] >= modulus)));
#endif
        }


        /**
        Returns (operand1 * operand2) mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        template <typename T>
        T multiply_uint_mod(
            T operand1, T operand2, const T modulus, const T* const_ratio)
        {
#ifdef XeHE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
#endif

#if 1
            T z[2];
            multiply_uint64<T>(operand1, operand2, z);
            return barrett_reduce_128(z, modulus, const_ratio);

#else
            XeHE_IF_CONSTEXPR(is_uint32_v<T>)
            {
                T z[2];
                multiply_uint64<T>(operand1, operand2, z);
                return barrett_reduce_128(z, modulus, const_ratio);
            }
            
            else XeHE_IF_CONSTEXPR(is_uint64_v<T>)
            {
                if ((modulus & 0xFFFC000000000000) || (operand1 & 0xFFFC000000000000) || (operand2 & 0xFFFC000000000000))
                {

                    T z[2];
                    multiply_uint64<T>(operand1, operand2, z);
                    return barrett_reduce_128(z, modulus, const_ratio);

                }

                else
                {
                    auto p = double(modulus);
                    auto u = double(1) / p;
                    auto x = double(operand1);
                    auto y = double(operand2);
                    auto h = x * y;
                    auto l = (x * y + (-h)); //fma(x, y, -h);
                    auto b = h * u;
                    auto c = double(T(b));
                    auto d = ((-c) * p + h); // fma(-c, p, h);
                    auto g = d + l;
                    g -= (g >= p) ? p : 0;
                    g += (g < 0) ? p : 0;
                    T gl = T(g);
                    return gl;
                }
            }
#endif
        }

//        /**
//        This struct contains a operand and a precomputed quotient: (operand << 64) / modulus, for a specific modulus.
//        When passed to multiply_uint_mod, a faster variant of Barrett reduction will be performed.
//        Operand must be less than modulus.
//        */
//        template<typename T>
//        struct MultiplyUIntModOperand
//        {
//            T operand;
//            T quotient;
//        };

//        template<typename T>
//        void set_quotient(const T modulus, MultiplyUIntModOperand<T>& mul_mod_operand)
        template<typename T>
        void set_quotient(const T modulus, T& mul_mod_operand_op, T& mul_mod_operand_quo)
        {
#ifdef XeHE_DEBUG
            if (mul_mod_operand_op >= modulus)
            {
                throw std::invalid_argument("input must be less than modulus");
            }
#endif
            T wide_quotient[2]{ 0, 0 };
            T wide_coeff[2]{ 0, mul_mod_operand_op };
            divide_uint128_uint64_inplace<T>(wide_coeff, modulus, wide_quotient);
            mul_mod_operand_quo = wide_quotient[0];
        }

//        template<typename T>
//        void set_operand(const T new_operand, const T modulus, MultiplyUIntModOperand<T>& mul_mod_operand)
        template<typename T>
        void set_operand(const T new_operand, const T modulus, T& mul_mod_operand_op, T& mul_mod_operand_quo)
        {

            mul_mod_operand_op = new_operand;
            set_quotient(modulus, mul_mod_operand_op, mul_mod_operand_quo);
        }

        /**
        Returns x * y mod modulus.
        This is a highly-optimized variant of Barrett reduction.
        Correctness: modulus should be at most 63-bit, and y must be less than modulus.
        */
        template <typename T>
        XeHE_NODISCARD inline T multiply_uint_mod(
            T x, const T &y_op, const T &y_quo, const T modulus)
        {
#ifdef XeHE_DEBUG
            if (y.operand >= modulus)
            {
                throw std::invalid_argument("operand y must be less than modulus");
            }
#endif
            T tmp1, tmp2;
            const T p = modulus;
            multiply_uint64_hw64(x, y_quo, &tmp1);
            tmp2 = y_op * x - tmp1 * p;
            return tmp2 - (p & static_cast<T>(-static_cast<T>(tmp2 >= p)));
        }

        /**
        Returns x * y mod modulus or x * y mod modulus + modulus.
        This is a highly-optimized variant of Barrett reduction and reduce to [0, 2 * modulus - 1].
        Correctness: modulus should be at most 63-bit, and y must be less than modulus.
        */
        template <typename T>
        XeHE_NODISCARD inline T multiply_uint_mod_lazy(
            T x, const T &y_op, const T &y_quo, const T modulus)
        {
#ifdef XeHE_DEBUG
            if (y.operand >= modulus)
            {
                throw std::invalid_argument("operand y must be less than modulus");
            }
#endif
            T tmp1;
            const T p = modulus;
            multiply_uint64_hw64(x, y_quo, &tmp1);
            return y_op * x - tmp1 * p;
        }

        /**
        Returns value[0] = value mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        template <typename T>
        inline void modulo_uint_inplace(T *value, std::size_t value_uint64_count, const T modulus, const T* const_ratio)
        {
#ifdef XeHE_DEBUG
            if (!value)
            {
                throw std::invalid_argument("value");
            }
            if (!value_uint64_count)
            {
                throw std::invalid_argument("value_uint64_count");
            }
#endif

            if (value_uint64_count == 1)
            {
                if (*value < modulus)
                {
                    return;
                }
                else
                {
                    *value = barrett_reduce_64(*value, modulus, const_ratio);
                }
            }

            // Starting from the top, reduce always 128-bit blocks
            for (std::size_t i = value_uint64_count - 1; i--;)
            {
                value[i] = barrett_reduce_128(value + i, modulus, const_ratio);
                value[i + 1] = 0;
            }
        }

        /**
        Returns value mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        template <typename T>
        XeHE_NODISCARD inline T modulo_uint(
            const T *value, std::size_t value_uint64_count, const T modulus, const T* const_ratio)
        {
#ifdef XeHE_DEBUG
            if (!value && value_uint64_count)
            {
                throw std::invalid_argument("value");
            }
            if (!value_uint64_count)
            {
                throw std::invalid_argument("value_uint64_count");
            }
#endif
            if (value_uint64_count == 1)
            {
                // If value < modulus no operation is needed
                if (*value < modulus)
                    return *value;
                else
                    return barrett_reduce_64(*value, modulus, const_ratio);
            }

            // Temporary space for 128-bit reductions
            T temp[2]{ 0, value[value_uint64_count - 1] };
            for (size_t k = value_uint64_count - 1; k--;)
            {
                temp[0] = value[k];
                temp[1] = barrett_reduce_128(temp, modulus, const_ratio);
            }

            // Save the result modulo i-th prime
            return temp[1];
        }

        /**
        Returns (operand1 * operand2) + operand3 mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        template <typename T>
        inline T multiply_add_uint_mod(
            T operand1, T operand2, T operand3, const T modulus, const T* const_ratio)
        {
            // Lazy reduction
            T temp[2];
            multiply_uint64(operand1, operand2, temp);
            temp[1] += add_uint64(temp[0], operand3, temp);
            return barrett_reduce_128(temp, modulus, const_ratio);
        }

        /**
        Returns (operand1 * operand2) + operand3 mod modulus.
        Correctness: Follows the condition of multiply_uint_mod.
        */
        template <typename T>
        inline T multiply_add_uint_mod(
            T operand1, T & operand2_op, T & operand2_quo, T operand3, const T modulus, const T* const_ratio)
        {
            return add_uint_mod(
                multiply_uint_mod(operand1, operand2_op, operand2_quo, modulus), barrett_reduce_64(operand3, modulus, const_ratio), modulus);
        }


#if 0
        inline bool try_invert_uint_mod(std::uint64_t operand, const Modulus &modulus, std::uint64_t &result)
        {
            return try_invert_uint_mod(operand, modulus.value(), result);
        }

        /**
        Returns operand^exponent mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        SEAL_NODISCARD std::uint64_t exponentiate_uint_mod(
            std::uint64_t operand, std::uint64_t exponent, const Modulus &modulus);

        /**
        Computes numerator = numerator mod modulus, quotient = numerator / modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        void divide_uint_mod_inplace(
            std::uint64_t *numerator, const Modulus &modulus, std::size_t uint64_count, std::uint64_t *quotient,
            MemoryPool &pool);

        /**
        Computes <opearnd1, operand2> mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        SEAL_NODISCARD std::uint64_t dot_product_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t count, const Modulus &modulus);
#endif
    } // namespace util
} // namespace xehe
