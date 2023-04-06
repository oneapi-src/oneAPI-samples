/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_UINTARITH_CORE_HPP
#define XeHE_UINTARITH_CORE_HPP

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <type_traits>

// XeHE
#include "xe_uintarith.hpp"


namespace xehe
{
    namespace native
    {
#if 0
        template<typename T>
        inline T add_uint(
            const T* operand1, const T* operand2,
            std::size_t uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!operand1)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // Unroll first iteration of loop. We assume uint64_count > 0.
            auto carry = xehe::native::add_uint<T>(*operand1++, *operand2++, 0, result++);

            // Do the rest
            for (; --uint64_count; operand1++, operand2++, result++)
            {
                T temp_result;
                carry = xehe::native::add_uint<T>(*operand1, *operand2, carry, &temp_result);
                *result = temp_result;
            }
            return carry;
        }

        template<typename T>
        inline T add_uint(
            const T* operand1, T operand2,
            std::size_t uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!operand1)
            {
                throw std::invalid_argument("operand1");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // Unroll first iteration of loop. We assume uint64_count > 0.
            auto carry = xehe::native::add_uint<T>(*operand1++, operand2, 0, result++);

            // Do the rest
            for (; --uint64_count; operand1++, result++)
            {
                T temp_result;
                carry = xehe::native::add_uint<T>(*operand1, std::uint64_t(0), carry, &temp_result);
                *result = temp_result;
            }
            return carry;
        }

        template <typename T>
        inline T add_uint(const T* operand1, const T* operand2, T* result)
        {
#ifdef XeHE_DEBUG
            if (!operand1)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            auto carry = xehe::native::add_uint(operand1[0], operand2[0], 0, result);
            auto ret = xehe::native::add_uint(operand1[1], operand2[1], carry, result + 1);
            return(ret);
        }

        template<typename T>
        inline T sub_uint(
            const T* operand1, std::size_t operand1_uint64_count,
            const T* operand2, std::size_t operand2_uint64_count,
            T borrow,
            std::size_t result_uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!result_uint64_count)
            {
                throw std::invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            for (std::size_t i = 0; i < result_uint64_count;
                i++, operand1++, operand2++, result++)
            {
                T temp_result;
                borrow = xehe::native::sub_uint<T>((i < operand1_uint64_count) ? *operand1 : 0,
                    (i < operand2_uint64_count) ? *operand2 : 0, borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }

        template<typename T>
        inline T sub_uint(
            const T* operand1, const T* operand2,
            std::size_t uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!operand1)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // Unroll first iteration of loop. We assume uint64_count > 0.
            auto borrow = xehe::native::sub_uint<T>(*operand1++, *operand2++, 0, result++);

            // Do the rest
            for (; --uint64_count; operand1++, operand2++, result++)
            {
                T temp_result;
                borrow = xehe::native::sub_uint<T>(*operand1, *operand2, borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }

        template<typename T>
        inline T sub_uint(
            const T* operand1, T operand2,
            std::size_t uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!operand1)
            {
                throw std::invalid_argument("operand1");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // Unroll first iteration of loop. We assume uint64_count > 0.
            auto borrow = xehe::native::sub_uint<T>(*operand1++, operand2, result++);

            // Do the rest
            for (; --uint64_count; operand1++, operand2++, result++)
            {
                T temp_result;
                borrow = xehe::native::sub_uint<T>(*operand1, std::uint64_t(0), borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }


        template<typename T>
        inline void left_shift2(
            const T *operand, int shift_amount, T *result)
        {
            const int bits_per_uint = int(xehe::native::bits_per_uint<T>());

#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 2 * bits_per_uint))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const auto shift_amount_sz = shift_amount;

            // Early return
            if (shift_amount_sz & bits_per_uint)
            {
                result[1] = operand[0];
                result[0] = 0;
            }
            else
            {
                result[1] = operand[1];
                result[0] = operand[0];
            }

            // How many bits to shift in addition to word shift
            auto bit_shift_amount = shift_amount_sz & (bits_per_uint - 1);

            // Do we have a word shift
            if (bit_shift_amount)
            {
                auto neg_bit_shift_amount = bits_per_uint - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[1] = xehe::native::left_shift<T>(result[1], bit_shift_amount) |
                    xehe::native::right_shift<T>(result[0], neg_bit_shift_amount);
                result[0] = xehe::native::left_shift<T>(result[0], bit_shift_amount);
            }
        }




        template<typename T>
        inline void right_shift2(
            const T *operand, int shift_amount, T *result)
        {


            const int bits_per_uint = int(xehe::native::bits_per_uint<T>());
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 2 * bits_per_uint))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const int shift_amount_sz = int(shift_amount);

            if (shift_amount_sz & bits_per_uint)
            {
                result[0] = operand[1];
                result[1] = 0;
            }
            else
            {
                result[1] = operand[1];
                result[0] = operand[0];
            }

            // How many bits to shift in addition to word shift
            auto bit_shift_amount = shift_amount_sz & (bits_per_uint - 1);

            if (bit_shift_amount)
            {
                auto neg_bit_shift_amount = bits_per_uint - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[0] = xehe::native::right_shift<T>(result[0], bit_shift_amount) |
                    xehe::native::left_shift<T>(result[1], neg_bit_shift_amount);
                result[1] = xehe::native::right_shift<T>(result[1], bit_shift_amount);
            }
        }
#endif
        template<typename T>
        inline void left_shift3(
            const T *operand, int shift_amount, T *result)
        {
            const auto bits_per_uint = int(xehe::native::bits_per_uint<T>());
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 3 * bits_per_uint))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const auto shift_amount_sz = shift_amount;

            if (shift_amount_sz & (bits_per_uint << 1))
            {
                result[2] = operand[0];
                result[1] = 0;
                result[0] = 0;
            }
            else if (shift_amount_sz & bits_per_uint)
            {
                result[2] = operand[1];
                result[1] = operand[0];
                result[0] = 0;
            }
            else
            {
                result[2] = operand[2];
                result[1] = operand[1];
                result[0] = operand[0];
            }

            // How many bits to shift in addition to word shift
            auto bit_shift_amount = shift_amount_sz & (bits_per_uint - 1);

            if (bit_shift_amount)
            {
                auto neg_bit_shift_amount = bits_per_uint - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[2] = xehe::native::left_shift<T>(result[2], bit_shift_amount) |
                    xehe::native::right_shift<T>(result[1], neg_bit_shift_amount);
                result[1] = xehe::native::left_shift<T>(result[1], bit_shift_amount) |
                    xehe::native::right_shift<T>(result[0], neg_bit_shift_amount);
                result[0] = xehe::native::left_shift<T>(result[0], bit_shift_amount);
            }
        }

        template<typename T>
        inline void right_shift3(
            const T *operand, int shift_amount, T *result)
        {
            const auto bits_per_uint = int(xehe::native::bits_per_uint<T>());
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 3 * bits_per_uint))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const auto shift_amount_sz = shift_amount;

            if (shift_amount_sz & (bits_per_uint << 1))
            {
                result[0] = operand[2];
                result[1] = 0;
                result[2] = 0;
            }
            else if (shift_amount_sz & bits_per_uint)
            {
                result[0] = operand[1];
                result[1] = operand[2];
                result[2] = 0;
            }
            else
            {
                result[2] = operand[2];
                result[1] = operand[1];
                result[0] = operand[0];
            }

            // How many bits to shift in addition to word shift
            auto bit_shift_amount = shift_amount_sz & (bits_per_uint - 1);

            if (bit_shift_amount)
            {
                auto neg_bit_shift_amount = bits_per_uint - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[0] = xehe::native::right_shift<T>(result[0], bit_shift_amount) |
                    xehe::native::left_shift<T>(result[1], neg_bit_shift_amount);
                result[1] = xehe::native::right_shift<T>(result[1], bit_shift_amount) |
                    xehe::native::left_shift<T>(result[2], neg_bit_shift_amount);
                result[2] = xehe::native::right_shift<T>(result[2], bit_shift_amount);
            }
        }

#if 0
        template<typename T>
        inline void half_round_up_uint(
            const T *operand, std::size_t uint64_count,
            T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand && uint64_count > 0)
            {
                throw std::invalid_argument("operand");
            }
            if (!result && uint64_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            if (!uint64_count)
            {
                return;
            }
            // Set result to (operand + 1) / 2. To prevent overflowing operand, right shift
            // and then increment result if low-bit of operand was set.
            bool low_bit_set = operand[0] & 1;
            int bits_per_uint = xehe::native::bits_per_uint<T>();

            for (std::size_t i = 0; i < uint64_count - 1; i++)
            {
                result[i] = (operand[i] >> 1) | (operand[i + 1] << (bits_per_uint - 1));
            }
            result[uint64_count - 1] = operand[uint64_count - 1] >> 1;

            if (low_bit_set)
            {
                increment_uint(result, uint64_count, result);
            }
        }

#endif

        template<typename T>
        void div_uint2(T* numerator,
            T denominator, T* quotient)
        {
#ifdef XeHE_DEBUG
            if (!numerator)
            {
                throw std::invalid_argument("numerator");
            }
            if (denominator == 0)
            {
                throw std::invalid_argument("denominator");
            }
            if (!quotient)
            {
                throw std::invalid_argument("quotient");
            }
            if (numerator == quotient)
            {
                throw std::invalid_argument("quotient cannot point to same value as numerator");
            }
#endif
            int bits_per_uint = xehe::native::bits_per_uint<T>();
            // We expect 128-bit input
            constexpr size_t uint_count = 2;

            // Clear quotient. Set it to zero.
            quotient[0] = 0;
            quotient[1] = 0;

            // Determine significant bits in numerator and denominator.
            int numerator_bits = xehe::native::get_significant_bit_count_uint<T>(numerator, uint_count);
            int denominator_bits = xehe::native::get_significant_bit_count<T>(denominator);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Create temporary space to store mutable copy of denominator.
            T shifted_denominator[uint_count]{ denominator, 0 };

            // Create temporary space to store difference calculation.
            T difference[uint_count]{ 0, 0 };

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;

            xehe::native::left_shift2<T>(shifted_denominator, denominator_shift, shifted_denominator);
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (xehe::native::sub_uint<T>(numerator, shifted_denominator, uint_count, difference))
                {
                    // numerator < shifted_denominator and MSBs are aligned,
                    // so current quotient bit is zero and next one is definitely one.
                    if (remaining_shifts == 0)
                    {
                        // No shifts remain and numerator < denominator so done.
                        break;
                    }

                    // Effectively shift numerator left by 1 by instead adding
                    // numerator to difference (to prevent overflow in numerator).
                    xehe::native::add_uint<T>(difference, numerator, uint_count, difference);

                    // Adjust quotient and remaining shifts as a result of shifting numerator.
                    quotient[1] = (quotient[1] << 1) | (quotient[0] >> (bits_per_uint - 1));
                    quotient[0] <<= 1;
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Determine amount to shift numerator to bring MSB in alignment
                // with denominator.
                numerator_bits = xehe::native::get_significant_bit_count_uint<T>(difference, uint_count);

                // Clip the maximum shift to determine only the integer
                // (as opposed to fractional) bits.
                int numerator_shift = std::min(denominator_bits - numerator_bits, remaining_shifts);

                // Shift and update numerator.
                // This may be faster; first set to zero and then update if needed

                // Difference is zero so no need to shift, just set to zero.
                numerator[0] = 0;
                numerator[1] = 0;

                if (numerator_bits > 0)
                {
                    xehe::native::left_shift2<T>(difference, numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                xehe::native::left_shift2<T>(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                xehe::native::right_shift2<T>(numerator, denominator_shift, numerator);
            }
        }



        template<typename T>
        void div_uint3(T* numerator, T denominator, T* quotient)
        {
#ifdef XeHE_DEBUG
            if (!numerator)
            {
                throw invalid_argument("numerator");
            }
            if (denominator == 0)
            {
                throw invalid_argument("denominator");
            }
            if (!quotient)
            {
                throw invalid_argument("quotient");
            }
            if (numerator == quotient)
            {
                throw invalid_argument("quotient cannot point to same value as numerator");
            }
#endif
            // We expect 192-bit input
            size_t uint_count = 3;

            // Clear quotient. Set it to zero.
            quotient[0] = 0;
            quotient[1] = 0;
            quotient[2] = 0;

            // Determine significant bits in numerator and denominator.
            int numerator_bits = xehe::native::get_significant_bit_count_uint<T>(numerator, uint_count);
            int denominator_bits = xehe::native::get_significant_bit_count<T>(denominator);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Only perform computation up to last non-zero uints.
            int bits_per_uint = xehe::native::bits_per_uint<T>();
            uint_count = size_t(xehe::native::divide_round_up(numerator_bits, bits_per_uint));

            // Handle fast case.
            if (uint_count == 1)
            {
                *quotient = *numerator / denominator;
                *numerator -= *quotient * denominator;
                return;
            }
            uint_count = std::max(size_t(3), uint_count);
            // Create temporary space to store mutable copy of denominator.
            std::vector<T> shifted_denominator(uint_count, 0);
            shifted_denominator[0] = denominator;

            // Create temporary space to store difference calculation.
            std::vector<T> difference(uint_count);

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;

            xehe::native::left_shift3<T>(shifted_denominator.data(), denominator_shift,
                shifted_denominator.data());
            denominator_bits += denominator_shift;
            
            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (xehe::native::sub_uint<T>(numerator, shifted_denominator.data(),
                    uint_count, difference.data()))
                {
                    // numerator < shifted_denominator and MSBs are aligned,
                    // so current quotient bit is zero and next one is definitely one.
                    if (remaining_shifts == 0)
                    {
                        // No shifts remain and numerator < denominator so done.
                        break;
                    }

                    // Effectively shift numerator left by 1 by instead adding
                    // numerator to difference (to prevent overflow in numerator).
                    xehe::native::add_uint<T>(difference.data(), numerator, uint_count, difference.data());

                    // Adjust quotient and remaining shifts as a result of shifting numerator.
                    xehe::native::left_shift3<T>(quotient, 1, quotient);
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Determine amount to shift numerator to bring MSB in alignment with denominator.
                numerator_bits = xehe::native::get_significant_bit_count_uint<T>(difference.data(), uint_count);
                int numerator_shift = denominator_bits - numerator_bits;
                if (numerator_shift > remaining_shifts)
                {
                    // Clip the maximum shift to determine only the integer
                    // (as opposed to fractional) bits.
                    numerator_shift = remaining_shifts;
                }

                // Shift and update numerator.
                if (numerator_bits > 0)
                {
                    xehe::native::left_shift3<T>(difference.data(), numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                    // Difference is zero so no need to shift, just set to zero.
                    xehe::native::set_zero_uint<T>(uint_count, numerator);
                }

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                xehe::native::left_shift3<T>(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                xehe::native::right_shift3<T>(numerator, denominator_shift, numerator);
            }
        }

        /*
        UTILITIES
        */
        template<typename T>
        T mod_reduce_generic2(T* in, T mod)
        {
            // generic reduction
            T in3[3]{ in[0], in[1], 0 };
            T quo2[2];

            // divide q = in/mod
            xehe::native::div_uint2<T>(in3, mod, quo2);

            auto ret = in3[0];

            return(ret);
        }

        template<typename T>
        T op_by_mod_inverse(T op, T mod)
        {
            // op*2^BitCount(T)/modulus
            T num[2]{ 0, op };
            T quo2[2];
            xehe::native::div_uint2<T>(num, mod, quo2);
            return(quo2[0]);
        }

        template<typename T>
        T mod_inverse1(T mod)
        {
            // 2^BitCount(T)/modulus
            T num[2]{ 0, 1 };
            T quo2[2];
            xehe::native::div_uint2<T>(num, mod, quo2);
            return(quo2[0]);
        }


        template<typename T>
        T mod_inverse2(T mod, T* high)
        {
            // 2^(2*BitCount(T))/modulus
            T quo3[3];
            T pow3[3]{ 0, 0, 1 }; 
            xehe::native::div_uint3<T>(pow3, mod, quo3);
            T ret = quo3[0];
            *high = quo3[1];
            return(ret);
        }

        /**
        Returns operand^exponent mod modulus.
        Correctness: Follows the condition of barret_reduce_128.
        */
        template <typename T>
        T exp_uint_mod(
            T operand, T exponent, T modulus, const T *const_ratio){
#ifdef XeHE_DEBUG
                if (!modulus)
                {
                    throw std::invalid_argument("modulus");
                }
                if (operand >= modulus)
                {
                    throw std::invalid_argument("operand");
                }
    #endif
                // Fast cases
                if (!exponent)
                {
                    // Result is supposed to be only one digit
                    return 1;
                }

                if (exponent == 1)
                {
                    return operand;
                }

                // Perform binary exponentiation.
                T power = operand;
                T product = 0;
                T intermediate = 1;

                // Initially: power = operand and intermediate = 1, product is irrelevant.
                while (true)
                {
                    if (exponent & 1)
                    {
                        product = xehe::native::mul_mod(power, intermediate, modulus, const_ratio);
                        std::swap<T>(product, intermediate);
                    }
                    exponent >>= 1;
                    if (exponent == 0)
                    {
                        break;
                    }
                    product = xehe::native::mul_mod(power, power, modulus, const_ratio);
                    std::swap<T>(product, power);
                }
                return intermediate;
            }

            
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
            xehe::native::div_uint2<T>(wide_coeff, modulus, wide_quotient);
            mul_mod_operand_quo = wide_quotient[0];
        }

        template<typename T>
        void set_operand(const T new_operand, const T modulus, T& mul_mod_operand_op, T& mul_mod_operand_quo)
        {

            mul_mod_operand_op = new_operand;
            xehe::native::set_quotient<T>(modulus, mul_mod_operand_op, mul_mod_operand_quo);
        }


    }
}

#endif
