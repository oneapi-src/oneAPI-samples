/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#include "util/xe_uintcore.h"
#include "util/xe_uintarith.h"
#include "util/common.h"
#include <algorithm>
#include <functional>
#include <array>

using namespace std;

namespace xehe
{
    namespace util
    {




        template<typename T>
        void divide_uint192_uint64_inplace(T *numerator,
           T denominator, T *quotient)
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
            size_t uint64_count = 3;

            // Clear quotient. Set it to zero.
            quotient[0] = 0;
            quotient[1] = 0;
            quotient[2] = 0;

            // Determine significant bits in numerator and denominator.
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Only perform computation up to last non-zero uint64s.
            uint64_count = safe_cast<size_t>(
                divide_round_up(numerator_bits, bits_per_uint64));

            // Handle fast case.
            if (uint64_count == 1)
            {
                *quotient = *numerator / denominator;
                *numerator -= *quotient * denominator;
                return;
            }

            // Create temporary space to store mutable copy of denominator.
            vector<uint64_t> shifted_denominator(uint64_count, 0);
            shifted_denominator[0] = denominator;

            // Create temporary space to store difference calculation.
            vector<uint64_t> difference(uint64_count);

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;

            left_shift_uint192(shifted_denominator.data(), denominator_shift,
                shifted_denominator.data());
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (sub_uint_uint(numerator, shifted_denominator.data(),
                    uint64_count, difference.data()))
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
                    add_uint_uint(difference.data(), numerator, uint64_count, difference.data());

                    // Adjust quotient and remaining shifts as a result of shifting numerator.
                    left_shift_uint192(quotient, 1, quotient);
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Determine amount to shift numerator to bring MSB in alignment with denominator.
                numerator_bits = get_significant_bit_count_uint(difference.data(), uint64_count);
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
                    left_shift_uint192(difference.data(), numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                    // Difference is zero so no need to shift, just set to zero.
                    set_zero_uint(uint64_count, numerator);
                }

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint192(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                right_shift_uint192(numerator, denominator_shift, numerator);
            }
        }

        template<typename T>
        void exponentiate_uint(const T *operand,
            size_t operand_uint64_count, const T *exponent,
            size_t exponent_uint64_count, size_t result_uint64_count,
            T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw invalid_argument("operand");
            }
            if (!operand_uint64_count)
            {
                throw invalid_argument("operand_uint64_count");
            }
            if (!exponent)
            {
                throw invalid_argument("exponent");
            }
            if (!exponent_uint64_count)
            {
                throw invalid_argument("exponent_uint64_count");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (!result_uint64_count)
            {
                throw invalid_argument("result_uint64_count");
            }
#endif
            // Fast cases
            if (is_zero_uint(exponent, exponent_uint64_count))
            {
                set_uint(1, result_uint64_count, result);
                return;
            }
            if (is_equal_uint(exponent, exponent_uint64_count, 1))
            {
                set_uint_uint(operand, operand_uint64_count, result_uint64_count, result);
                return;
            }

            // Need to make a copy of exponent
            auto exponent_copy = allocate_uint<T>(exponent_uint64_count);
            set_uint_uint(exponent, exponent_uint64_count, exponent_copy.get());

            // Perform binary exponentiation.
            auto big_alloc = allocate_uint<T>(
                result_uint64_count + result_uint64_count + result_uint64_count);

            auto powerptr = big_alloc.get();
            auto productptr = powerptr + result_uint64_count;
            auto intermediateptr = productptr + result_uint64_count;

            set_uint_uint(operand, operand_uint64_count, result_uint64_count, powerptr);
            set_uint(1, result_uint64_count, intermediateptr);

            // Initially: power = operand and intermediate = 1, product is not initialized.
            while (true)
            {
                if ((*exponent_copy.get() % 2) == 1)
                {
                    multiply_truncate_uint_uint(powerptr, intermediateptr,
                        result_uint64_count, productptr);
                    swap(productptr, intermediateptr);
                }
                right_shift_uint(exponent_copy.get(), 1, exponent_uint64_count,
                    exponent_copy.get());
                if (is_zero_uint(exponent_copy.get(), exponent_uint64_count))
                {
                    break;
                }
                multiply_truncate_uint_uint(powerptr, powerptr, result_uint64_count,
                    productptr);
                swap(productptr, powerptr);
            }
            set_uint_uint(intermediateptr, result_uint64_count, result);
        }

        template<typename T>
        T exponentiate_uint64_safe(T operand, T exponent)
        {
            // Fast cases
            if (exponent == 0)
            {
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

            // Initially: power = operand and intermediate = 1, product irrelevant.
            while (true)
            {
                if (exponent & 1)
                {
                    product = mul_safe(power, intermediate);
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = mul_safe(power, power);
                swap(product, power);
            }

            return intermediate;
        }

        template<typename T>
        T exponentiate_uint64(T operand, T exponent)
        {
            // Fast cases
            if (exponent == 0)
            {
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

            // Initially: power = operand and intermediate = 1, product irrelevant.
            while (true)
            {
                if (exponent & 1)
                {
                    product = power * intermediate;
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = power * power;
                swap(product, power);
            }

            return intermediate;
        }

    }
}
