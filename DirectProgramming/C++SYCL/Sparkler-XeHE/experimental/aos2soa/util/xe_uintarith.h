/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_UINTARITH_H
#define XeHE_UINTARITH_H

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <type_traits>
#include "util/common.h"
#include "util/xe_uintcore.h"

namespace xehe
{
    namespace util
    {
        template<typename T>
        void multiply_uint_uint64(
            const T* operand1, std::size_t operand1_uint64_count,
            T operand2, std::size_t result_uint64_count,
            T* result);

        template<typename T, typename S, typename = std::enable_if<is_uint64_v<T, S>>, typename = std::enable_if<is_uint32_v<T, S>>>
        XeHE_NODISCARD inline unsigned char add_uint64_generic(
            T operand1, S operand2, unsigned char carry,
            T *result)
        {
#ifdef XeHE_DEBUG
            if (!result)
            {
                throw std::invalid_argument("result cannot be null");
            }
#endif
            operand1 += operand2;
            *result = operand1 + carry;
            return (operand1 < operand2) || (~operand1 < carry);
        }


        template<typename T, typename S, typename = std::enable_if<is_uint64_v<T, S>>, typename = std::enable_if<is_uint32_v<T, S>>>
        XeHE_NODISCARD inline unsigned char add_uint64(
            T operand1, S operand2, unsigned char carry,
            T *result)
        {
            return XeHE_ADD_CARRY_UINT64(operand1, operand2, carry, result);
        }


        template<typename T, typename S, typename R,
            typename = std::enable_if<is_uint64_v<T, S, R>>,
            typename = std::enable_if<is_uint32_v<T, S, R>>>
        XeHE_NODISCARD inline unsigned char add_uint64(
            T operand1, S operand2, R *result)
        {
            *result = operand1 + operand2;
            return static_cast<unsigned char>(*result < operand1);
        }

 
        template<typename T>
        inline unsigned char add_uint_uint(
            const T *operand1, std::size_t operand1_uint64_count,
            const T *operand2, std::size_t operand2_uint64_count,
            unsigned char carry,
            std::size_t result_uint64_count, T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand1_uint64_count)
            {
                throw std::invalid_argument("operand1_uint64_count");
            }
            if (!operand2_uint64_count)
            {
                throw std::invalid_argument("operand2_uint64_count");
            }
            if (!result_uint64_count)
            {
                throw std::invalid_argument("result_uint64_count");
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
            for (std::size_t i = 0; i < result_uint64_count; i++)
            {
                T temp_result;
                carry = add_uint64(
                    (i < operand1_uint64_count) ? *operand1++ : 0,
                    (i < operand2_uint64_count) ? *operand2++ : 0,
                    carry, &temp_result);
                *result++ = temp_result;
            }
            return carry;
        }


        template<typename T>
        inline unsigned char add_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
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
            unsigned char carry = add_uint64(*operand1++, *operand2++, result++);

            // Do the rest
            for(; --uint64_count; operand1++, operand2++, result++)
            {
                T temp_result;
                carry = add_uint64(*operand1, *operand2, carry, &temp_result);
                *result = temp_result;
            }
            return carry;
        }

        template<typename T>
        inline unsigned char add_uint_uint64(
            const T *operand1, T operand2,
            std::size_t uint64_count, T *result)
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
            unsigned char carry = add_uint64(*operand1++, operand2, result++);

            // Do the rest
            for(; --uint64_count; operand1++, result++)
            {
                T temp_result;
                carry = add_uint64(*operand1, std::uint64_t(0), carry, &temp_result);
                *result = temp_result;
            }
            return carry;
        }

        template <typename T>
        inline unsigned char add_uint128(const T* operand1, const T* operand2, T* result)
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
            unsigned char carry = add_uint64(operand1[0], operand2[0], result);
            return add_uint64(operand1[1], operand2[1], carry, result + 1);
        }



        template<typename T, typename S,
            typename = std::enable_if<is_uint64_v<T, S>>,
            typename = std::enable_if<is_uint32_v<T, S>>>
        XeHE_NODISCARD inline unsigned char sub_uint64_generic(
            T operand1, S operand2,
            unsigned char borrow, T *result)
        {
#ifdef XeHE_DEBUG
            if (!result)
            {
                throw std::invalid_argument("result cannot be null");
            }
#endif
            auto diff = operand1 - operand2;
            *result = diff - (borrow != 0);
            return (diff > operand1) || (diff < borrow);
        }

        template<typename T, typename S,
            typename = std::enable_if<is_uint64_v<T, S>>,
            typename = std::enable_if<is_uint32_v<T, S>>>
        XeHE_NODISCARD inline unsigned char sub_uint64(
            T operand1, S operand2,
            unsigned char borrow, T *result)
        {
            return XeHE_SUB_BORROW_UINT64(operand1, operand2, borrow, result);
        }

        template<typename T, typename S, typename R,
            typename = std::enable_if<is_uint64_v<T, S, R>>,
            typename = std::enable_if<is_uint32_v<T, S, R>>>
        XeHE_NODISCARD inline unsigned char sub_uint64(
            T operand1, S operand2, R *result)
        {
            *result = operand1 - operand2;
            return static_cast<unsigned char>(operand2 > operand1);
        }

        template<typename T>
        inline unsigned char sub_uint_uint(
            const T *operand1, std::size_t operand1_uint64_count,
            const T *operand2, std::size_t operand2_uint64_count,
            unsigned char borrow,
            std::size_t result_uint64_count, T *result)
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
                borrow = sub_uint64((i < operand1_uint64_count) ? *operand1 : 0,
                    (i < operand2_uint64_count) ? *operand2 : 0, borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }

        template<typename T>
        inline unsigned char sub_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
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
            unsigned char borrow = sub_uint64(*operand1++, *operand2++, result++);

            // Do the rest
            for(; --uint64_count; operand1++, operand2++, result++)
            {
                T temp_result;
                borrow = sub_uint64(*operand1, *operand2, borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }

        template<typename T>
        inline unsigned char sub_uint_uint64(
            const T *operand1, T operand2,
            std::size_t uint64_count, T *result)
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
            unsigned char borrow = sub_uint64(*operand1++, operand2, result++);

            // Do the rest
            for(; --uint64_count; operand1++, operand2++, result++)
            {
                T temp_result;
                borrow = sub_uint64(*operand1, std::uint64_t(0), borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }

        template<typename T>
        inline unsigned char increment_uint(
            const T *operand, std::size_t uint64_count,
            T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            return add_uint_uint64(operand, static_cast <T>(1), uint64_count, result);
        }

        template<typename T>
        inline unsigned char decrement_uint(
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
            return sub_uint_uint64(operand, 1, uint64_count, result);
        }

        template<typename T>
        inline void negate_uint(
            const T *operand, std::size_t uint64_count,
            T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // Negation is equivalent to inverting bits and adding 1.
            unsigned char carry = add_uint64(~*operand++, std::uint64_t(1), result++);
            for(; --uint64_count; operand++, result++)
            {
                T temp_result;
                carry = add_uint64(
                    ~*operand, std::uint64_t(0), carry, &temp_result);
                *result = temp_result;
            }
        }

        template<typename T>
        inline void left_shift_uint(const T *operand,
            int shift_amount, std::size_t uint64_count, T *result)
        {
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

            const std::size_t bits_per_uint64_sz =
                static_cast<std::size_t>(bits_per_uint);
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount,
                    mul_safe(uint64_count, bits_per_uint64_sz)))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // How many words to shift
            std::size_t uint64_shift_amount =
                static_cast<std::size_t>(shift_amount) / bits_per_uint64_sz;

            // Shift words
            for (std::size_t i = 0; i < uint64_count - uint64_shift_amount; i++)
            {
                result[uint64_count - i - 1] = operand[uint64_count - i - 1 - uint64_shift_amount];
            }
            for (std::size_t i = uint64_count - uint64_shift_amount; i < uint64_count; i++)
            {
                result[uint64_count - i - 1] = 0;
            }

            // How many bits to shift in addition
            std::size_t bit_shift_amount = static_cast<std::size_t>(shift_amount)
                - (uint64_shift_amount * bits_per_uint64_sz);

            if (bit_shift_amount)
            {
                std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

                for (std::size_t i = uint64_count - 1; i > 0; i--)
                {
                    result[i] = (result[i] << bit_shift_amount) |
                        (result[i - 1] >> neg_bit_shift_amount);
                }
                result[0] = result[0] << bit_shift_amount;
            }
        }

        template<typename T>
        inline void right_shift_uint(const T *operand,
            int shift_amount, std::size_t uint64_count,T *result)
        {
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

            const std::size_t bits_per_uint64_sz =
                static_cast<std::size_t>(bits_per_uint);
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount,
                    mul_safe(uint64_count, bits_per_uint64_sz)))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            // How many words to shift
            std::size_t uint64_shift_amount =
                static_cast<std::size_t>(shift_amount) / bits_per_uint64_sz;

            // Shift words
            for (std::size_t i = 0; i < uint64_count - uint64_shift_amount; i++)
            {
                result[i] = operand[i + uint64_shift_amount];
            }
            for (std::size_t i = uint64_count - uint64_shift_amount; i < uint64_count; i++)
            {
                result[i] = 0;
            }

            // How many bits to shift in addition
            std::size_t bit_shift_amount = static_cast<std::size_t>(shift_amount)
                - (uint64_shift_amount * bits_per_uint64_sz);

            if (bit_shift_amount)
            {
                std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

                for (std::size_t i = 0; i < uint64_count - 1; i++)
                {
                    result[i] = (result[i] >> bit_shift_amount) |
                        (result[i + 1] << neg_bit_shift_amount);
                }
                result[uint64_count - 1] = result[uint64_count - 1] >> bit_shift_amount;
            }
        }

        template<typename T>
        inline void left_shift_uint128(
            const T *operand, int shift_amount, T *result)
        {
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

            const std::size_t bits_per_uint64_sz =
                static_cast<std::size_t>(bits_per_uint);
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 2 * bits_per_uint64_sz))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const std::size_t shift_amount_sz =
                static_cast<std::size_t>(shift_amount);

            // Early return
            if (shift_amount_sz & bits_per_uint64_sz)
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
            std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

            // Do we have a word shift
            if (bit_shift_amount)
            {
                std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[1] = (result[1] << bit_shift_amount) |
                    (result[0] >> neg_bit_shift_amount);
                result[0] = result[0] << bit_shift_amount;
            }
        }

        template<typename T>
        inline void right_shift_uint128(
            const T *operand, int shift_amount, T *result)
        {
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

            const std::size_t bits_per_uint64_sz =
                static_cast<std::size_t>(bits_per_uint);
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 2 * bits_per_uint64_sz))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const std::size_t shift_amount_sz =
                static_cast<std::size_t>(shift_amount);

            if (shift_amount_sz & bits_per_uint64_sz)
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
            std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

            if (bit_shift_amount)
            {
                std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[0] = (result[0] >> bit_shift_amount) |
                    (result[1] << neg_bit_shift_amount);
                result[1] = result[1] >> bit_shift_amount;
            }
        }

        template<typename T>
        inline void left_shift_uint192(
            const T *operand, int shift_amount, T *result)
        {
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

            const std::size_t bits_per_uint64_sz =
                static_cast<std::size_t>(bits_per_uint);
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 3 * bits_per_uint64_sz))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const std::size_t shift_amount_sz =
                static_cast<std::size_t>(shift_amount);

            if (shift_amount_sz & (bits_per_uint64_sz << 1))
            {
                result[2] = operand[0];
                result[1] = 0;
                result[0] = 0;
            }
            else if (shift_amount_sz & bits_per_uint64_sz)
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
            std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

            if (bit_shift_amount)
            {
                std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[2] = (result[2] << bit_shift_amount) |
                    (result[1] >> neg_bit_shift_amount);
                result[1] = (result[1] << bit_shift_amount) |
                    (result[0] >> neg_bit_shift_amount);
                result[0] = result[0] << bit_shift_amount;
            }
        }

        template<typename T>
        inline void right_shift_uint192(
            const T *operand, int shift_amount, T *result)
        {
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

            const std::size_t bits_per_uint64_sz =
                static_cast<std::size_t>(bits_per_uint);
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                unsigned_geq(shift_amount, 3 * bits_per_uint64_sz))
            {
                throw std::invalid_argument("shift_amount");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
#endif
            const std::size_t shift_amount_sz =
                static_cast<std::size_t>(shift_amount);

            if (shift_amount_sz & (bits_per_uint64_sz << 1))
            {
                result[0] = operand[2];
                result[1] = 0;
                result[2] = 0;
            }
            else if (shift_amount_sz & bits_per_uint64_sz)
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
            std::size_t bit_shift_amount = shift_amount_sz & (bits_per_uint64_sz - 1);

            if (bit_shift_amount)
            {
                std::size_t neg_bit_shift_amount = bits_per_uint64_sz - bit_shift_amount;

                // Warning: if bit_shift_amount == 0 this is incorrect
                result[0] = (result[0] >> bit_shift_amount) |
                    (result[1] << neg_bit_shift_amount);
                result[1] = (result[1] >> bit_shift_amount) |
                    (result[2] << neg_bit_shift_amount);
                result[2] = result[2] >> bit_shift_amount;
            }
        }

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
            int bits_per_uint = (sizeof(T) == 8) ? bits_per_uint64 : bits_per_uint32;

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

        template<typename T>
        inline void not_uint(
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
            for (; uint64_count--; result++, operand++)
            {
                *result = ~*operand;
            }
        }

        template<typename T>
        inline void and_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand1 && uint64_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && uint64_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && uint64_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            for (; uint64_count--; result++, operand1++, operand2++)
            {
                *result = *operand1 & *operand2;
            }
        }

        template<typename T>
        inline void or_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand1 && uint64_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && uint64_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && uint64_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            for (; uint64_count--; result++, operand1++, operand2++)
            {
                *result = *operand1 | *operand2;
            }
        }

        template<typename T>
        inline void xor_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
        {
#ifdef XeHE_DEBUG
            if (!operand1 && uint64_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && uint64_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && uint64_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            for (; uint64_count--; result++, operand1++, operand2++)
            {
                *result = *operand1 ^ *operand2;
            }
        }

        template<typename T, typename S,
            typename = std::enable_if<is_uint64_v<T, S>>,
            typename = std::enable_if<is_uint32_v<T, S>>>
        inline void multiply_uint64_generic(
            T operand1, S operand2, T *result128)
        {
#ifdef XeHE_DEBUG
            if (!result128)
            {
                throw std::invalid_argument("result128 cannot be null");
            }
#endif
            XeHE_IF_CONSTEXPR(is_uint32_v<T>)
            {
                // uint32_t
                // asummes underlying 64 bit arithmetic exitsts but slow
                uint64_t result64 = uint64_t(operand1) * uint64_t(operand2);
                result128[0] = (T)(result64 & 0x00000000FFFFFFFF);
                result128[1] = (T)(result64 >> 32);
                
            }
            else XeHE_IF_CONSTEXPR(is_uint64_v<T>)
            {
#if 0

#if 0


                result128[1] = operand2;
                result128[0] = operand1;
#else
                result128[1] = operand1 + operand2;
                result128[0] = result128[1] + operand1;
#endif
#else
                // unit64_t
                //std::cout << "generic multiply uint64" << std::endl;
                auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
                auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
                uint64_t tmp1 = operand1;
                tmp1 >>= 32;
                operand1 = T(tmp1);
                uint64_t tmp2 = operand2;
                tmp2 >>= 32;
                operand2 = T(tmp2);

                auto middle1 = operand1 * operand2_coeff_right;
                T middle;
                auto left = operand1 * operand2 + (static_cast<uint64_t>(add_uint64(
                    middle1, operand2 * operand1_coeff_right, &middle)) << 32);
                auto right = operand1_coeff_right * operand2_coeff_right;
                auto temp_sum = (uint64_t(right) >> 32) + (middle & 0x00000000FFFFFFFF);

                result128[1] = static_cast<unsigned long long>(
                    left + (uint64_t(middle) >> 32) + (temp_sum >> 32));
                result128[0] = static_cast<unsigned long long>(
                    (uint64_t(temp_sum) << 32) | (right & 0x00000000FFFFFFFF));
#endif
            }
        }



        template<typename T, typename S,
            typename = std::enable_if<is_uint64_v<T, S>>,
            typename = std::enable_if<is_uint32_v<T, S>>>
        inline void multiply_uint64(
            T operand1, S operand2, T *result128)
        {
            XeHE_MULTIPLY_UINT64(operand1, operand2, result128);
        }

        template<typename T, typename S,
            typename = std::enable_if<is_uint64_v<T, S>>,
            typename = std::enable_if<is_uint32_v<T, S>>>
            inline void multiply_uint64_hw64_generic(
            T operand1, S operand2, T *hw64)
        {
            T result128[2];
            XeHE_MULTIPLY_UINT64(operand1, operand2, result128);
            *hw64 = result128[1];
        }

        template<typename T, typename S,
            typename = std::enable_if<is_uint64_v<T, S>>,
            typename = std::enable_if<is_uint32_v<T, S>>>
            inline void multiply_uint64_hw64(
            T operand1, S operand2, T *hw64)
        {
            XeHE_MULTIPLY_UINT64_HW64(operand1, operand2, hw64);
        }

        template<typename T>
        void multiply_uint_uint(
            const T* operand1, std::size_t operand1_uint64_count,
            const T* operand2, std::size_t operand2_uint64_count,
            std::size_t result_uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!operand1 && operand1_uint64_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && operand2_uint64_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result_uint64_count)
            {
                throw std::invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (result != nullptr && (operand1 == result || operand2 == result))
            {
                throw std::invalid_argument("result cannot point to the same value as operand1 or operand2");
            }
#endif
            // Handle fast cases.
            if (!operand1_uint64_count || !operand2_uint64_count)
            {
                // If either operand is 0, then result is 0.
                set_zero_uint(result_uint64_count, result);
                return;
            }
            if (result_uint64_count == 1)
            {
                *result = (*operand1) * (*operand2);
                return;
            }

            // In some cases these improve performance.
            operand1_uint64_count = get_significant_uint64_count_uint(
                operand1, operand1_uint64_count);
            operand2_uint64_count = get_significant_uint64_count_uint(
                operand2, operand2_uint64_count);

            // More fast cases
            if (operand1_uint64_count == 1)
            {
                multiply_uint_uint64<T>(operand2, operand2_uint64_count,
                    *operand1, result_uint64_count, result);
                return;
            }
            if (operand2_uint64_count == 1)
            {
                multiply_uint_uint64<T>(operand1, operand1_uint64_count,
                    *operand2, result_uint64_count, result);
                return;
            }

            // Clear out result.
            set_zero_uint(result_uint64_count, result);

            // Multiply operand1 and operand2.
            size_t operand1_index_max = std::min(operand1_uint64_count,
                result_uint64_count);
            for (size_t operand1_index = 0;
                operand1_index < operand1_index_max; operand1_index++)
            {
                const T* inner_operand2 = operand2;
                T* inner_result = result++;
                T carry = 0;
                size_t operand2_index = 0;
                size_t operand2_index_max = std::min(operand2_uint64_count,
                    result_uint64_count - operand1_index);
                for (; operand2_index < operand2_index_max; operand2_index++)
                {
                    // Perform 64-bit multiplication of operand1 and operand2
                    T temp_result[2];
                    multiply_uint64(*operand1, *inner_operand2++, temp_result);
                    carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, temp_result);
                    T temp;
                    carry += add_uint64(*inner_result, temp_result[0], 0, &temp);
                    *inner_result++ = temp;
                }

                // Write carry if there is room in result
                if (operand1_index + operand2_index_max < result_uint64_count)
                {
                    *inner_result = carry;
                }

                operand1++;
            }
        }


        template<typename T>
        inline void multiply_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
        {
            multiply_uint_uint(operand1, uint64_count, operand2, uint64_count,
                uint64_count * 2, result);
        }

        template<typename T>
        void multiply_uint_uint64(
            const T* operand1, std::size_t operand1_uint64_count,
            T operand2, std::size_t result_uint64_count,
            T* result)
        {
#ifdef XeHE_DEBUG
            if (!operand1 && operand1_uint64_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!result_uint64_count)
            {
                throw std::invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (result != nullptr && operand1 == result)
            {
                throw std::invalid_argument("result cannot point to the same value as operand1");
            }
#endif
            // Handle fast cases.
            if (!operand1_uint64_count || !operand2)
            {
                // If either operand is 0, then result is 0.
                set_zero_uint(result_uint64_count, result);
                return;
            }
            if (result_uint64_count == 1)
            {
                *result = *operand1 * operand2;
                return;
            }

            // More fast cases
            //if (result_uint64_count == 2 && operand1_uint64_count > 1)
            //{
            //    unsigned long long temp_result;
            //    multiply_uint64(*operand1, operand2, &temp_result);
            //    *result = temp_result;
            //    *(result + 1) += *(operand1 + 1) * operand2;
            //    return;
            //}

            // Clear out result.
            set_zero_uint(result_uint64_count, result);

            // Multiply operand1 and operand2.
            T carry = 0;
            size_t operand1_index_max = std::min(operand1_uint64_count,
                result_uint64_count);
            for (size_t operand1_index = 0;
                operand1_index < operand1_index_max; operand1_index++)
            {
                T temp_result[2];
                multiply_uint64(*operand1++, operand2, temp_result);
                T temp;
                carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, &temp);
                *result++ = temp;
            }

            // Write carry if there is room in result
            if (operand1_index_max < result_uint64_count)
            {
                *result = carry;
            }
        }


        template<typename T>
        inline void multiply_truncate_uint_uint(
            const T *operand1, const T *operand2,
            std::size_t uint64_count, T *result)
        {
            multiply_uint_uint(operand1, uint64_count, operand2, uint64_count,
                uint64_count, result);
        }

        template<typename T>
        void divide_uint_uint_inplace(T* numerator,
            const T* denominator, size_t uint64_count,
            T* quotient)
        {
#ifdef XeHE_DEBUG
            if (!numerator && uint64_count > 0)
            {
                throw std::invalid_argument("numerator");
            }
            if (!denominator && uint64_count > 0)
            {
                throw std::invalid_argument("denominator");
            }
            if (!quotient && uint64_count > 0)
            {
                throw std::invalid_argument("quotient");
            }
            if (is_zero_uint(denominator, uint64_count) && uint64_count > 0)
            {
                throw std::invalid_argument("denominator");
            }
            if (quotient && (numerator == quotient || denominator == quotient))
            {
                throw std::invalid_argument("quotient cannot point to same value as numerator or denominator");
            }
#endif
            if (!uint64_count)
            {
                return;
            }

            // Clear quotient. Set it to zero.
            set_zero_uint(uint64_count, quotient);

            // Determine significant bits in numerator and denominator.
            int numerator_bits =
                get_significant_bit_count_uint<T>(numerator, uint64_count);
            int denominator_bits =
                get_significant_bit_count_uint<T>(denominator, uint64_count);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Only perform computation up to last non-zero uint64s.
            uint64_count = safe_cast<size_t>(
                divide_round_up<T>(numerator_bits, bits_per_uint64));

            // Handle fast case.
            if (uint64_count == 1)
            {
                *quotient = *numerator / *denominator;
                *numerator -= *quotient * *denominator;
                return;
            }

            auto alloc_anchor = allocate_uint<T>(uint64_count << 1);

            // Create temporary space to store mutable copy of denominator.
            uint64_t* shifted_denominator = alloc_anchor.get();

            // Create temporary space to store difference calculation.
            uint64_t* difference = shifted_denominator + uint64_count;

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;
            left_shift_uint<T>(denominator, denominator_shift, uint64_count,
                shifted_denominator);
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (sub_uint_uint<T>(numerator, shifted_denominator,
                    uint64_count, difference))
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
                    add_uint_uint<T>(difference, numerator, uint64_count, difference);

                    // Adjust quotient and remaining shifts as a result of
                    // shifting numerator.
                    left_shift_uint<T>(quotient, 1, uint64_count, quotient);
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Determine amount to shift numerator to bring MSB in alignment
                // with denominator.
                numerator_bits = get_significant_bit_count_uint<T>(difference, uint64_count);
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
                    left_shift_uint<T>(difference, numerator_shift, uint64_count, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                    // Difference is zero so no need to shift, just set to zero.
                    set_zero_uint<T>(uint64_count, numerator);
                }

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint<T>(quotient, numerator_shift, uint64_count, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                right_shift_uint<T>(numerator, denominator_shift, uint64_count, numerator);
            }
        }

        template<typename T>
        inline void divide_uint_uint(
            const T *numerator, const T *denominator,
            std::size_t uint64_count, T *quotient,
            T *remainder)
        {
            set_uint_uint<T>(numerator, uint64_count, remainder);
            divide_uint_uint_inplace<T>(remainder, denominator, uint64_count, quotient);
        }

        template<typename T>
        void divide_uint128_uint64_inplace_generic(T* numerator,
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
            // We expect 129-bit input
            constexpr size_t uint64_count = 2;

            // Clear quotient. Set it to zero.
            quotient[0] = 0;
            quotient[1] = 0;

            // Determine significant bits in numerator and denominator.
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Create temporary space to store mutable copy of denominator.
            uint64_t shifted_denominator[uint64_count]{ denominator, 0 };

            // Create temporary space to store difference calculation.
            uint64_t difference[uint64_count]{ 0, 0 };

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;

            left_shift_uint128(shifted_denominator, denominator_shift, shifted_denominator);
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (sub_uint_uint(numerator, shifted_denominator, uint64_count, difference))
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
                    add_uint_uint(difference, numerator, uint64_count, difference);

                    // Adjust quotient and remaining shifts as a result of shifting numerator.
                    quotient[1] = (quotient[1] << 1) | (quotient[0] >> (bits_per_uint64 - 1));
                    quotient[0] <<= 1;
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Determine amount to shift numerator to bring MSB in alignment
                // with denominator.
                numerator_bits = get_significant_bit_count_uint(difference, uint64_count);

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
                    left_shift_uint128(difference, numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint128(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                right_shift_uint128(numerator, denominator_shift, numerator);
            }
        }

        template<typename T>
        inline void divide_uint128_uint64_inplace(
            T *numerator, T denominator,
            T *quotient)
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
            XeHE_DIVIDE_UINT128_UINT64(numerator, denominator, quotient);
        }

        template<typename T>
        void divide_uint128_uint64_inplace(
            T *numerator, T denominator,
            T *quotient);

        template<typename T>
        void divide_uint192_uint64_inplace(
            T *numerator, T denominator,
            T *quotient);

        template<typename T>
        void exponentiate_uint(
            const T* operand, std::size_t operand_uint64_count,
            const T* exponent, std::size_t exponent_uint64_count,
            std::size_t result_uint64_count, T* result);

        template<typename T>
        XeHE_NODISCARD T exponentiate_uint64_safe(
            T operand, T exponent);

        template<typename T>
        XeHE_NODISCARD T exponentiate_uint64(
            T operand, T exponent);
        
    }
}

#endif
