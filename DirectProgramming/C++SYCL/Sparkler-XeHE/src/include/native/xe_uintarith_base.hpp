/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_UINTARITH_BASE_HPP
#define XeHE_UINTARITH_BASE_HPP

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <type_traits>

// XeHE
#include "../dpcpp_utils.h"

namespace xehe
{
    namespace native
    {

        template<typename T>
        inline T and_int(T op1, T op2)
        {
            T ret = op1 && op2;
            return ret;
        }


        template<typename T>
        inline T or_int(T op1, T op2)
        {
            T ret = op1 || op2;
            return ret;
        }


        // op1 >= op2?
        template<typename T>
        inline bool ge(T op1, T op2)
        {
            bool ret = (op1 >= op2);
            return(ret);
        }

        // op1 > op2?
        template<typename T>
        inline bool gt(T op1, T op2)
        {
            bool ret = (op1 > op2);
            return(ret);
        }

        // op1 <= op2?
        template<typename T>
        inline bool le(T op1, T op2)
        {
            bool ret = (op1 <= op2);
            return(ret);
        }

        // op1 < op2?
        template<typename T>
        inline bool lt(T op1, T op2)
        {
            bool ret = (op1 < op2);
            return(ret);
        }


        // op1 != op2
        template<typename T>
        inline bool ne(T op1, T op2)
        {
            bool ret = (op1 != op2);
            return(ret);
        }

        template<typename T>
        inline T bit_and_int(T op1, T op2)
        {
            T ret = op1 & op2;
            return ret;
        }

        template<typename T>
        inline T bit_or_int(T op1, T op2)
        {
            T ret = op1 | op2;
            return ret;
        }

        /**
        basic int add.

        @param[in] operand1
        @param[in] operand2
        @throws std::invalid_argument if the encryption parameters are not valid
        */
        template<typename T>
        inline T add_int(
            T operand1, T operand2)
        {
            auto ret = operand1 + operand2;
            return (ret);
        }

        /**
        basic int sub.

        @param[in] operand1
        @param[in] operand2
        */
        template<typename T>
        inline T sub_int(
            T operand1, T operand2)
        {
            auto ret = operand1 - operand2;
            return (ret);
        }

        /**
        basic left shift.

        @param[in] operand
        @param[in] shift
        */

        template<typename T>
        inline T left_shift(
            T operand, int shift)
        {
            auto ret = T(operand << shift);
            return (ret);
        }

        /**
        basic right shift.

        @param[in] operand
        @param[in] shift
        */

        template<typename T>
        inline T right_shift(
            T operand, int shift)
        {
            auto ret = T(operand >> shift);
            return (ret);
        }


        template<typename T>
        inline int bits_per_uint(void)
        {
            return(sizeof(T)* 8);
        }

        template<typename T>
        inline int max_mod_bits(void)
        {

            return(xehe::native::bits_per_uint<T>() - 4);
        }

        template<typename T>
        inline int max_mul_accum_mod_count(void)
        {

            return(xehe::native::bits_per_uint<T>() - xehe::native::max_mod_bits<T>());
        }


        template<typename T>
        inline uint32_t get_msb_index(T val)
        {


            uint32_t res = 0;
            if (sizeof(T) == 4)
            {
                res = uint32_t(ilogb(double(val)));
            }
            else if (sizeof(T) == 8)
            {
                const uint64_t m = 0xffff000000000000;
                auto masked = val & m;
                res = (masked == 0) ? uint32_t(ilogb(double(val))) : uint32_t(ilogb(float((masked >> 48)))) + 48;
            }
            return(res);
        }

        template<typename T>
        inline int get_significant_bit_count(T value)
        {
            if (value == 0)
            {
                return 0;
            }

            uint32_t result = xehe::native::get_msb_index<T>(value);
            return static_cast<int>(result + 1);
        }

        template<typename T>
        inline int get_significant_bit_count_uint(
            const T* value, std::size_t uint64_count)
        {
            auto bits_per_uint_sz = xehe::native::bits_per_uint<T>();

#ifdef XeHE_DEBUG
            if (!value && uint64_count)
            {
                throw std::invalid_argument("value");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
#endif
            value += uint64_count - 1;
            for (; *value == 0 && uint64_count > 1; uint64_count--)
            {
                value--;
            }

            return static_cast<int>(uint64_count - 1) * bits_per_uint_sz +
                xehe::native::get_significant_bit_count<T>(*value);
        }

        template<typename T>
        inline void set_zero_uint(std::size_t uint64_count, T* result)
        {
#ifdef XeHE_DEBUG
            if (!result && uint64_count)
            {
                throw std::invalid_argument("result");
            }
#endif
            std::fill_n(result, uint64_count, T(0));
        }

        template<typename T>
        inline constexpr T add_safe(T in1, T in2)
        {
            if constexpr (std::is_unsigned<T>::value)
            {
                T result = xehe::native::add_int<T>(in1, in2);
#ifdef XeHE_DEBUG
                if (result < in1)
                {
                    throw std::logic_error("unsigned overflow");
                }
#endif
                return result;
            }
            else
            {
#ifdef XeHE_DEBUG
                if (in1 > 0 && (in2 > std::numeric_limits<T>::max() - in1))
                {
                    throw std::logic_error("signed overflow");
                }
                else if (in1 < 0 &&
                    (in2 < std::numeric_limits<T>::min() - in1))
                {
                    throw std::logic_error("signed underflow");
                }
#endif
                return (xehe::native::add_int<T>(in1, in2));
            }
        }


        template<typename T>
        inline T sub_safe(T in1, T in2)
        {
            if constexpr (std::is_unsigned<T>::value)
            {
                T result = xehe::native::sub_int<T>(in1, in2);
#ifdef XeHE_DEBUG
                if (result > in1)
                {
                    throw std::logic_error("unsigned underflow");
                }
#endif
                return result;
            }
            else
            {
#ifdef XeHE_DEBUG
                if (in1 < 0 && (in2 > std::numeric_limits<T>::max() + in1))
                {
                    throw std::logic_error("signed underflow");
                }
                else if (in1 > 0 &&
                    (in2 < std::numeric_limits<T>::min() + in1))
                {
                    throw std::logic_error("signed overflow");
                }
#endif
                return (xehe::native::sub_int<T>(in1, in2));
            }
        }

        template<typename T>
        inline constexpr T mul_safe(T in1, T in2)
        {
            if constexpr (std::is_unsigned<T>::value)
            {
                if (in1 && (in2 > std::numeric_limits<T>::max() / in1))
                {
                    throw std::logic_error("unsigned overflow");
                }
            }
            else
            {
                // Positive inputs
                if ((in1 > 0) && (in2 > 0) &&
                    (in2 > std::numeric_limits<T>::max() / in1))
                {
                    throw std::logic_error("signed overflow");
                }

                // Negative inputs
                else if ((in1 < 0) && (in2 < 0) &&
                    ((-in2) > std::numeric_limits<T>::max() / (-in1)))
                {
                    throw std::logic_error("signed overflow");
                }
                // Negative in1; positive in2
                else if ((in1 < 0) && (in2 > 0) &&
                    (in2 > std::numeric_limits<T>::max() / (-in1)))
                {
                    throw std::logic_error("signed underflow");
                }
                // Positive in1; negative in2
                else if ((in1 > 0) && (in2 < 0) &&
                    (in2 < std::numeric_limits<T>::min() / in1))
                {
                    throw std::logic_error("signed underflow");
                }
            }
            return in1 * in2;
        }


        template<typename T>
        inline constexpr T reverse_bits(T operand) noexcept
        {
            if constexpr (sizeof(T)==4)
            {
                operand = (((operand & T(0xaaaaaaaa)) >> 1) | ((operand & T(0x55555555)) << 1));
                operand = (((operand & T(0xcccccccc)) >> 2) | ((operand & T(0x33333333)) << 2));
                operand = (((operand & T(0xf0f0f0f0)) >> 4) | ((operand & T(0x0f0f0f0f)) << 4));
                operand = (((operand & T(0xff00ff00)) >> 8) | ((operand & T(0x00ff00ff)) << 8));
                return static_cast<T>(operand >> 16) | static_cast<T>(operand << 16);
            }
            else if constexpr (sizeof(T)==8)
            {
                return static_cast<T>(reverse_bits(static_cast<std::uint32_t>(operand >> 32))) |
                    (static_cast<T>(reverse_bits(static_cast<std::uint32_t>(operand & T(0xFFFFFFFF)))) << 32);
            }
        }

        template<typename T>
        inline T reverse_bits(T operand, int bit_count)
        {
#ifdef XeHE_DEBUG
            if (bit_count < 0 ||
                static_cast<std::size_t>(bit_count) >
                    mul_safe(sizeof(T), static_cast<std::size_t>(xehe::native::bits_per_uint<T>())))
            {
                throw std::invalid_argument("bit_count");
            }
#endif
            int bits_per_uint = xehe::native::bits_per_uint<T>();
            // Just return zero if bit_count is zero
            return (bit_count == 0) ? T(0) : reverse_bits(operand) >> (
                static_cast<std::size_t>(bits_per_uint)
                    - static_cast<std::size_t>(bit_count));
        }


        template<typename T>
        inline T divide_round_up(T value, T divisor)
        {
#ifdef XeHE_DEBUG
            if (value < 0)
            {
                throw std::invalid_argument("value");
            }
            if (divisor <= 0)
            {
                throw std::invalid_argument("divisor");
            }
#endif
            return (xehe::native::add_safe<T>(value, divisor - 1)) / divisor;
        }


    }
}

#endif
