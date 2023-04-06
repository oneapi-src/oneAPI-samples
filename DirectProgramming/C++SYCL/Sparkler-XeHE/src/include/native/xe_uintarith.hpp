/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_UINTARITH_HPP
#define XeHE_UINTARITH_HPP


#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <type_traits>

// XeHE
#include "xe_uintarith_base.hpp"
#include "lib_utils.h"
#include "util/inline_kernels.hpp"

namespace xehe
{
    namespace native
    {



        /**
        add 2 uint values with carry.

        @param[in] operand1
        @param[in] operand2
        @param[in] carry
        @param[out] result

        @return carry value in the daisy chain
        */
        template<typename T>
        inline T add_uint(
            T operand1, T operand2, T carry,
            T *result)
        {
            operand1 = xehe::native::add_int<T>(operand1, operand2);
            *result = xehe::native::add_int<T>(operand1, carry);
            T ret = T((operand1 < operand2) || (~operand1 < carry));
            return (ret);
        }


        template<typename T>
        inline T add_uint(
            T operand1, T operand2,
            T* result)
        {
            operand1 = xehe::native::add_int<T>(operand1, operand2);
            *result = operand1;
            T ret = T(operand1 < operand2);
            return (ret);
        }

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
            auto carry = xehe::native::add_uint<T>(*operand1++, *operand2++, result++);

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
            auto carry = xehe::native::add_uint<T>(operand1[0], operand2[0], result);
            auto ret = xehe::native::add_uint<T>(operand1[1], operand2[1], carry, result + 1);
            return(ret);
        }

        /**
        sub 2 uint values with borrow.

        @param[in] operand1
        @param[in] operand2
        @param[in] borrow
        @param[out] result

        @return borrow value in the daisy chain
        */

        template<typename T>
        inline T sub_uint(
                T operand1, T operand2,
                T borrow, T* result)
        {
            auto diff = xehe::native::sub_int<T>(operand1, operand2);
            //auto b_ne_0 = T(borrow != 0);
            *result = xehe::native::sub_int<T>(diff, borrow);
            T ret = T((diff > operand1) || (diff < borrow));
            return (ret);
        }

        template<typename T>
        inline T sub_uint(
            T operand1, T operand2,
            T* result)
        {
            auto diff = xehe::native::sub_int<T>(operand1, operand2);

            *result = diff;
            T ret = T(diff > operand1);
            return (ret);
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
            auto borrow = xehe::native::sub_uint<T>(*operand1++, *operand2++, result++);

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
            const T* operand, int shift_amount, T* result)
        {
            const int bits_per_uint = int(xehe::native::bits_per_uint<T>());

#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                xehe::native::ge(shift_amount, 2 * bits_per_uint))
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
            const T* operand, int shift_amount, T* result)
        {


            const int bits_per_uint = int(xehe::native::bits_per_uint<T>());
#ifdef XeHE_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (shift_amount < 0 ||
                xehe::native::ge(shift_amount, 2 * bits_per_uint))
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


        template<typename T>
        inline T inc_uint(T operand, T*result)
        {

            return xehe::native::add_uint(operand, T(1), T(0), result);
        }

        template<typename T>
        inline T dec_uint(T operand, T* result)
        {
            return xehe::native::sub_uint(operand, T(1), T(0), result);
        }

        template<typename T>
        inline T neg_uint(T operand)
        {
            // Negation is equivalent to inverting bits and adding 1.
            T ret;
            xehe::native::add_uint(~operand, T(1), T(0), ret);
            return (ret);
        }

        /**
        multiplies 2 uint values. returns the low half part of the result

        @param[in] operand1
        @param[in] operand2

        @return result
        */


        template<typename T>
        inline T mul_uint_low(T operand1, T operand2)
        {
            T ret = operand1 * operand2;
            return(ret);
        }





        /**
        multiplies 2 uint values. returns the low part of the result

        @param[in] operand1
        @param[in] operand2
        @param[out] high_product - high part of the result

        @return low part of the result
        */


        template<typename T>
        inline T mul_uint_kara(T op1_low, T op1_high, T op2_low, T op2_high, int shift, T* prod_high)
        {
            auto right = xehe::native::mul_uint_low<T>(op1_low, op2_low);
            auto left = xehe::native::mul_uint_low<T>(op1_high, op2_high);

            xehe::w64_t t1, t2;
            t1.value64b = xehe::native::add_int<T>(op1_low, op1_high);
            t2.value64b = xehe::native::add_int<T>(op2_low, op2_high);

            // full t1 * t2 = (2^32 + t1.low) * (2^32 + t2.low) = 2^64 + t1.low*2^32 + t2.low*2^32 + t1.low*t2.low
            // + t2 << 32  
            auto tt1 = (T(uint32_t(-int(t1.high)) & t2.low) << 32);
            // + t1 << 32           
            auto tt2 = (T(uint32_t(-int(t2.high)) & t1.low) <<32);
            

            // + 1<< 64
            auto kara_bit = ((t1.value64b & T(0x100000000)) & (t2.value64b & T(0x100000000)));

            t1.high = 0;
            t2.high = 0;
            // WATCH FOT THIS MUL
            T z1[2];
            z1[0] = xehe::native::mul_uint_low<T>(t1.value64b, t2.value64b);
            z1[1] = xehe::native::add_uint<T>(z1[0], tt1, z1);
            z1[1] += xehe::native::add_uint<T>(z1[0], tt2, z1);
            z1[1] += (kara_bit >> 32);

            // Z0 + Z2
            T z0z2[2];
            z0z2[1] = xehe::native::add_uint<T>(left, right, z0z2);

            // Z1 - (Z2 + Z0)
            T mid_tmp[2];
            //xehe::native::sub_uint<T>((const T*)z1, (const T*)z0z2, size_t(2), (T*)&mid_tmp[0].value64b);
            auto borrow = xehe::native::sub_uint<T>(z1[0], z0z2[0], mid_tmp);
            mid_tmp[1] = xehe::native::sub_int<T>(z1[1], z0z2[1]);
            mid_tmp[1] = xehe::native::sub_int<T>(mid_tmp[1], borrow );            


            //xehe::native::left_shift2(mid_tmp, 32, mid_tmp);
            mid_tmp[1] <<= 32;
            T tmp = (mid_tmp[0] >> 32);
            mid_tmp[1] |= tmp;
            mid_tmp[0] <<= 32;

            T prod_low;
            //xehe::native::add_uint<T>((const T*)res, (const T*)&mid_tmp[0].value64b, size_t(2), (T*)res);
            auto carry = xehe::native::add_uint<T>(right, mid_tmp[0], &prod_low);

            mid_tmp[1] = xehe::native::add_int<T>(left, mid_tmp[1]);
            *prod_high = xehe::native::add_int<T>(mid_tmp[1], carry);
                     

            return(prod_low);
        }

        // template<typename T>
        // inline T mul_uint_intnl(T op1_low, T op1_high, T op2_low, T op2_high, int shift, T* prod_high)
        // {
        //     auto right = xehe::native::mul_uint_low<T>(op1_low, op2_low);
        //     auto left = xehe::native::mul_uint_low<T>(op1_high, op2_high);
        //     T middle;
 
        //     auto h1_l2 = xehe::native::mul_uint_low<T>(op1_high, op2_low);
        //     auto l1_h2 = xehe::native::mul_uint_low<T>(op1_low, op2_high);

        //     auto middle_carry = xehe::native::add_uint<T>(h1_l2, l1_h2, &middle);

        //     auto middle_low = xehe::native::left_shift<T>(middle, shift);

        //     T prod_low;
        //     auto carry1 = xehe::native::add_uint<T>(right, middle_low, &prod_low);
        //     middle_carry = xehe::native::left_shift<T>(middle_carry, shift);
        //     carry1 = xehe::native::add_int<T>(carry1, middle_carry);
        //     auto middle_high = xehe::native::right_shift<T>(middle, shift);
        //     xehe::native::add_uint<T>(left, middle_high, carry1, prod_high);

        //     return(prod_low);
        // }

        template<typename T>
        inline T mul_uint_intnl(T op1_low, T op1_high, T op2_low, T op2_high, int shift, T* prod_high)
        {
            T right, left, middle, h1_l2, l1_h2;

#if defined(XeHE_INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
            asm(MUL_UINT_OPT_64_STR(_SIMD_WIDTH_)
                : "+rw"(right)
                : "rw"(op1_low), "rw"(op2_low));
            
            asm(MUL_UINT_OPT_64_STR(_SIMD_WIDTH_)
                : "+rw"(left)
                : "rw"(op1_high), "rw"(op2_high));

            asm(MUL_UINT_OPT_64_STR(_SIMD_WIDTH_)
                : "+rw"(h1_l2)
                : "rw"(op1_high), "rw"(op2_low));

            asm(MUL_UINT_OPT_64_STR(_SIMD_WIDTH_)
                : "+rw"(l1_h2)
                : "rw"(op1_low), "rw"(op2_high));
#else 
            right = xehe::native::mul_uint_low<T>(op1_low, op2_low);
            left = xehe::native::mul_uint_low<T>(op1_high, op2_high); 
            h1_l2 = xehe::native::mul_uint_low<T>(op1_high, op2_low);
            l1_h2 = xehe::native::mul_uint_low<T>(op1_low, op2_high);
#endif
            
            auto middle_carry = xehe::native::add_uint<T>(h1_l2, l1_h2, &middle);

            auto middle_low = xehe::native::left_shift<T>(middle, shift);

            T prod_low;
            auto carry1 = xehe::native::add_uint<T>(right, middle_low, &prod_low);
            middle_carry = xehe::native::left_shift<T>(middle_carry, shift);
            carry1 = xehe::native::add_int<T>(carry1, middle_carry);
            auto middle_high = xehe::native::right_shift<T>(middle, shift);
            xehe::native::add_uint<T>(left, middle_high, carry1, prod_high);

            return(prod_low);
        }

        template<typename T>
        inline T mul_uint(T op1, T op2, T* prod_high)
        {

            T ret = 0;
            *prod_high = 0;

            if constexpr (sizeof(T) == 4)
            {


                auto ret64 = xehe::native::mul_uint_low<uint64_t>(uint64_t(op1), uint64_t(op2));
                ret = ((T*)&ret64)[0];
                *prod_high = ((T*)&ret64)[1];
                
            }
            else  if constexpr (sizeof(T) == 8)
            {
                auto mask = T(0x00000000FFFFFFFF);
                auto shift = int(32);
                auto op1_low = op1 & mask;
                auto op2_low = op2 & mask;
                auto op1_high = xehe::native::right_shift(op1, shift);
                auto op2_high = xehe::native::right_shift(op2, shift);

#if 0
                {
                    ret = mul_uint_kara(op1_low, op1_high, op2_low, op2_high, shift, prod_high);
                }   
#else              
              
                {            
                    ret = mul_uint_intnl<T>(op1_low, op1_high, op2_low, op2_high, shift, prod_high);   
                }
#endif
         
            }
            
            return(ret);
        }

        template<typename T>
        inline void mul_uint2(T op1, T op2, T* result){
            result[0] = native::mul_uint<T>(op1, op2, result + 1);
        }

        /**
        multiplies 2 uint values. returns the high half part of the result

        @param[in] operand1
        @param[in] operand2

        @return result
        */


        template<typename T>
        inline T mul_uint_high(T op1, T op2)
        {
            T ret = 0;
            xehe::native::mul_uint(op1, op2, &ret);
            return(ret);
        }

        // op1 * op2 + op3[size=2]
        // op3 is an array containing low/high parts of a number
        // op1 and op2 should be <= 63 bits if T=uint64_t
        // to avoid overflowing
        template<typename T>
        inline T mad_uint(T op1, T op2, T *op3, T *prod_high)
        {
            T z[2], tmp[2];
            z[0] = mul_uint(op1, op2, z+1);
            xehe::native::add_uint(z, op3, tmp);
            *prod_high = tmp[1];
            return tmp[0];
        }

        template<typename T>
        inline T mad_uint(T op1, T op2, T op3, T *prod_high)
        {
            T low, high, ret;
            low = mul_uint(op1, op2, &high);
            auto carry = xehe::native::add_uint(low, op3, &ret);
            *prod_high = xehe::native::add_int(high, carry);//assumed not overflowing
            return ret;
        }


        /******************************************************************************************
        *          
        * Correctness conditions:
        * operand1,operand2 < mod
        * bit_count(mod) <= (bit_size(T) - 1)
        *
        * ****************************************************************************************/


        template<typename T>
        inline  T inc_mod(T operand, T modulus)
        {
            T ret = 0;
            auto carry = xehe::native::inc_uint(operand, &ret);
            if (carry || xehe::native::ge<T>(ret, modulus))
            {
                xehe::native::sub_uint<T>(ret, modulus, T(0), &ret);
            }

            return(ret);
        }

        template<typename T>
        inline T dec_mod(T operand, T modulus)
        {
            T ret = 0;
            if (xehe::native::dec_uint(operand, &ret))
            {
                xehe::native::add_uint(ret, modulus, T(0), &ret);
            }

            return(ret);
        }

        template<typename T>
        inline T neg_mod(T op, T mod)
        {

            T ret = 0;

            // operand > 0 and < modulus so substract modulus - operand.
            ret = (op == 0)? 0 : xehe::native::sub_int(mod, op);

            return(ret);
        }

        template<typename T>
        inline T div2_mod(T operand, T modulus)
        {
            T ret = 0;
            if (operand & 1)
            {
                T carry = xehe::native::add_uint(operand, modulus, T(0), &ret);
                ret = xehe::native::right_shift<T>(ret, 1);
                if (carry)
                {
                    auto bits = int(xehe::native::bits_per_uint<T>());
                    ret |= (T(1) << (bits - 1));
                }
            }
            else
            {
                ret = xehe::native::right_shift<T>(operand, 1);
            }
        }


        //  /**
        //  mod(add). returns mod of sum of 2 uint values.
        // Correctness conditions:
        // the same

        //  @param[in] operand1
        //  @param[in] operand2
        //  @param[in] modulus

        //  @return mod(op1 + op2)
        //  */


        // template<typename T>
        // inline T add_mod(const T operand1, const T operand2, const T modulus)
        // {

        //     T ret = 0;

// #ifdef Xe_HE_DEBUG
//             if (modulus == 0 || (operand1 + operand2) >= (modulus << 1) || (modulus & (T(1) << (sizeof(T) * 8 - 1))))
//             {
//                 throw std::invalid_argument("parmeters are out of range");
//             }
// #endif
//             // Sum of operands % mod can never wrap around 2^NumOfBits(T)
//             auto temp0 = xehe::native::add_int<T>(operand1, operand2);
//             auto t = -static_cast<int>(xehe::native::ge<T>(temp0, modulus));
//             auto temp1 = static_cast<T>(t);
//             temp1 = xehe::native::bit_and_int<T>(temp1,modulus);
//             ret = xehe::native::sub_int<T>(temp0,temp1);

//             return (ret);
//         }

        /**
         mod(add) using inline asm. returns mod of sum of 2 uint values.
        Correctness conditions:
        the same

         @param[in] operand1
         @param[in] operand2
         @param[in] modulus

         @return mod(op1 + op2)
         */


        template<typename T>
        inline T add_mod(const T operand1, const T operand2, const T modulus)
        {
            T ret = 0;

#ifdef Xe_HE_DEBUG
            if (modulus == 0 || (operand1 + operand2) >= (modulus << 1) || (modulus & (T(1) << (sizeof(T) * 8 - 1))))
            {
                throw std::invalid_argument("parmeters are out of range");
            }
#endif

#if defined(XeHE_INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
            if constexpr (sizeof(T) == 4)
            {
                asm(ADDMOD_OPT_32_STR(_SIMD_WIDTH_)
                : "+rw"(ret)
                : "rw"(operand1), "rw"(operand2), "rw"(modulus));
            }
            else if constexpr (sizeof(T) == 8)
            {
                asm(ADDMOD_OPT_64_STR(_SIMD_WIDTH_)
                : "+rw"(ret)
                : "rw"(operand1), "rw"(operand2), "rw"(modulus));
            }
#else
            // Sum of operands % mod can never wrap around 2^NumOfBits(T)
            auto temp0 = xehe::native::add_int<T>(operand1, operand2);
            auto t = -static_cast<int>(xehe::native::ge<T>(temp0, modulus));
            auto temp1 = static_cast<T>(t);
            temp1 = xehe::native::bit_and_int<T>(temp1,modulus);
            ret = xehe::native::sub_int<T>(temp0,temp1);
#endif
            return (ret);
        }

        /**
        mod(sub). returns mod of sub of 2 uint values.
        Correctness conditions:
        the same 

        @param[in] operand1
        @param[in] operand2
        @param[in] modulus

        @return mod(op1 - op2)
        */
        template<typename T>
        inline T sub_mod(const T operand1, const T operand2, const T modulus)
        {
            T ret = 0;
#ifdef Xe_HE_DEBUG
            if (modulus == 0 || (operand1 + operand2) >= (modulus << 1) || (modulus & (T(1) << (sizeof(T) * 8 - 1))))
            {
                throw std::invalid_argument("parmeters are out of range");
            }
#endif
            T temp;
            auto borrow = xehe::native::sub_uint<T>(operand1, operand2, 0, &temp);
            ret = xehe::native::bit_and_int<T>(modulus, T(-int(borrow)));
            ret = xehe::native::add_int<T>(ret,temp);
            return (ret);

        }

        /**
        It's very often the case when one operand of the mod(mul) is a constant over the lifetime of the app.
        NTT transform's coefficients are of such case.
        Than it's more effcient apply a special case of mod(mul) instead of generic one.
        This is such special case. It uses a precompute value operand2 * 2^bit_size(T) / modulus 
        
        Returns (x * y) mod modulus.
        Correctness conditions: 
         operand2 < mod
         bit_count(mod) <= (bit_size(T) - 1)

        @param[in] operand1
        @param[in] operand2
        @param[in] modulus
        @param[in] y_mod_quotent = operand2 * 2^bit_count(T) / modulus
        */
        template <typename T>
        inline T mul_quotent_mod(T x, T y, T modulus, T y_mod_quotent )
        {
            T ret = 0;
#ifdef Xe_HE_DEBUG
            if (y >= modulus || (modulus & (T(1) << (sizeof(T) * 8 - 1))))
            {
                throw std::invalid_argument("parmeters are out of range");
            }
#endif
            auto tmp1 = xehe::native::mul_uint_high(x, y_mod_quotent);
            tmp1 = xehe::native::mul_uint_low(tmp1, modulus);
            auto tmp2 =xehe::native::mul_uint_low(y, x);
            tmp2 = xehe::native::sub_int<T>(tmp2, tmp1);
            auto tmp3 = static_cast<T>(-static_cast<int>(xehe::native::ge<T>(tmp2, modulus)));
            auto p = modulus;
            p = xehe::native::bit_and_int<T>(p,tmp3);
            ret = xehe::native::sub_int<T>(tmp2, p);
            return (ret);
        }

        template <typename T>
        inline T mul_uint_mod_lazy(T x, T y, T modulus, T y_mod_quotent )
        {
#ifdef Xe_HE_DEBUG
            if (y >= modulus || (modulus & (T(1) << (sizeof(T) * 8 - 1))))
            {
                throw std::invalid_argument("parmeters are out of range");
            }
#endif
            auto tmp1 = xehe::native::mul_uint_high(x, y_mod_quotent);
            tmp1 = xehe::native::mul_uint_low(tmp1, modulus);
            auto tmp2 =xehe::native::mul_uint_low(y, x);
            tmp2 = xehe::native::sub_int<T>(tmp2, tmp1);
            return (tmp2);
        }


        /**
        It's very often the case when modulus  is a constant over the lifetime of the app.
        Than it's more effcient apply a special case of mod(val) instead of generic one.
        This is such special case. It uses a precompute value 2^bit_size(T) / modulus

        Returns (x) mod modulus.
        Correctness conditions:
        bit_count(mod) <= (bit_size(T) - 1)

        @param[in] input
        @param[in] operand2
        @param[in] modulus
        @param[in] mod_inv_quotent = 2^bit_size(T) / modulus
        */
        template <typename T>
        inline T barrett_reduce(T input, T modulus, T mod_inv_quotent)
        {
#ifdef Xe_HE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
#endif

            // Reduces input using base 2^64 Barrett reduction

            auto tmp = xehe::native::mul_uint_high<T>(input, mod_inv_quotent); 

            // Barrett subtraction
            auto tmp1 = input - xehe::native::mul_uint_low<T>(tmp, modulus);

            // One more subtraction is enough
            auto tmp3 = static_cast<T>(-static_cast<int>(xehe::native::ge<T>(tmp1, modulus)));
            auto tmp2 = xehe::native::bit_and_int<T>(modulus,tmp3);
            auto ret = xehe::native::sub_int<T>(tmp1, tmp2);

            return (ret);

        }



        /**
        It's very often the case when modulus  is a constant over the lifetime of the app.
        Than it's more effcient apply a special case of mod(val) instead of generic one.
        This is such special case. It uses a precompute value 2^bit_size(T[2]) / modulus

        Returns (x) mod modulus.
        Correctness conditions:
        bit_count(input) <= bit_size(T[2])  
        bit_count(mod) <= (bit_size(T) - 1)
        bit count(mod_inv_qoutent2) <= bit_size(T[2]) 

        @param[in] input
        @param[in] operand2
        @param[in] modulus
        @param[in] mod_inv_quotent2 = 2^bit_size(T[2]) / modulus
        */

        template <typename T>
            inline T barrett_reduce2(const T* input, T modulus, const T* mod_inv_quotent2)
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

            // input allocation size must be bit_size(T[2]) bits

            T tmp1, tmp2[2];

            // Multiply input and const_ratio
            // Round 1
            auto carry = xehe::native::mul_uint_high<T>(input[0], mod_inv_quotent2[0]);

            // 128 bit output
            tmp2[0]= xehe::native::mul_uint<T>(input[0], mod_inv_quotent2[1], tmp2+1);

            auto tmp3 = xehe::native::add_uint(tmp2[0], carry, T(0), &tmp1);
            tmp3 = xehe::native::add_int<T>(tmp2[1],tmp3);

            // Round 2
            tmp2[0] = xehe::native::mul_uint<T>(input[1], mod_inv_quotent2[0], tmp2+1);

            carry = xehe::native::add_uint<T>(tmp1, tmp2[0], T(0), &tmp1);
            carry = xehe::native::add_int<T>(tmp2[1], carry);

            // This is all we care about
            tmp1 = xehe::native::mul_uint_low<T>(input[1], mod_inv_quotent2[1]);
            tmp3 = xehe::native::add_int<T>(tmp3, carry);
            tmp1 = xehe::native::add_int<T>(tmp1, tmp3);

            // Barrett subtraction
            tmp1 = xehe::native::mul_uint_low<T>(tmp1, modulus);
            tmp3 = xehe::native::sub_int<T>(input[0],tmp1);

            // One more subtraction is enough
            auto tmp4 = static_cast<T>(-static_cast<int>(xehe::native::ge<T>(tmp3, modulus)));
            auto tmp5 = xehe::native::bit_and_int<T>(modulus, tmp4);
            auto ret = xehe::native::sub_int<T>(tmp3,tmp5);
            
            return (ret);
        }

            /**
            It's very often the case when modulus  is a constant over the lifetime of the app.
            Than it's more effcient apply a special case of mod(mul) instead of generic one.
            This is such special case. It uses a precompute value 2^bit_size(T[2]) / modulus

            Returns (x) mod modulus.
            Correctness conditions:
            bit_count(mod) <= (bit_size(T) - 1)
            bit count(mod_inv_qoutent2) <= bit_size(T[2])

            @param[in] input
            @param[in] operand2
            @param[in] modulus
            @param[in] mod_inv_quotent2 = 2^bit_size(T[2]) / modulus
            */


        template <typename T>
        inline T mul_mod(
            T operand1, T operand2, T modulus, const T* mod_inv_quotent2)
        {
#ifdef XeHE_DEBUG
            if (modulus == 0)
            {
                throw std::invalid_argument("modulus");
            }
#endif

            if constexpr (sizeof(T) == 8)
            {
                T z[2];
                z[0] = xehe::native::mul_uint<T>(operand1, operand2, z + 1);
                auto ret = barrett_reduce2(z, modulus, mod_inv_quotent2);
                return (ret);
            }




            else if constexpr (sizeof(T) == 4)
            {
                T ret;
// TO DO: force -fprecise model on clang
// otherwise fma does not work
#if 1 //defined(__clang_major__) 
                T z[2];
                z[0] = xehe::native::mul_uint<T>(operand1, operand2, z + 1);
                ret = barrett_reduce2(z, modulus, mod_inv_quotent2);
#else
    /*
    *
    Atention:
    For the code to work it has to be compiled with option
    -fp-model_precise in lnx
    and
    /fp:precise in windows.
    *
    */
                auto p = double(modulus);
                auto u = double(1) / p;
                auto x = double(operand1);
                auto y = double(operand2);
                auto h = x * y;
                auto l = std::fma(x, y, -h);
                auto b = h * u;
                // round
                auto c = double(T(b));
                auto d = std::fma(-c, p, h);
                auto g = d + l;
                g -= (g >= p) ? p : 0;
                g += (g < 0) ? p : 0;
                ret = T(g);
#endif
                return ret;
            }

        }

        // (op1*op2+op3[0:1]) mod [modulus]
        // correctness: 
        //   op1,op2 both < 63 bits for T=uint64_t
        //   op3[0:1]: 128 bits contaning both low and high bits
        //   op1*op2+op3[0:1] < modulus
        template<typename T>
        inline T mad_uint_mod(T op1, T op2, T *op3, T modulus, const T* mod_inv_quotent2)
        {
            T z[2];
            z[0] = mad_uint(op1, op2, op3, z+1);
            auto ret = barrett_reduce2(z, modulus, mod_inv_quotent2);
            return ret;
        }

        template<typename T>
        inline T mad_uint_mod(T op1, T op2, T op3, T modulus, const T* mod_inv_quotent2)
        {
            T z[2];
            z[0] = mad_uint(op1, op2, op3, z+1);
            auto ret = barrett_reduce2(z, modulus, mod_inv_quotent2);
            return ret;
        }


    }

}

#endif
