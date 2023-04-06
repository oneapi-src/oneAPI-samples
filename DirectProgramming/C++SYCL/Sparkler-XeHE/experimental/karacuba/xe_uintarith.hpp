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
#include "xe_uintarith_base.hpp"

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
            auto b_ne_0 = T(borrow != 0);
            *result = xehe::native::sub_int<T>(diff, b_ne_0);
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


        template<typename T>
        inline void mul_uint_internal(T operand1, T operand2, T& left, T& middle, T& right, T& temp_sum, T mask, int shift)
        {
            auto operand1_coeff_right = operand1 & mask;
            auto operand2_coeff_right = operand2 & mask;
            T tmp1 = operand1;
            tmp1 = xehe::native::right_shift(tmp1, shift);
            operand1 = T(tmp1);
            T tmp2 = operand2;
            tmp2 = xehe::native::right_shift(tmp2, shift);
            operand2 = T(tmp2);

            auto middle1 = xehe::native::mul_uint_low<T>(operand1, operand2_coeff_right);

            auto middle_temp = xehe::native::mul_uint_low<T>(operand2, operand1_coeff_right);
            T carry = xehe::native::add_uint<T>(middle1, middle_temp, &middle);
            left = xehe::native::add_int<T>(xehe::native::mul_uint_low<T>(operand1, operand2), xehe::native::left_shift(carry, shift));
            right = xehe::native::mul_uint_low<T>(operand1_coeff_right, operand2_coeff_right);
            temp_sum = xehe::native::add_int<T>(xehe::native::right_shift<T>(right, shift), (middle & mask));
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
            T middle;


            auto t1 = xehe::native::add_int<T>(op1_low, op1_high);
            auto t2 = xehe::native::add_int<T>(op2_low, op2_high);
            // WATCH FOT THIS MUL
            auto z1 = xehe::native::mul_uint_low<T>(t1, t2);
            // Z1 - Z2
            auto t3 = xehe::native::sub_int<T>(z1, left);
            // -Z0
            middle = xehe::native::sub_int<T>(t3, right);
            auto middle_low = xehe::native::left_shift<T>(middle, shift);

            T prod_low;
            auto carry1 = xehe::native::add_uint<T>(right, middle_low, &prod_low);

            auto middle_high = xehe::native::right_shift<T>(middle, shift);
            xehe::native::add_uint<T>(left, middle_high, carry1, prod_high);

            return(prod_low);
        }

        template<typename T>
        inline T mul_uint_kara_diff(T op1_low, T op1_high, T op2_low, T op2_high, int shift, T* prod_high)
        {
            auto right = xehe::native::mul_uint_low<T>(op1_low, op2_low);
            auto left = xehe::native::mul_uint_low<T>(op1_high, op2_high);

            xehe::w64_t t1, t2;
            t1.value64b = xehe::native::sub_int<T>(op1_low, op1_high);
            t2.value64b = xehe::native::sub_int<T>(op2_high, op2_low);
            const uint32_t sign_mask = 0x80000000;
            auto sign1 = (t1.high & sign_mask);
            auto sign2 = (t2.high & sign_mask);
            // abs values
// ATTENTION BUG:
// on windows kernel compiler failed if  not abs<int64_t>()
// on lnx and MSVC17 on windows does not aloow the specialization and aks for abs() 
// This is work around
#if defined(__clang_major__) && defined(WIN32) 
            t1.value64b = T(std::abs<int64_t>(int64_t(t1.value64b))); // (sign1) ? T(-int64_t(t1.value64b)) : t1.value64b;
            t2.value64b = T(std::abs<int64_t>(int64_t(t2.value64b))); // (sign2) ? T(-int64_t(t2.value64b)) : t2.value64b;
#else
            t1.value64b = T(std::abs(int64_t(t1.value64b))); // (sign1) ? T(-int64_t(t1.value64b)) : t1.value64b;
            t2.value64b = T(std::abs(int64_t(t2.value64b))); // (sign2) ? T(-int64_t(t2.value64b)) : t2.value64b;

#endif
            // WATCH FOT THIS MUL
            xehe::w64_t z1;
            z1.value64b = xehe::native::mul_uint_low<T>(t1.value64b, t2.value64b);
            // restore sign
            auto new_sign = ((sign1 ^ sign2) & sign_mask);

            z1.value64b = (new_sign)? T(-int64_t(z1.value64b)) : z1.value64b;
            // Z1 + Z2
            xehe::w64_t t3;
            auto carry0 = xehe::native::add_uint<T>(z1.value64b, left, &t3.value64b);
            carry0 = (z1.high & sign_mask) ? 0 : carry0;
            //t3.value64b = xehe::native::add_int<T>(z1.value64b, left);
            // + Z0
            T middle;
            carry0 = xehe::native::add_uint<T>(t3.value64b, right, carry0 , &middle);
            carry0 = (t3.high & sign_mask) ? 0 : carry0;
            //middle = xehe::native::add_int<T>(t3, right);
            //auto middle_low = xehe::native::left_shift<T>(middle, shift);
            auto middle_low = uint64_t(int64_t(middle) << shift);

            T prod_low;
            auto carry1 = xehe::native::add_uint<T>(right, middle_low, &prod_low);
            carry1 += (carry0 << shift);
            //auto middle_high = xehe::native::right_shift<T>(middle, shift);
            auto middle_high = T(int64_t(middle) >> shift);
            xehe::native::add_uint<T>(left, middle_high, carry1, prod_high);

            return(prod_low);
        }


        template<typename T>
        inline T mul_uint_kara2(T left, T right, T t1, T t2,  int shift, T* prod_high)
        {
            T middle;

            // WATCH FOT THIS MUL
            auto z1 = xehe::native::mul_uint_low<T>(t1, t2);
            // Z1 - Z2
            auto t3 = xehe::native::sub_int<T>(z1, left);
            // -Z0
            middle = xehe::native::sub_int<T>(t3, right);
            auto middle_low = xehe::native::left_shift<T>(middle, shift);

            T prod_low;
            auto carry1 = xehe::native::add_uint<T>(right, middle_low, &prod_low);

            auto middle_high = xehe::native::right_shift<T>(middle, shift);
            xehe::native::add_uint<T>(left, middle_high, carry1, prod_high);

            return(prod_low);
        }



        template<typename T>
        inline T mul_uint_intnl(T op1_low, T op1_high, T op2_low, T op2_high, int shift, T* prod_high)
        {
            auto right = xehe::native::mul_uint_low<T>(op1_low, op2_low);
            auto left = xehe::native::mul_uint_low<T>(op1_high, op2_high);
            T middle;
 
            auto h1_l2 = xehe::native::mul_uint_low<T>(op1_high, op2_low);
            auto l1_h2 = xehe::native::mul_uint_low<T>(op1_low, op2_high);

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
        inline T mul_uint_intnl2(T op1_low, T op1_high, T op2_low, T op2_high, T left, T right, int shift, T* prod_high)
        {
            T middle;
            T middle_carry = 0;
            auto h1_l2 = xehe::native::mul_uint_low<T>(op1_high, op2_low);
            auto l1_h2 = xehe::native::mul_uint_low<T>(op1_low, op2_high);
            middle_carry = xehe::native::add_uint<T>(h1_l2, l1_h2, &middle);

            auto middle_low = xehe::native::left_shift<T>(middle, shift);

            // prod low
            T prod_low;
            auto carry1 = xehe::native::add_uint<T>(right, middle_low, &prod_low);

            // prod high
            middle_carry = xehe::native::left_shift<T>(middle_carry, shift);
            carry1 = xehe::native::add_int<T>(carry1, middle_carry);
            auto middle_high = xehe::native::right_shift<T>(middle, shift);
            xehe::native::add_uint<T>(left, middle_high, carry1, prod_high);

            return(prod_low);
        }

#if 1
        template<typename T>
        inline T mul_uint(T op1, T op2, T* prod_high)
        {

            T ret = 0;

            if constexpr (sizeof(T) == 4)
            {

                auto ret64 = xehe::native::mul_uint_low<uint64_t>(uint64_t(op1), uint64_t(op2));
                ret = T(ret64);
                *prod_high = *((T*)&ret64 + 1);
            }
            else  if constexpr (sizeof(T) == 8)
            {
                auto mask = T(0x00000000FFFFFFFF);
                auto shift = int(32);
                auto op1_low = op1 & mask;
                auto op2_low = op2 & mask;
                auto op1_high = xehe::native::right_shift(op1, shift);
                auto op2_high = xehe::native::right_shift(op2, shift);

                //ret = mul_uint_kara_diff<T>(op1_low, op1_high, op2_low, op2_high, shift, prod_high);
                ret = mul_uint_intnl<T>(op1_low, op1_high, op2_low, op2_high, shift, prod_high);             
            }
            return(ret);
        }

#else
        template<typename T>
        inline T mul_uint(T op1, T op2, T* prod_high)
        {
            // old mul
            T ret = 0;
            T left = 0, middle = 0, right = 0, temp_sum = 0;
            mul_uint_internal<T>(op1, op2, left, middle, right, temp_sum, mask, shift);

            auto t1 = xehe::native::right_shift<T>(middle, shift);
            auto t2 = xehe::native::add_int<T>(left, t1);
            auto t3 = xehe::native::right_shift <T>(temp_sum, shift);
            *prod_high = xehe::native::add_int<T>(t2, t3);

            t1 = xehe::native::left_shift<T>(temp_sum, shift);
            ret = (t1 | (right & mask));

            return(ret);
        }
#endif

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

            // operand > 0 and < modulus so subtract modulus - operand.
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


         /**
         mod(add). returns mod of sum of 2 uint values.
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
            // Sum of operands % mod can never wrap around 2^NumOfBits(T)
            auto temp0 = xehe::native::add_int<T>(operand1, operand2);
            auto t = -static_cast<int>(xehe::native::ge<T>(temp0, modulus));
            auto temp1 = static_cast<T>(t);
            temp1 = xehe::native::bit_and_int<T>(temp1,modulus);
            ret = xehe::native::sub_int<T>(temp0,temp1);
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
#if defined(__clang_major__) 
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

    }

}

#endif
