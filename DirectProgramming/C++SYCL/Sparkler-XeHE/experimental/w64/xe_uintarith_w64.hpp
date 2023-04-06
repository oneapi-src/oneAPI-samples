/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_UINTARITH_W64_HPP
#define XeHE_UINTARITH_W64_HPP

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <type_traits>
#include "xe_uintarith.hpp"

namespace xehe
{
    namespace native
    {

        template<>
        inline xehe::w64_t and_int(xehe::w64_t op1, xehe::w64_t op2)
        {
            xehe::w64_t ret;
            ret.part[0] = op1.part[0] & op2.part[0];
            ret.part[1] = op1.part[1] & op2.part[1];
            return ret;
        }

        // op1 >= op2? 
        template<>
        inline bool ge(xehe::w64_t op1, xehe::w64_t op2)
        {
            bool ret = (op1.part[1] < op2.part[1]) ? false : (op1.part[1] > op2.part[1]) ? true
                : (op1.part[0] >= op2.part[0]);
            return(ret);
        }
        // op1 != op2
        template<>
        inline bool ne(xehe::w64_t op1, xehe::w64_t op2)
        {
            bool ret = ((op1.part[1] != op2.part[1]) || (op1.part[0] != op2.part[0]));
            return(ret);
        }

        template<>
        inline xehe::w64_t right_shift(xehe::w64_t op, int shift)
        {
            xehe::w64_t ret;

            right_shift2<uint32_t>(op.part, shift, ret.part);            
            return ret;
        }


        template<>
        inline xehe::w64_t add_uint(
            xehe::w64_t op1, xehe::w64_t op2, xehe::w64_t carry_in,
            xehe::w64_t* result)
        {
            xehe::w64_t carry;
            auto carry_low = xehe::native::add_uint<uint32_t>(op1.part[0], op2.part[0], carry_in.part[0], &(result->part[0]));
            carry.part[0] = xehe::native::add_uint<uint32_t>(op1.part[1], op2.part[1], carry_low, &(result->part[1]));
            carry.part[1] = 0;
            return carry;
        }

        template<>
        inline xehe::w64_t add_uint(
            xehe::w64_t op1, xehe::w64_t op2,
            xehe::w64_t* result)
        {
            xehe::w64_t carry;
            auto carry_low = xehe::native::add_uint<uint32_t>(op1.part[0], op2.part[0], &(result->part[0]));
            carry.part[0] = xehe::native::add_uint<uint32_t>(op1.part[1], op2.part[1], carry_low, &(result->part[1]));
            carry.part[1] = 0;
            return carry;
        }


        template<>
        inline xehe::w64_t add_int(
            xehe::w64_t op1, xehe::w64_t op2)
        {
            xehe::w64_t ret;
            xehe::native::add_uint<xehe::w64_t>(op1, op2, &ret);
            return (ret);
        }

        template<>
        inline xehe::w64_t sub_uint(
            xehe::w64_t op1, xehe::w64_t op2,
            xehe::w64_t borrow_in, xehe::w64_t* result)
        {

            auto borrow_low = xehe::native::sub_uint<uint32_t>(op1.part[0], op2.part[0], borrow_in.part[0], &(result->part[0]));

            xehe::w64_t borrow;

            borrow.part[0] = xehe::native::sub_uint<uint32_t>(op1.part[1], op2.part[1], borrow_low, &(result->part[1]));
            borrow.part[1] = 0;
            return borrow;

        }

        template<>
        inline xehe::w64_t sub_uint(
            xehe::w64_t op1, xehe::w64_t op2,
            xehe::w64_t* result)
        {

            auto borrow_low = xehe::native::sub_uint<uint32_t>(op1.part[0], op2.part[0], &(result->part[0]));

            xehe::w64_t borrow;

            borrow.part[0] = xehe::native::sub_uint<uint32_t>(op1.part[1], op2.part[1], borrow_low, &(result->part[1]));
            borrow.part[1] = 0;
            return borrow;

        }

        template<>
        inline xehe::w64_t sub_int(
            xehe::w64_t op1, xehe::w64_t op2)
        {
            xehe::w64_t ret;

            xehe::native::sub_uint<xehe::w64_t>(op1, op2, &ret);
            return (ret);
        }

        template<>
        inline xehe::w64_t inc_uint(xehe::w64_t opd, xehe::w64_t* result)
        {
            xehe::w64_t one;
            one.part[1] = 0;
            one.part[0] = 1;
            return xehe::native::add_uint<xehe::w64_t>(opd, one,  result);
        }

        template<>
        inline xehe::w64_t dec_uint(xehe::w64_t opd, xehe::w64_t* result)
        {
            xehe::w64_t one;
            one.part[1] = 0;
            one.part[0] = 1;

            return xehe::native::sub_uint<xehe::w64_t>(opd, one, result);
        }

        template<>
        inline xehe::w64_t neg_uint(xehe::w64_t opd)
        {
            // Negation is equivalent to inverting bits and adding 1.

            xehe::w64_t one;
            one.part[1] = 0;
            one.part[0] = 1;


            xehe::w64_t ret;
            opd.part[0] = ~opd.part[0];
            opd.part[1] = ~opd.part[1];

            xehe::native::add_uint<xehe::w64_t>(opd, one, &ret);
            return (ret);
        }


        inline void mul_uint_internal_low(xehe::w64_t op1, xehe::w64_t op2, xehe::w64_t& right, xehe::w64_t& middle, xehe::w64_t& temp_sum, xehe::w64_t& carry1)
        {
            xehe::w64_t middle1, middle2;

            middle1.part[0] = xehe::native::mul_uint<uint32_t>(op1.part[1], op2.part[0], middle1.part + 1);

            middle2.part[0] = xehe::native::mul_uint<uint32_t>(op2.part[1], op1.part[0], middle2.part + 1);

            // middle
            carry1.part[0] = 0;
            carry1.part[1] = xehe::native::add_uint<uint32_t>(middle1.part, middle2.part, 2, middle.part);
            // right
            right.part[0] = xehe::native::mul_uint<uint32_t>(op1.part[0], op2.part[0], right.part + 1);
            middle1.part[1] = 0;
            middle1.part[0] = right.part[1];
            middle2.part[1] = 0;
            middle2.part[0] = middle.part[0];
            xehe::native::add_uint<xehe::w64_t>(middle1, middle2, &temp_sum);

        }


        inline void mul_uint_internal_high(xehe::w64_t op1, xehe::w64_t op2,
            xehe::w64_t& left, xehe::w64_t& middle, xehe::w64_t& right, xehe::w64_t& temp_sum)
        {
            xehe::w64_t carry1, left1;

            mul_uint_internal_low(op1, op2, right, middle, temp_sum, carry1);
            // left
            left1.part[0] = xehe::native::mul_uint<uint32_t>(op1.part[1], op2.part[1], left1.part + 1);
            xehe::native::add_uint<xehe::w64_t>(left1, carry1, &left);


        }



        template<>
        inline xehe::w64_t mul_uint(xehe::w64_t op1, xehe::w64_t op2, xehe::w64_t* prod_high)
        {
            xehe::w64_t left, middle, right, temp_sum;
            mul_uint_internal_high(op1, op2, left, middle, right, temp_sum);
            xehe::w64_t tmp1, tmp2;
            tmp1.part[1] = 0;
            tmp1.part[0] = middle.part[1];
            xehe::native::add_uint<xehe::w64_t>(left, tmp1, &tmp2);
            tmp1.part[1] = 0;
            tmp1.part[0] = temp_sum.part[1];
            // high product
            xehe::native::add_uint<xehe::w64_t>(tmp2, tmp1, prod_high);
            xehe::w64_t prod_low;
            prod_low.part[1] = temp_sum.part[0];
            prod_low.part[0] = right.part[0];
            return(prod_low);
        }

        
        inline xehe::w128_t mul_uint(xehe::w64_t op1, xehe::w64_t op2)
        {
            xehe::w64_t left, middle, right, temp_sum;
            mul_uint_internal_high(op1, op2, left, middle, right, temp_sum);
            xehe::w64_t tmp1, tmp2;
            tmp1.part[1] = 0;
            tmp1.part[0] = middle.part[1];
            xehe::native::add_uint<xehe::w64_t>(left, tmp1, &tmp2);
            tmp1.part[1] = 0;
            tmp1.part[0] = temp_sum.part[1];
            xehe::w128_t ret;
            // high product
            xehe::native::add_uint<xehe::w64_t>(tmp2, tmp1, &ret.part[1]);
            
            // low product
            ret.part[0].part[1] = temp_sum.part[0];
            ret.part[0].part[0] = right.part[0];
            return(ret);
        }


        template<>
        inline xehe::w64_t mul_uint_high(xehe::w64_t op1, xehe::w64_t op2)
        {
            xehe::w64_t ret, left, middle, right, temp_sum;

            mul_uint_internal_high(op1, op2, left, middle, right, temp_sum);

            xehe::w64_t tmp1, tmp2;
            tmp1.part[1] = 0;
            tmp1.part[0] = middle.part[1];
            xehe::native::add_uint<uint32_t>(left.part, tmp1.part, 2, tmp2.part);
            tmp1.part[1] = 0;
            tmp1.part[0] = temp_sum.part[1];
            // high product
            xehe::native::add_uint<uint32_t>(tmp2.part, tmp1.part, 2, ret.part);
            return(ret);

        }

        template<>
        inline xehe::w64_t mul_uint_low(xehe::w64_t op1, xehe::w64_t op2)
        {
            xehe::w64_t ret, right, middle, temp_sum, carry1;

            mul_uint_internal_low(op1, op2, right, middle, temp_sum, carry1);
            ret.part[1] = temp_sum.part[0];
            ret.part[0] = right.part[0];
            return(ret);
        }


        /******************************************************************************************
        *
        * Correctness conditions:
        * operand1,operand2 < mod
        * bit_count(mod) <= (bit_size(T) - 1)
        *
        * ****************************************************************************************/



        template<>
        inline xehe::w64_t inc_mod(xehe::w64_t operand, xehe::w64_t modulus)
        {
            xehe::w64_t ret;
            auto carry = xehe::native::inc_uint(operand, &ret);
            if (carry.part[0] || ge(ret, modulus))
            {
                xehe::native::sub_uint(ret, modulus, &ret);
            }

            return(ret);
        }

        template<>
        inline xehe::w64_t  dec_mod(xehe::w64_t  operand, xehe::w64_t  modulus)
        {
            xehe::w64_t  ret;
            auto carry = xehe::native::dec_uint(operand, &ret);
            if (carry.part[0])
            {
                xehe::native::add_uint(ret, modulus, &ret);
            }

            return(ret);
        }

        template<>
        inline xehe::w64_t  neg_mod(xehe::w64_t  operand, xehe::w64_t  modulus)
        {

            xehe::w64_t  ret;
            ret.part[0] = ret.part[1] = 0;


            // operand != 0
            if (ne(operand, ret))
            {
                // operand > 0 and < modulus so subtract modulus - operand.
                xehe::native::sub_uint(modulus, operand, &ret);
            }
            
            return(ret);
        }

        template<>
        inline xehe::w64_t div2_mod(xehe::w64_t operand, xehe::w64_t modulus)
        {
            xehe::w64_t ret;
            ret.part[0] = ret.part[1] = 0;
            if (operand.part[0] & 1)
            {
                auto carry = xehe::native::add_uint(operand, modulus, &ret);
                ret = xehe::native::right_shift<xehe::w64_t>(ret, 1);
                if (carry.part[0])
                {
                    ret.part[1] |= 0x80000000;
                }
            }
            else
            {
                ret = xehe::native::right_shift<xehe::w64_t>(operand, 1);
            }

            return(ret);
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


        template<>
        inline xehe::w64_t add_mod(const xehe::w64_t operand1, const xehe::w64_t operand2, const xehe::w64_t modulus)
        {

            xehe::w64_t ret, temp, temp1;
            ret.part[0] = ret.part[1] = 0;

            // Sum of operands % mod can never wrap around 2^NumOfBits(T)
            xehe::native::add_uint<xehe::w64_t>(operand1, operand2, &temp);
           
            temp1.part[0] = uint32_t(-static_cast<int>(ge(temp,modulus)));
            // extend
            temp1.part[1] = temp1.part[0];
            temp1 = xehe::native::and_int<xehe::w64_t>(temp1,modulus);
            xehe::native::sub_uint<xehe::w64_t>(temp, temp1, &ret);
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
        template<>
        inline xehe::w64_t sub_mod(xehe::w64_t  operand1, xehe::w64_t  operand2, xehe::w64_t  modulus)
        {
            xehe::w64_t ret;
            ret.part[0] = ret.part[1] = 0;
            xehe::w64_t  temp;

            auto borrow = xehe::native::sub_uint<xehe::w64_t>(operand1, operand2, &temp);
            // ret = 0 - borrow
            //ret.part[0] = uint32_t(-int(borrow.part[0]));
            //ret.part[1] = ret.part[0];
            xehe::native::sub_uint<xehe::w64_t>(ret, borrow, &ret);
            // ret &= modulus
            ret = xehe::native::and_int<xehe::w64_t>(ret, modulus);
            // ret = (op1-op2) + 0 or modulus
            ret = xehe::native::add_int<xehe::w64_t>(temp, ret);
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
        template <>
        inline xehe::w64_t mul_quotent_mod(xehe::w64_t x, xehe::w64_t y, xehe::w64_t modulus, xehe::w64_t y_mod_quotent)
        {
            xehe::w64_t ret;
            auto tmp1 = xehe::native::mul_uint_high(x, y_mod_quotent);
            tmp1 = xehe::native::mul_uint_low(tmp1, modulus);
            auto tmp2 = xehe::native::mul_uint_low(y, x);
            tmp2 = xehe::native::sub_int<xehe::w64_t>(tmp2, tmp1);
            w64_t tmp3;
            tmp3.part[0] = uint32_t(-static_cast<int>(ge(tmp2, modulus)));
            // extend
            tmp3.part[1] = tmp3.part[0];
            auto p = modulus;
            p = xehe::native::and_int<xehe::w64_t>(p, tmp3);
            ret = xehe::native::sub_int<xehe::w64_t>(tmp2, p);
            return (ret);
        }



    }
}

#endif
