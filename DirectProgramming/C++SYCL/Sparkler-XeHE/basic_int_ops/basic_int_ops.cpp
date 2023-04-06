/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

/*
* 
Atention:
The code has to be compiled with option
-fp-model_precise in lnx
and
/fp:precise in windows.
*
*/

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <intrin.h>


#include "native/xe_uintarith.hpp"
#include "native/xe_uintarith_core.hpp"

template<typename T>
T random(T bound)
{
    auto rd0 = std::min(std::max(1, rand()), (RAND_MAX - 1));
    auto rd1 = double(rd0) / RAND_MAX;
    rd1 *= double(bound);
    T rd = T(rd1);
    return rd;
}

template<typename T>
void basic_ops(uint64_t test_size = 1000)
{
    if constexpr (sizeof(T) == 8)
    {
        std::cout << "64 BIT" << std::endl;

    }
    else if constexpr (sizeof(T) == 4)
    {
        std::cout << "32 BIT" << std::endl;
    }

    auto test_sz = test_size;
    T data_bound = (T(-1));
    T carry_in = 0;
    T carry_out = 0;
    T borrow_in = 0;
    T borrow_out = 0;
    T mod_bound = (T(-1) >> 1);


    std::vector<T> op(test_sz);
    std::vector<T> op1(test_sz);
    std::vector<T> op2(test_sz);
    std::vector<T> mod(test_sz);
    std::vector<T> op2_quotent_mod(test_sz);
    std::vector<T> inv_mod_quotent(test_sz);
    std::vector<T[2]> inv_mod_quotent2(test_sz);
    T res_low;
    T exp_res_low;
    T res_high;
    T exp_res_high;

    
    for (size_t i = 0; i < test_sz; i++)
    {
        T rd = random(mod_bound);
        T rd_op = random(data_bound);


        op[i] = rd_op;

        mod[i] = static_cast<T>(rd % mod_bound);
        rd = random(mod_bound);
        // operands < mod
        op1[i] = static_cast<T>(rd % mod[i]);
        rd = random(mod_bound);
        op2[i] = static_cast<T>(rd % mod[i]);

        // op2*2^BitCount/modulus

        op2_quotent_mod[i] = xehe::native::op_by_mod_inverse(op2[i], mod[i]);
        inv_mod_quotent[i] = xehe::native::mod_inverse1(mod[i]); // 2^64 / mod
        inv_mod_quotent2[i][0] = xehe::native::mod_inverse2(mod[i], inv_mod_quotent2[i] + 1);
    }
    bool right_stuff = true;


#if 1
    std::cout << "uint add." << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {
        /*
        *  carry has not been tested
        */
        carry_out =
            xehe::native::add_uint<T>(
                op1[i],
                op2[i],
                carry_in,
                &res_low);

        exp_res_low = op1[i] + op2[i];


        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }

    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

    std::cout << "uint sub." << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {
        /*
        *  carry has not been tested
        */
        borrow_out =
            xehe::native::sub_uint<T>(
                op1[i],
                op2[i],
                borrow_in,
                &res_low);

        exp_res_low = op1[i] - op2[i];


        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }



    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

#endif
    std::cout << "uint mul_high." << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {

        res_high = xehe::native::mul_uint_high<T>(op1[i], op2[i]);

        if constexpr (sizeof(T) == 8)
        {
            exp_res_low = _mul128(
                op1[i],
                op2[i],
                (int64_t*)&exp_res_high
            );


        }
        else if constexpr (sizeof(T) == 4)
        {
            uint64_t ret = uint64_t(op1[i]) * uint64_t(op2[i]);
            int shift = sizeof(T) * 8;
            exp_res_high = T(ret >> shift);


        }


        if (res_high != exp_res_high)
        {
            right_stuff = false;
        }

    }

    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }



    std::cout << "uint mul." << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {



        res_low = xehe::native::mul_uint<T>(op1[i], op2[i], &res_high);

        if constexpr (sizeof(T) == 8)
        {
            exp_res_low = _mul128(
                op1[i],
                op2[i],
                (int64_t*)&exp_res_high
            );

        }
        else if constexpr (sizeof(T) == 4)
        {
            uint64_t ret = uint64_t(op1[i]) * uint64_t(op2[i]);
            int shift = sizeof(T) * 8;
            auto temp = ((ret << shift) >> shift);
            exp_res_low = T(temp);
            exp_res_high = T(ret >> shift);

        }
        if (res_low != exp_res_low || res_high != exp_res_high)
        {
            right_stuff = false;
        }

    }

    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

#if 1

    std::cout << "uint mod(add)" << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {

        res_low =
            xehe::native::add_mod<T>(
                op1[i],
                op2[i],
                mod[i]);

        exp_res_low = (op1[i] + op2[i]) % mod[i];
        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }
    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

    std::cout << "uint mod(sub)" << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {

        res_low =
            xehe::native::sub_mod<T>(
                op1[i],
                op2[i],
                mod[i]);

        exp_res_low = (op1[i] - op2[i]);
        bool negative = false;
        if constexpr (sizeof(T) == 8)
        {
            negative = (int64_t(exp_res_low) < 0);
        }
        else if constexpr (sizeof(T) == 4)
        {
            negative = (int32_t(exp_res_low) < 0);
        }

        exp_res_low += (negative) ? mod[i] : 0;
        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }
    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

    std::cout << "uint mod(x * y_qoutent)" << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {
        res_low =
            xehe::native::mul_quotent_mod(
                op1[i],
                op2[i],
                mod[i],
                op2_quotent_mod[i]
                );

        if constexpr (sizeof(T) == 8)
        {
            exp_res_low = _mul128(
                op1[i],
                op2[i],
                (int64_t*)&exp_res_high);
        }
        else if constexpr (sizeof(T) == 4)
        {
            uint64_t ret = uint64_t(op1[i]) * uint64_t(op2[i]);
            int shift = sizeof(T) * 8;
            auto temp = ((ret << shift) >> shift);
            exp_res_low = T(temp);
            exp_res_high = T(ret >> shift);
        }

        T num[2]{ exp_res_low, exp_res_high };
        T quo[2];

        xehe::native::div_uint2<T>(num, mod[i], quo);
        exp_res_low = num[0];

        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }
    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

    std::cout << "uint mod(x)" << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {
        res_low =
            xehe::native::barrett_reduce(
                op[i],
                mod[i],
                inv_mod_quotent[i]);


        exp_res_low = op[i] % mod[i] ;

        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }
    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }

    std::cout << "uint mod(x2)" << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {
        // prepare T2 value
        T in[2];
        in[0] = xehe::native::mul_uint<T>(op1[i], op2[i], in+1);

        // reduction T2
        res_low =
            xehe::native::barrett_reduce2(
                in,
                mod[i],
                inv_mod_quotent2[i]);

        // generic reduction

        exp_res_low = xehe::native::mod_reduce_generic2<T>(in, mod[i]);

        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }
    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }


    std::cout << "uint mod(mul2(x*y))" << std::endl;
    right_stuff = true;
    for (size_t i = 0; i < test_sz && right_stuff; i++)
    {


        // mod(mul2)
        res_low = xehe::native::mul_mod(op1[i], op2[i], mod[i], inv_mod_quotent2[i]);


        // prepare T2 value
        T in[2];
        in[0] = xehe::native::mul_uint<T>(op1[i], op2[i], in + 1);
        exp_res_low = xehe::native::mod_reduce_generic2<T>(in, mod[i]);

        if (res_low != exp_res_low)
        {
            right_stuff = false;
        }
    }
    if (right_stuff)
    {
        std::cout << "passed!" << std::endl;
    }
    else
    {
        std::cout << "failed?" << std::endl;
    }
#endif

}


int main(int argc, char* argv[]){
	std::cout << "Basic INT ops spcification." << std::endl;
    basic_ops<uint64_t>(10000);
    basic_ops<uint32_t>();


#if 0
    uint64_t modulus = 10000001;

    uint64_t num2[2]{ 0, 1 };
    uint64_t quo2[2];
    xehe::native::div_uint2<uint64_t>(num2, modulus, quo2);

    uint64_t num3[3]{ 0, 1, 0 };
    uint64_t quo3[3];
    xehe::native::div_uint3<uint64_t>(num3, modulus, quo3);


    //printf_s("%#I64x * %#I64x = %#I64x%I64x\n", a, b, c, d);
    printf_s("%#I64x  %#I64x, %#I64x  %#I64x, %#I64x  %#I64x, %#I64x  %#I64x\n", 
        num2[0], num2[1], quo2[0], quo2[1], num3[0], num3[1], quo3[0], quo3[1]);
#endif

}