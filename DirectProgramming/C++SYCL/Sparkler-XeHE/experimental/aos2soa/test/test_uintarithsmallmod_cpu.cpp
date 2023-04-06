// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch2/catch.hpp"
#include <vector>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <cstdint>

#ifdef BUILD_WITH_SEAL
#include "seal/util/defines.h"
#include "seal/modulus.h"
#endif

#include "util/common.h"
#include "util/xe_uintarith.h"
#include "util/xe_uintarithmod.h"
#include "util/uintarithsmallmod.h"
#include "util/xe_uintcore.h"


using namespace xehe::util;
using namespace xehe;
using namespace std;

namespace xehetest
{
    namespace util
    {
#if 0
        TEST(UIntArithSmallMod, IncrementUIntMod)
        {
            Modulus mod(2);
            ASSERT_EQ(1ULL, increment_uint_mod(0, mod));
            ASSERT_EQ(0ULL, increment_uint_mod(1ULL, mod));

            mod = 0x10000;
            ASSERT_EQ(1ULL, increment_uint_mod(0, mod));
            ASSERT_EQ(2ULL, increment_uint_mod(1ULL, mod));
            ASSERT_EQ(0ULL, increment_uint_mod(0xFFFFULL, mod));

            mod = 2305843009211596801ULL;
            ASSERT_EQ(1ULL, increment_uint_mod(0, mod));
            ASSERT_EQ(0ULL, increment_uint_mod(2305843009211596800ULL, mod));
            ASSERT_EQ(1ULL, increment_uint_mod(0, mod));
        }

        TEST(UIntArithSmallMod, DecrementUIntMod)
        {
            Modulus mod(2);
            ASSERT_EQ(0ULL, decrement_uint_mod(1, mod));
            ASSERT_EQ(1ULL, decrement_uint_mod(0ULL, mod));

            mod = 0x10000;
            ASSERT_EQ(0ULL, decrement_uint_mod(1, mod));
            ASSERT_EQ(1ULL, decrement_uint_mod(2ULL, mod));
            ASSERT_EQ(0xFFFFULL, decrement_uint_mod(0ULL, mod));

            mod = 2305843009211596801ULL;
            ASSERT_EQ(0ULL, decrement_uint_mod(1, mod));
            ASSERT_EQ(2305843009211596800ULL, decrement_uint_mod(0ULL, mod));
            ASSERT_EQ(0ULL, decrement_uint_mod(1, mod));
        }
#endif
        template <typename T>
        void UIntArithSmallMod_NegateUIntMod(void)
        //TEST(UIntArithSmallMod, NegateUIntMod)
        {
            T mod(2);
            REQUIRE(T(0) == negate_uint_mod(T(0), mod));
            REQUIRE(T(1) == negate_uint_mod(T(1), mod));

            mod = T(0xFFFF);
            REQUIRE(T(0) == negate_uint_mod(T(0), mod));
            REQUIRE(T(0xFFFE) == negate_uint_mod(T(1), mod));
            REQUIRE(T(0x1) == negate_uint_mod(T(0xFFFE), mod));

            mod = T(0x10000);
            REQUIRE(T(0) == negate_uint_mod(T(0), mod));
            REQUIRE(T(0xFFFF) == negate_uint_mod(T(1), mod));
            REQUIRE(T(0x1) == negate_uint_mod(T(0xFFFF), mod));

            mod = T(2305843009211596801ULL);
            REQUIRE(T(0) == negate_uint_mod(T(0), mod));
            REQUIRE(T(2305843009211596800ULL) == negate_uint_mod(T(1), mod));
        }

        TEST_CASE("XeHE UIntArithSmallMod NegateUIntMod", "[NegateUIntMod][cpu][XeHE]")
        {

            UIntArithSmallMod_NegateUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod NegateUIntMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_NegateUIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod NegateUIntMod32 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_Div2UIntMod(void)
        //TEST(UIntArithSmallMod, Div2UIntMod)
        {
            T mod(3);
            REQUIRE(T(0) == div2_uint_mod(T(0), mod));
            REQUIRE(T(2) == div2_uint_mod(T(1), mod));

            mod = T(17);
            REQUIRE(T(11) == div2_uint_mod(T(5), mod));
            REQUIRE(T(4) == div2_uint_mod(T(8), mod));

            mod = T(0xFFFFFFFFFFFFFFFULL);
            REQUIRE(T(0x800000000000000ULL) == div2_uint_mod(T(1), mod));
            REQUIRE(T(0x800000000000001ULL) == div2_uint_mod(T(3), mod));
        }

        TEST_CASE("XeHE UIntArithSmallMod Div2UIntMod", "[Div2UIntMod][cpu][XeHE]")
        {

            UIntArithSmallMod_Div2UIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod Div2UIntMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_Div2UIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod Div2UIntMod32 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_AddUIntMod(void)
        //TEST(UIntArithSmallMod, AddUIntMod)
        {
            T mod(2);
            REQUIRE(T(0) == add_uint_mod(T(0), T(0), mod));
            REQUIRE(T(1) == add_uint_mod(T(0), T(1), mod));
            REQUIRE(T(1) == add_uint_mod(T(1), T(0), mod));
            REQUIRE(T(0) == add_uint_mod(T(1), T(1), mod));

            mod = T(10);
            REQUIRE(T(0) == add_uint_mod(T(0), T(0), mod));
            REQUIRE(T(1) == add_uint_mod(T(0), T(1), mod));
            REQUIRE(T(1) == add_uint_mod(T(1), T(0), mod));
            REQUIRE(T(2) == add_uint_mod(T(1), T(1), mod));
            REQUIRE(T(4) == add_uint_mod(T(7), T(7), mod));
            REQUIRE(T(3) == add_uint_mod(T(6), T(7), mod));

            mod = T(2305843009211596801ULL);
            REQUIRE(T(0) == add_uint_mod(T(0), T(0), mod));
            REQUIRE(T(1) == add_uint_mod(T(0), T(1), mod));
            REQUIRE(T(1) == add_uint_mod(T(1), T(0), mod));
            REQUIRE(T(2) == add_uint_mod(T(1), T(1), mod));
            REQUIRE(T(0) == add_uint_mod(T(1152921504605798400ULL), T(1152921504605798401ULL), mod));
            REQUIRE(T(1) == add_uint_mod(T(1152921504605798401ULL), T(1152921504605798401ULL), mod));
            REQUIRE(T(2305843009211596799ULL) == add_uint_mod(T(2305843009211596800ULL), T(2305843009211596800ULL), mod));
        }

        TEST_CASE("XeHE UIntArithSmallMod AddUIntMod", "[AddUIntMod][cpu][XeHE]")
        {

            UIntArithSmallMod_AddUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod AddUIntMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_AddUIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod AddUIntMod32 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_SubUIntMod(void)
        //TEST(UIntArithSmallMod, SubUIntMod)
        {
            T mod(2);
            REQUIRE(T(0) == sub_uint_mod(T(0), T(0), mod));
            REQUIRE(T(1) == sub_uint_mod(T(0), T(1), mod));
            REQUIRE(T(1) == sub_uint_mod(T(1), T(0), mod));
            REQUIRE(T(0) == sub_uint_mod(T(1), T(1), mod));

            mod = T(10);
            REQUIRE(T(0) == sub_uint_mod(T(0), T(0), mod));
            REQUIRE(T(9) == sub_uint_mod(T(0), T(1), mod));
            REQUIRE(T(1) == sub_uint_mod(T(1), T(0), mod));
            REQUIRE(T(0) == sub_uint_mod(T(1), T(1), mod));
            REQUIRE(T(0) == sub_uint_mod(T(7), T(7), mod));
            REQUIRE(T(9) == sub_uint_mod(T(6), T(7), mod));
            REQUIRE(T(1) == sub_uint_mod(T(7), T(6), mod));

            mod = T(2305843009211596801ULL);
            REQUIRE(T(0) == sub_uint_mod(T(0), T(0), mod));
            REQUIRE(T(2305843009211596800ULL) == sub_uint_mod(T(0), T(1), mod));
            REQUIRE(T(1) == sub_uint_mod(T(1), T(0), mod));
            REQUIRE(T(0) == sub_uint_mod(T(1), T(1), mod));
            REQUIRE(T(2305843009211596800ULL) == sub_uint_mod(T(1152921504605798400ULL), T(1152921504605798401ULL), mod));
            REQUIRE(T(1) == sub_uint_mod(T(1152921504605798401ULL), T(1152921504605798400ULL), mod));
            REQUIRE(T(0) == sub_uint_mod(T(1152921504605798401ULL), T(1152921504605798401ULL), mod));
            REQUIRE(T(0) == sub_uint_mod(T(2305843009211596800ULL), T(2305843009211596800ULL), mod));
        }

        TEST_CASE("XeHE UIntArithSmallMod SubUIntMod", "[SubUIntMod][cpu][XeHE]")
        {

            UIntArithSmallMod_SubUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod SubUIntMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_SubUIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod SubUIntMod32 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_BarrettReduce128(void)
        //TEST(UIntArithSmallMod, BarrettReduce128)
        {

            T input[2];
            T mod(2);
            const std::uint64_t* const_ratio = nullptr;
            {
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(2);
                if (sizeof(T) == 8)
                {
                    mod = seal_mod.value();
                    const_ratio = seal_mod.const_ratio().data();
                }
#endif


                input[0] = 0;
                input[1] = 0;
                REQUIRE(0ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 1;
                input[1] = 0;
                REQUIRE(1ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 0xFFFFFFFFFFFFFFFFULL;
                input[1] = 0xFFFFFFFFFFFFFFFFULL;
                REQUIRE(1ULL == barrett_reduce_128(input, mod, const_ratio));
            }
            {

#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(3);
                if (sizeof(T) == 8)
                {
                    mod = seal_mod.value();
                    const_ratio = seal_mod.const_ratio().data();
                }
#endif
                input[0] = 0;
                input[1] = 0;
                REQUIRE(0ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 1;
                input[1] = 0;
                REQUIRE(1ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 123;
                input[1] = 456;
                REQUIRE(0ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 0xFFFFFFFFFFFFFFFFULL;
                input[1] = 0xFFFFFFFFFFFFFFFFULL;
                REQUIRE(0ULL == barrett_reduce_128(input, mod, const_ratio));
            }
            {

#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(13131313131313ULL);
                if (sizeof(T) == 8)
                {
                    mod = seal_mod.value();
                    const_ratio = seal_mod.const_ratio().data();
                }
#endif

                input[0] = 0;
                input[1] = 0;
                REQUIRE(0ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 1;
                input[1] = 0;
                REQUIRE(1ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 123;
                input[1] = 456;
                REQUIRE(8722750765283ULL == barrett_reduce_128(input, mod, const_ratio));
                input[0] = 24242424242424;
                input[1] = 79797979797979;
                REQUIRE(1010101010101ULL == barrett_reduce_128(input, mod, const_ratio));
            }
        }

        TEST_CASE("XeHE UIntArithSmallMod BarrettReduce128", "[BarrettReduce128][cpu][XeHE]")
        {

            UIntArithSmallMod_BarrettReduce128<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod BarrettReduce12864 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyUIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyUInt32 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_MultiplyUIntMod(void)
        //TEST(UIntArithSmallMod, MultiplyUIntMod)
        {
            T mod(2);
            const T* const_ratio = nullptr;
            {
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(2);
                if (sizeof(T) == 8)
                {
                    mod = seal_mod.value();
                    const_ratio = seal_mod.const_ratio().data();
                }
#endif
                REQUIRE(T(0) == multiply_uint_mod<T>(0, 0, mod, const_ratio));
                REQUIRE(T(0) == multiply_uint_mod<T>(0, 1, mod, const_ratio));
                REQUIRE(T(0) == multiply_uint_mod<T>(1, 0, mod, const_ratio));
                REQUIRE(T(1) == multiply_uint_mod<T>(1, 1, mod, const_ratio));
            }
            {
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(10);
                if (sizeof(T) == 8)
                {
                    mod = seal_mod.value();
                    const_ratio = seal_mod.const_ratio().data();
                }
#endif
                REQUIRE(T(0) == multiply_uint_mod<T>(0, 0, mod, const_ratio));
                REQUIRE(T(0) == multiply_uint_mod<T>(0, 1, mod, const_ratio));
                REQUIRE(T(0) == multiply_uint_mod<T>(1, 0, mod, const_ratio));
                REQUIRE(T(1) == multiply_uint_mod<T>(1, 1, mod, const_ratio));
                REQUIRE(T(9) == multiply_uint_mod<T>(7, 7, mod, const_ratio));
                REQUIRE(T(2) == multiply_uint_mod<T>(6, 7, mod, const_ratio));
                REQUIRE(T(2) == multiply_uint_mod<T>(7, 6, mod, const_ratio));
            }

            if (sizeof(T) == 8)
            {
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(2305843009211596801ULL);
                mod = seal_mod.value();
                const_ratio = seal_mod.const_ratio().data();
#endif
                REQUIRE(T(0) == multiply_uint_mod<T>(0, 0, mod, const_ratio));
                REQUIRE(T(0) == multiply_uint_mod<T>(0, 1, mod, const_ratio));
                REQUIRE(T(0) == multiply_uint_mod<T>(1, 0, mod, const_ratio));
                REQUIRE(T(1) == multiply_uint_mod<T>(1, 1, mod, const_ratio));
                REQUIRE(T(576460752302899200) == multiply_uint_mod<T>(T(1152921504605798400), T(1152921504605798401), mod, const_ratio));
                REQUIRE(T(576460752302899200) == multiply_uint_mod<T>(T(1152921504605798401), T(1152921504605798400), mod, const_ratio));
                REQUIRE(T(1729382256908697601) == multiply_uint_mod<T>(T(1152921504605798401), T(1152921504605798401), mod, const_ratio));
                REQUIRE(T(1) == multiply_uint_mod<T>(T(2305843009211596800), T(2305843009211596800), mod, const_ratio));

            }
        }

        TEST_CASE("XeHE UIntArithSmallMod MultiplyUIntMod", "[MultiplyUIntMod][cpu][XeHE]")
        {

            UIntArithSmallMod_MultiplyUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyUIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntMod32 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_MultiplyAddMod(void)
        //TEST(UIntArithSmallMod, MultiplyAddMod)
        {
            T mod(7);
            const T* const_ratio = nullptr;
            {
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(mod);
                const_ratio = seal_mod.const_ratio().data();        
#endif

                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(0), T(0), T(0), mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(1), T(0), T(0), mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(0), T(1), T(0), mod, const_ratio));
                REQUIRE(T(1) == multiply_add_uint_mod<T>(T(0), T(0), T(1), mod, const_ratio));
                REQUIRE(T(3) == multiply_add_uint_mod<T>(T(3), T(4), T(5), mod, const_ratio));
            }


            if (sizeof(T) == 8)
            {
                mod = 0x1FFFFFFFFFFFFFFFULL;
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(mod);
                const_ratio = seal_mod.const_ratio().data();
#endif

                REQUIRE(T(0) == multiply_add_uint_mod(T(0), T(0), T(0), mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod(T(1), T(0), T(0), mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod(T(0), T(1), T(0), mod, const_ratio));
                REQUIRE(T(1) == multiply_add_uint_mod(T(0), T(0), T(1), mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod(mod - 1, mod - 1, mod - 1, mod, const_ratio));
            }
        }

        TEST_CASE("XeHE UIntArithSmallMod MultiplyAddMod", "[MultiplyAddMod][cpu][XeHE]")
        {

            UIntArithSmallMod_MultiplyAddMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod MultiplyAddMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyAddMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyAddMod32 tests passed-------" << std::endl;
        }


        template <typename T>
        void UIntArithSmallMod_ModuloUIntMod(void)
        //TEST(UIntArithSmallMod, ModuloUIntMod)
        {
            auto s_value = allocate_uint<T>(4);
            auto value = s_value.get(); 
            const T* const_ratio = nullptr;
            T mod(2);
            {

#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(mod);
                const_ratio = seal_mod.const_ratio().data();
#endif

                value[0] = 0;
                value[1] = 0;
                value[2] = 0;
                modulo_uint_inplace(value, 3, mod, const_ratio);
                REQUIRE(T(0) == value[0]);
                REQUIRE(T(0) == value[1]);
                REQUIRE(T(0) == value[2]);

                value[0] = 1;
                value[1] = 0;
                value[2] = 0;
                modulo_uint_inplace(value, 3, mod, const_ratio);
                REQUIRE(T(1) == value[0]);
                REQUIRE(T(0) == value[1]);
                REQUIRE(T(0) == value[2]);

                value[0] = 2;
                value[1] = 0;
                value[2] = 0;
                modulo_uint_inplace(value, 3, mod, const_ratio);
                REQUIRE(T(0) == value[0]);
                REQUIRE(T(0) == value[1]);
                REQUIRE(T(0) == value[2]);

                value[0] = 3;
                value[1] = 0;
                value[2] = 0;
                modulo_uint_inplace(value, 3, mod, const_ratio);
                REQUIRE(T(1) == value[0]);
                REQUIRE(T(0) == value[1]);
                REQUIRE(T(0) == value[2]);
            }



            if (sizeof(T) == 8)
            {
                mod = 0xFFFF;

                {
#ifdef BUILD_WITH_SEAL
                    seal::Modulus seal_mod(mod);
                    const_ratio = seal_mod.const_ratio().data();
#endif

                    value[0] = 9585656442714717620ul;
                    value[1] = 1817697005049051848;
                    value[2] = 0;
                    modulo_uint_inplace(value, 3, mod, const_ratio);
                    REQUIRE(T(65143) == value[0]);
                    REQUIRE(T(0) == value[1]);
                    REQUIRE(T(0) == value[2]);
                }

                mod = 0x1000;

                {

#ifdef BUILD_WITH_SEAL
                    seal::Modulus seal_mod(mod);
                    const_ratio = seal_mod.const_ratio().data();
#endif
                    value[0] = 9585656442714717620ul;
                    value[1] = 1817697005049051848;
                    value[2] = 0;
                    modulo_uint_inplace(value, 3, mod, const_ratio);
                    REQUIRE(T(0xDB4) == value[0]);
                    REQUIRE(T(0) == value[1]);
                    REQUIRE(T(0) == value[2]);
                }

                mod = 0xFFFFFFFFC001ULL;

                {




#ifdef BUILD_WITH_SEAL
                    seal::Modulus seal_mod(mod);
                    const_ratio = seal_mod.const_ratio().data();
#endif
                    value[0] = 9585656442714717620ul;
                    value[1] = 1817697005049051848;
                    value[2] = 14447416709120365380ul;
                    value[3] = 67450014862939159;
                    modulo_uint_inplace(value, 4, mod, const_ratio);
                    REQUIRE(T(124510066632001ULL) == value[0]);
                    REQUIRE(T(0) == value[1]);
                    REQUIRE(T(0) == value[2]);
                    REQUIRE(T(0) == value[3]);
                }
            }
        }

        TEST_CASE("XeHE UIntArithSmallMod ModuloUIntMod", "[ModuloUIntMod][cpu][XeHE]")
        {

            UIntArithSmallMod_ModuloUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod ModuloUIntMod64 tests passed-------" << std::endl;
            //UIntArithSmallMod_ModuloUIntMod<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod ModuloUIntMod32 tests passed-------" << std::endl;
        }


#if 0
        TEST(UIntArithSmallMod, TryInvertUIntMod)
        {
            uint64_t result;
            Modulus mod(5);
            ASSERT_FALSE(try_invert_uint_mod(0, mod, result));
            ASSERT_TRUE(try_invert_uint_mod(1, mod, result));
            ASSERT_EQ(1ULL, result);
            ASSERT_TRUE(try_invert_uint_mod(2, mod, result));
            ASSERT_EQ(3ULL, result);
            ASSERT_TRUE(try_invert_uint_mod(3, mod, result));
            ASSERT_EQ(2ULL, result);
            ASSERT_TRUE(try_invert_uint_mod(4, mod, result));
            ASSERT_EQ(4ULL, result);

            mod = 6;
            ASSERT_FALSE(try_invert_uint_mod(2, mod, result));
            ASSERT_FALSE(try_invert_uint_mod(3, mod, result));
            ASSERT_TRUE(try_invert_uint_mod(5, mod, result));
            ASSERT_EQ(5ULL, result);

            mod = 1351315121;
            ASSERT_TRUE(try_invert_uint_mod(331975426, mod, result));
            ASSERT_EQ(1052541512ULL, result);
        }

        TEST(UIntArithSmallMod, ExponentiateUIntMod)
        {
            Modulus mod(5);
            ASSERT_EQ(1ULL, exponentiate_uint_mod(1, 0, mod));
            ASSERT_EQ(1ULL, exponentiate_uint_mod(1, 0xFFFFFFFFFFFFFFFFULL, mod));
            ASSERT_EQ(3ULL, exponentiate_uint_mod(2, 0xFFFFFFFFFFFFFFFFULL, mod));

            mod = 0x1000000000000000ULL;
            ASSERT_EQ(0ULL, exponentiate_uint_mod(2, 60, mod));
            ASSERT_EQ(0x800000000000000ULL, exponentiate_uint_mod(2, 59, mod));

            mod = 131313131313;
            ASSERT_EQ(39418477653ULL, exponentiate_uint_mod(2424242424, 16, mod));
        }

        TEST(UIntArithSmallMod, DotProductMod)
        {
            Modulus mod(5);
            uint64_t arr1[64], arr2[64];
            for (size_t i = 0; i < 64; i++)
            {
                arr1[i] = 2;
                arr2[i] = 3;
            }

            ASSERT_EQ(0, dot_product_mod(arr1, arr2, 0, mod));
            ASSERT_EQ(1, dot_product_mod(arr1, arr2, 1, mod));
            ASSERT_EQ(2, dot_product_mod(arr1, arr2, 2, mod));
            ASSERT_EQ(15 % mod.value(), dot_product_mod(arr1, arr2, 15, mod));
            ASSERT_EQ(16 % mod.value(), dot_product_mod(arr1, arr2, 16, mod));
            ASSERT_EQ(17 % mod.value(), dot_product_mod(arr1, arr2, 17, mod));
            ASSERT_EQ(32 % mod.value(), dot_product_mod(arr1, arr2, 32, mod));
            ASSERT_EQ(64 % mod.value(), dot_product_mod(arr1, arr2, 64, mod));

            mod = get_prime(1024, SEAL_MOD_BIT_COUNT_MAX);
            for (size_t i = 0; i < 64; i++)
            {
                arr1[i] = mod.value() - 1;
                arr2[i] = mod.value() - 1;
            }

            ASSERT_EQ(0, dot_product_mod(arr1, arr2, 0, mod));
            ASSERT_EQ(1, dot_product_mod(arr1, arr2, 1, mod));
            ASSERT_EQ(2, dot_product_mod(arr1, arr2, 2, mod));
            ASSERT_EQ(15, dot_product_mod(arr1, arr2, 15, mod));
            ASSERT_EQ(16, dot_product_mod(arr1, arr2, 16, mod));
            ASSERT_EQ(17, dot_product_mod(arr1, arr2, 17, mod));
            ASSERT_EQ(32, dot_product_mod(arr1, arr2, 32, mod));
            ASSERT_EQ(64, dot_product_mod(arr1, arr2, 64, mod));
        }
#endif

        template <typename T>
        void UIntArithSmallMod_MultiplyUIntModOperand(void)
        //TEST(UIntArithSmallMod, MultiplyUIntModOperand)
        {
            T mod(3);
            MultiplyUIntModOperand<T> y;
            set_operand(T(1), mod, y);
            REQUIRE(T(1) == y.operand);
            REQUIRE(T(6148914691236517205ULL) == y.quotient);
            set_operand(T(2), mod, y);
            set_quotient(mod, y);
            REQUIRE(T(2) == y.operand);
            REQUIRE(T(12297829382473034410ULL) == y.quotient);

            mod = 2147483647ULL;
            set_operand(T(1), mod, y);
            REQUIRE(T(1) == y.operand);
            REQUIRE(T(8589934596) == y.quotient);
            set_operand(T(2147483646), mod, y);
            set_quotient(mod, y);
            REQUIRE(T(2147483646) == y.operand);
            REQUIRE(T(18446744065119617019ULL) == y.quotient);

            mod = T(2305843009211596801ULL);
            set_operand(T(1), mod, y);
            REQUIRE(T(1) == y.operand);
            REQUIRE(T(8) == y.quotient);
            set_operand(T(2305843009211596800ULL), mod, y);
            set_quotient(mod, y);
            REQUIRE(T(2305843009211596800ULL) == y.operand);
            REQUIRE(T(18446744073709551607ULL) == y.quotient);
        }

        TEST_CASE("XeHE UIntArithSmallMod MultiplyUIntModOperand", "[MultiplyUIntModOperand][cpu][XeHE]")
        {

            UIntArithSmallMod_MultiplyUIntModOperand<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntModOperand64 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyUIntModOperand<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntModOperand32 tests passed-------" << std::endl;
        }


        template <typename T>
        void UIntArithSmallMod_MultiplyUIntMod2(void)
        //TEST(UIntArithSmallMod, MultiplyUIntMod2)
        {
            T mod(2);
            MultiplyUIntModOperand<T> y;
            set_operand(T(0), mod, y);
            REQUIRE(T(0) == multiply_uint_mod(T(0), y, mod));
            REQUIRE(T(0) == multiply_uint_mod(T(1), y, mod));
            set_operand(T(1), mod, y);
            REQUIRE(T(0) == multiply_uint_mod(T(0), y, mod));
            REQUIRE(T(1) == multiply_uint_mod(T(1), y, mod));

            mod = 10;
            set_operand(T(0), mod, y);
            REQUIRE(T(0) == multiply_uint_mod(T(0), y, mod));
            REQUIRE(T(0) == multiply_uint_mod(T(1), y, mod));
            set_operand(T(1), mod, y);
            REQUIRE(T(0) == multiply_uint_mod(T(0), y, mod));
            REQUIRE(T(1) == multiply_uint_mod(T(1), y, mod));
            set_operand(T(6), mod, y);
            REQUIRE(T(2) == multiply_uint_mod(T(7), y, mod));
            set_operand(T(7), mod, y);
            REQUIRE(T(9) == multiply_uint_mod(T(7), y, mod));
            REQUIRE(T(2) == multiply_uint_mod(T(6), y, mod));

            mod = T(2305843009211596801ULL);
            set_operand(T(0), mod, y);
            REQUIRE(T(0) == multiply_uint_mod(T(0), y, mod));
            REQUIRE(T(0) == multiply_uint_mod(T(1), y, mod));
            set_operand(T(1), mod, y);
            REQUIRE(T(0) == multiply_uint_mod(T(0), y, mod));
            REQUIRE(T(1) == multiply_uint_mod(T(1), y, mod));
            set_operand(T(1152921504605798400ULL), mod, y);
            REQUIRE(T(576460752302899200ULL) == multiply_uint_mod(T(1152921504605798401ULL), y, mod));
            set_operand(T(1152921504605798401ULL), mod, y);
            REQUIRE(T(576460752302899200ULL) == multiply_uint_mod(T(1152921504605798400ULL), y, mod));
            REQUIRE(T(1729382256908697601ULL) == multiply_uint_mod(T(1152921504605798401ULL), y, mod));
            set_operand(T(2305843009211596800ULL), mod, y);
            REQUIRE(T(1) == multiply_uint_mod(T(2305843009211596800ULL), y, mod));
        }

        TEST_CASE("XeHE UIntArithSmallMod MultiplyUIntMod2", "[MultiplyUIntMod2][cpu][XeHE]")
        {

            UIntArithSmallMod_MultiplyUIntMod2<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntMod264 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyUIntMod2<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntMod232 tests passed-------" << std::endl;
        }

        template <typename T>
        void UIntArithSmallMod_MultiplyUIntModLazy(void)
        //TEST(UIntArithSmallMod, MultiplyUIntModLazy)
        {
            T mod(2);
            MultiplyUIntModOperand<T> y;
            set_operand(T(0), mod, y);
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(0), y, mod));
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(1), y, mod));
            set_operand(T(1), mod, y);
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(0), y, mod));
            REQUIRE(T(1) == multiply_uint_mod_lazy(T(1), y, mod));

            mod = 10;
            set_operand(T(0), mod, y);
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(0), y, mod));
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(1), y, mod));
            set_operand(T(1), mod, y);
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(0), y, mod));
            REQUIRE(T(1) == multiply_uint_mod_lazy(T(1), y, mod));
            set_operand(T(6), mod, y);
            REQUIRE(T(2) == multiply_uint_mod_lazy(T(7), y, mod));
            set_operand(T(7), mod, y);
            REQUIRE(T(9) == multiply_uint_mod_lazy(T(7), y, mod));
            REQUIRE(T(2) == multiply_uint_mod_lazy(T(6), y, mod));

            mod = T(2305843009211596801ULL);
            set_operand(T(0), mod, y);
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(0), y, mod));
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(1), y, mod));
            set_operand(T(1), mod, y);
            REQUIRE(T(0) == multiply_uint_mod_lazy(T(0), y, mod));
            REQUIRE(T(1) == multiply_uint_mod_lazy(T(1), y, mod));
            set_operand(T(1152921504605798400ULL), mod, y);
            REQUIRE(T(576460752302899200ULL) == multiply_uint_mod_lazy(T(1152921504605798401ULL), y, mod));
            set_operand(T(1152921504605798401ULL), mod, y);
            REQUIRE(T(576460752302899200ULL) == multiply_uint_mod_lazy(T(1152921504605798400ULL), y, mod));
            REQUIRE(T(1729382256908697601ULL) == multiply_uint_mod_lazy(T(1152921504605798401ULL), y, mod));
            set_operand(T(2305843009211596800ULL), mod, y);
            REQUIRE(T(2305843009211596802ULL) == multiply_uint_mod_lazy(T(2305843009211596800ULL), y, mod));
        }

        TEST_CASE("XeHE UIntArithSmallMod MultiplyUIntModLazy", "[MultiplyUIntModLazy][cpu][XeHE]")
        {

            UIntArithSmallMod_MultiplyUIntModLazy<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntModLazy64 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyUIntModLazy<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyUIntModLazy32 tests passed-------" << std::endl;
        }


        template <typename T>
        void UIntArithSmallMod_MultiplyAddMod2(void)
        //EST(UIntArithSmallMod, MultiplyAddMod2)
        {
            T mod(7);
            MultiplyUIntModOperand<T> y;
            const T* const_ratio = nullptr;
            {

#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(mod);
                const_ratio = seal_mod.const_ratio().data();
#endif

                set_operand(T(0), mod, y);
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(0), y, T(0), mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(1), y, T(0), mod, const_ratio));
                REQUIRE(T(1) == multiply_add_uint_mod<T>(T(0), 0, 1, mod, const_ratio));
                set_operand(T(1), mod, y);
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(0), y, 0, mod, const_ratio));
                set_operand(T(4), mod, y);
                REQUIRE(T(3) == multiply_add_uint_mod<T>(T(3), y, 5, mod, const_ratio));
            }
            {
                mod = 0x1FFFFFFFFFFFFFFFULL;
#ifdef BUILD_WITH_SEAL
                seal::Modulus seal_mod(mod);
                const_ratio = seal_mod.const_ratio().data();
#endif
                set_operand(T(0), mod, y);
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(0), y, 0, mod, const_ratio));
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(1), y, 0, mod, const_ratio));
                REQUIRE(T(1) == multiply_add_uint_mod<T>(T(0), y, 1, mod, const_ratio));
                set_operand(T(1), mod, y);
                REQUIRE(T(0) == multiply_add_uint_mod<T>(T(0), y, 0, mod, const_ratio));
                set_operand(mod - 1, mod, y);
                REQUIRE(T(0) == multiply_add_uint_mod<T>(mod - 1, y, mod - 1, mod, const_ratio));
            }
        }

        TEST_CASE("XeHE UIntArithSmallMod MultiplyAddMod2", "[MultiplyAddMod2][cpu][XeHE]")
        {

            UIntArithSmallMod_MultiplyAddMod2<uint64_t>();
            std::cout << "-------XeHE UIntArithSmallMod MultiplyAddMod264 tests passed-------" << std::endl;
            //UIntArithSmallMod_MultiplyAddMod2<uint32_t>();
            //std::cout << "-------XeHE UIntArithSmallMod MultiplyAddMod232 tests passed-------" << std::endl;
        }
    } // namespace util
} // namespace xehetest
