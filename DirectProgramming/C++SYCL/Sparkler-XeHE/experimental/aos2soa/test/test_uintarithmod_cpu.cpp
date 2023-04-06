// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch2/catch.hpp"
#include <vector>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <cstdint>

#include "util/xe_uintarithmod.h"
#include "util/xe_uintcore.h"



using namespace xehe::util;
using namespace std;

namespace xehtest
{
    namespace util
    {
        template <typename T>
        void UIntArithMod_IncrementUIntMod(void)
        {

            auto sp_value = allocate_uint<T>(2);
            auto sp_modulus = allocate_uint<T>(2);
            auto value = sp_value.get();
            auto modulus = sp_modulus.get();
            value[0] = 0;
            value[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            increment_uint_mod(value, modulus, 2, value);
            REQUIRE(1ULL ==  value[0]);
            REQUIRE(static_cast<T>(0) == value[1]);
            increment_uint_mod(value, modulus, 2, value);
            REQUIRE(static_cast<T>(2) == value[0]);
            REQUIRE(static_cast<T>(0) == value[1]);
            increment_uint_mod(value, modulus, 2, value);
            REQUIRE(static_cast<T>(0) == value[0]);
            REQUIRE(static_cast<T>(0) == value[1]);

            value[0] = static_cast<T>(0xFFFFFFFFFFFFFFFD);
            value[1] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            modulus[0] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            modulus[1] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            increment_uint_mod(value, modulus, 2, value);
            REQUIRE(static_cast<T>(0xFFFFFFFFFFFFFFFE) == value[0]);
            REQUIRE(static_cast<T>(0xFFFFFFFFFFFFFFFF) == value[1]);
            increment_uint_mod(value, modulus, 2, value);
            REQUIRE(static_cast<T>(0) == value[0]);
            REQUIRE(static_cast<T>(0) == value[1]);
            increment_uint_mod(value, modulus, 2, value);
            REQUIRE(1ULL == value[0]);
            REQUIRE(static_cast<T>(0) == value[1]);
        }

        TEST_CASE("XeHE UIntArithMod IncrementUIntMod", "[IncrementUIntMod][cpu][XeHE]")
        {

            UIntArithMod_IncrementUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithMod IncrementUIntMod64 tests passed-------" << std::endl;
            UIntArithMod_IncrementUIntMod<uint32_t>();
            std::cout << "-------XeHE UIntArithMod IncrementUIntMod32 tests passed-------" << std::endl;
        }


#if 0
        TEST(UIntArithMod, DecrementUIntMod)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto value(allocate_uint(2, pool));
            auto modulus(allocate_uint(2, pool));
            value[0] = 2;
            value[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            decrement_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(1ULL, value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);
            decrement_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(0), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);
            decrement_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(2), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 1;
            value[1] = 0;
            modulus[0] = 0xFFFFFFFFFFFFFFFF;
            modulus[1] = 0xFFFFFFFFFFFFFFFF;
            decrement_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(0), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);
            decrement_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFE), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF), value[1]);
            decrement_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFD), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF), value[1]);
        }

        TEST(UIntArithMod, NegateUIntMod)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto value(allocate_uint(2, pool));
            auto modulus(allocate_uint(2, pool));
            value[0] = 0;
            value[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            negate_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(0), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 1;
            value[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            negate_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(2), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);
            negate_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(1ULL, value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 2;
            value[1] = 0;
            modulus[0] = 0xFFFFFFFFFFFFFFFF;
            modulus[1] = 0xFFFFFFFFFFFFFFFF;
            negate_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFD), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF), value[1]);
            negate_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(static_cast<uint64_t>(2), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);
        }

        TEST(UIntArithMod, Div2UIntMod)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto value(allocate_uint(2, pool));
            auto modulus(allocate_uint(2, pool));
            value[0] = 0;
            value[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            div2_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(0ULL, value[0]);
            ASSERT_EQ(0ULL, value[1]);

            value[0] = 1;
            value[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            div2_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(2ULL, value[0]);
            ASSERT_EQ(0ULL, value[1]);

            value[0] = 8;
            value[1] = 0;
            modulus[0] = 17;
            modulus[1] = 0;
            div2_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(4ULL, value[0]);
            ASSERT_EQ(0ULL, value[1]);

            value[0] = 5;
            value[1] = 0;
            modulus[0] = 17;
            modulus[1] = 0;
            div2_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(11ULL, value[0]);
            ASSERT_EQ(0ULL, value[1]);

            value[0] = 1;
            value[1] = 0;
            modulus[0] = 0xFFFFFFFFFFFFFFFFULL;
            modulus[1] = 0xFFFFFFFFFFFFFFFFULL;
            div2_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(0ULL, value[0]);
            ASSERT_EQ(0x8000000000000000ULL, value[1]);

            value[0] = 3;
            value[1] = 0;
            modulus[0] = 0xFFFFFFFFFFFFFFFFULL;
            modulus[1] = 0xFFFFFFFFFFFFFFFFULL;
            div2_uint_mod(value.get(), modulus.get(), 2, value.get());
            ASSERT_EQ(1ULL, value[0]);
            ASSERT_EQ(0x8000000000000000ULL, value[1]);
        }
#endif

        template <typename T>
        void UIntArithMod_AddUIntMod(void)
        {

            auto sp_value1 = allocate_uint<T>(2);
            auto sp_value2 = allocate_uint<T>(2);
            auto sp_modulus = allocate_uint<T>(2);
            auto value1 = sp_value1.get();
            auto value2 = sp_value2.get();
            auto modulus = sp_modulus.get();
            value1[0] = 0;
            value1[1] = 0;
            value2[0] = 0;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            add_uint_uint_mod(value1, value2, modulus, 2, value1);
            REQUIRE(static_cast<T>(0) == value1[0]);
            REQUIRE(static_cast<T>(0) == value1[1]);

            value1[0] = 1;
            value1[1] = 0;
            value2[0] = 1;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            add_uint_uint_mod(value1, value2, modulus, 2, value1);
            REQUIRE(static_cast<T>(2) ==  value1[0]);
            REQUIRE(static_cast<T>(0) == value1[1]);

            value1[0] = 1;
            value1[1] = 0;
            value2[0] = 2;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            add_uint_uint_mod(value1, value2, modulus, 2, value1);
            REQUIRE(static_cast<T>(0) == value1[0]);
            REQUIRE(static_cast<T>(0) == value1[1]);

            value1[0] = 2;
            value1[1] = 0;
            value2[0] = 2;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            add_uint_uint_mod(value1, value2, modulus, 2, value1);
            REQUIRE(1ULL == value1[0]);
            REQUIRE(static_cast<T>(0) == value1[1]);

            value1[0] = static_cast<T>(0xFFFFFFFFFFFFFFFE);
            value1[1] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            value2[0] = static_cast<T>(0xFFFFFFFFFFFFFFFE);
            value2[1] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            modulus[0] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            modulus[1] = static_cast<T>(0xFFFFFFFFFFFFFFFF);
            add_uint_uint_mod(value1, value2, modulus, 2, value1);
            REQUIRE(static_cast<T>(0xFFFFFFFFFFFFFFFD) == value1[0]);
            REQUIRE(static_cast<T>(0xFFFFFFFFFFFFFFFF) == value1[1]);
        }

        TEST_CASE("XeHE UIntArithMod AddUIntMod", "[AddUIntMod][cpu][XeHE]")
        {

            UIntArithMod_AddUIntMod<uint64_t>();
            std::cout << "-------XeHE UIntArithMod AddUIntMod64 tests passed-------" << std::endl;
            UIntArithMod_AddUIntMod<uint32_t>();
            std::cout << "-------XeHE UIntArithMod AddUIntMod32 tests passed-------" << std::endl;
        }

#if 0
        TEST(UIntArithMod, SubUIntMod)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto value1(allocate_uint(2, pool));
            auto value2(allocate_uint(2, pool));
            auto modulus(allocate_uint(2, pool));
            value1[0] = 0;
            value1[1] = 0;
            value2[0] = 0;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            sub_uint_uint_mod(value1.get(), value2.get(), modulus.get(), 2, value1.get());
            ASSERT_EQ(static_cast<uint64_t>(0), value1[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value1[1]);

            value1[0] = 2;
            value1[1] = 0;
            value2[0] = 1;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            sub_uint_uint_mod(value1.get(), value2.get(), modulus.get(), 2, value1.get());
            ASSERT_EQ(1ULL, value1[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value1[1]);

            value1[0] = 1;
            value1[1] = 0;
            value2[0] = 2;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            sub_uint_uint_mod(value1.get(), value2.get(), modulus.get(), 2, value1.get());
            ASSERT_EQ(static_cast<uint64_t>(2), value1[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value1[1]);

            value1[0] = 2;
            value1[1] = 0;
            value2[0] = 2;
            value2[1] = 0;
            modulus[0] = 3;
            modulus[1] = 0;
            sub_uint_uint_mod(value1.get(), value2.get(), modulus.get(), 2, value1.get());
            ASSERT_EQ(static_cast<uint64_t>(0), value1[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value1[1]);

            value1[0] = 1;
            value1[1] = 0;
            value2[0] = 0xFFFFFFFFFFFFFFFE;
            value2[1] = 0xFFFFFFFFFFFFFFFF;
            modulus[0] = 0xFFFFFFFFFFFFFFFF;
            modulus[1] = 0xFFFFFFFFFFFFFFFF;
            sub_uint_uint_mod(value1.get(), value2.get(), modulus.get(), 2, value1.get());
            ASSERT_EQ(static_cast<uint64_t>(2), value1[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value1[1]);
        }

        TEST(UIntArithMod, TryInvertUIntMod)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto value(allocate_uint(2, pool));
            auto modulus(allocate_uint(2, pool));
            value[0] = 0;
            value[1] = 0;
            modulus[0] = 5;
            modulus[1] = 0;
            ASSERT_FALSE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));

            value[0] = 1;
            value[1] = 0;
            modulus[0] = 5;
            modulus[1] = 0;
            ASSERT_TRUE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));
            ASSERT_EQ(1ULL, value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 2;
            value[1] = 0;
            modulus[0] = 5;
            modulus[1] = 0;
            ASSERT_TRUE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));
            ASSERT_EQ(static_cast<uint64_t>(3), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 3;
            value[1] = 0;
            modulus[0] = 5;
            modulus[1] = 0;
            ASSERT_TRUE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));
            ASSERT_EQ(static_cast<uint64_t>(2), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 4;
            value[1] = 0;
            modulus[0] = 5;
            modulus[1] = 0;
            ASSERT_TRUE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));
            ASSERT_EQ(static_cast<uint64_t>(4), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);

            value[0] = 2;
            value[1] = 0;
            modulus[0] = 6;
            modulus[1] = 0;
            ASSERT_FALSE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));

            value[0] = 3;
            value[1] = 0;
            modulus[0] = 6;
            modulus[1] = 0;
            ASSERT_FALSE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));

            value[0] = 331975426;
            value[1] = 0;
            modulus[0] = 1351315121;
            modulus[1] = 0;
            ASSERT_TRUE(try_invert_uint_mod(value.get(), modulus.get(), 2, value.get(), pool));
            ASSERT_EQ(static_cast<uint64_t>(1052541512), value[0]);
            ASSERT_EQ(static_cast<uint64_t>(0), value[1]);
        }
#endif
    } // namespace util
} // namespace sealtest
