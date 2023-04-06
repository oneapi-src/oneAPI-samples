/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch2/catch.hpp"
#include <vector>
#include <ctime>
#include <assert.h>
#include <iostream>

// XeHE
//#include "util/common.h"
#include "util/xe_uintarith.h"


#include <cstdint>

using namespace xehe::util;
using namespace std;

template<typename T>
void UIntArith_AddUInt64Generic(void)
{
    T result;
    REQUIRE_FALSE(add_uint64_generic(T(0), T(0), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE_FALSE(add_uint64_generic(T(1), T(1), 0, &result));
    REQUIRE(T(2) == result);
    REQUIRE_FALSE(add_uint64_generic(T(1), T(0), 1, &result));
    REQUIRE(T(2) == result);
    REQUIRE_FALSE(add_uint64_generic(T(0), T(1), 1, &result));
    REQUIRE(T(2) == result);
    REQUIRE_FALSE(add_uint64_generic(T(1), T(1), 1, &result));
    REQUIRE(T(3) == result);
    REQUIRE(add_uint64_generic(T(-1), T(1), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE(add_uint64_generic(T(1), T(-1), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE(add_uint64_generic(T(1), T(-1), 1, &result));
    REQUIRE(T(1) == result);
    REQUIRE(add_uint64_generic(T(2), T(-2), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE(add_uint64_generic(T(2), T(-2), 1, &result));
    REQUIRE(T(1) == result);
    {
        auto opnd1 = (T)(sizeof(T) == 8) ? T(0xF00F00F00F00F00F) : T(0xF00F00F0);
        auto opnd2 = (T)(sizeof(T) == 8) ? T(0x0FF0FF0FF0FF0FF0) : T(0x0FF0FF0F);
        REQUIRE_FALSE(add_uint64_generic(opnd1, opnd2, 0, &result));
        REQUIRE(T(-1) == result);
    }

    {
        auto opnd1 = (T)(sizeof(T) == 8) ? T(0xF00F00F00F00F00F) : T(0xF00F00F0);
        auto opnd2 = (T)(sizeof(T) == 8) ? T(0x0FF0FF0FF0FF0FF0) : T(0x0FF0FF0F);
        REQUIRE(add_uint64_generic(opnd1, opnd2, 1, &result));
        REQUIRE(T(0) == result);
    }
}

#if 0
TEST_CASE("XeHE UIntArith AddUIntGeneric", "[AddUIntGeneric][cpu][XeHE]")
{

    UIntArith_AddUInt64Generic<uint64_t>();
    std::cout << "-------XeHE UIntArith AddUInt64Generic tests passed-------" << std::endl;

    UIntArith_AddUInt64Generic<uint32_t>();
    std::cout << "-------XeHE UIntArith AddUInt32Generic tests passed-------" << std::endl;

}
#endif



#if SEAL_COMPILER == SEAL_COMPILER_MSVC
#pragma optimize ("", off)
#elif SEAL_COMPILER == SEAL_COMPILER_GCC
#pragma GCC push_options
#pragma GCC optimize ("O0")
#elif SEAL_COMPILER == SEAL_COMPILER_CLANG
#pragma clang optimize off
#endif
template <typename T>
void UIntArith_AddUInt(void)
{
    T result;
    REQUIRE_FALSE(add_uint64(T(0), T(0), 0, &result));
    REQUIRE(T(0) ==  result);
    REQUIRE_FALSE(add_uint64(T(1), T(1), 0, &result));
    REQUIRE(T(2) == result);
    REQUIRE_FALSE(add_uint64(T(1), T(0), 1, &result));
    REQUIRE(T(2) == result);
    REQUIRE_FALSE(add_uint64(T(0), T(1), 1, &result));
    REQUIRE(T(2) == result);
    REQUIRE_FALSE(add_uint64(T(1), T(1), 1, &result));
    REQUIRE(T(3) == result);
    REQUIRE(add_uint64(T(-1), T(1), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE(add_uint64(T(1), T(-1), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE(add_uint64(T(1), T(-1), 1, &result));
    REQUIRE(T(1) == result);
    REQUIRE(add_uint64(T(2), T(-2), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE(add_uint64(T(2), T(-2), 1, &result));
    REQUIRE(T(1) == result);
    {
        auto opnd1 = (T)(sizeof(T) == 8) ? T(0xF00F00F00F00F00F) : T(0xF00F00F0);
        auto opnd2 = (T)(sizeof(T) == 8) ? T(0x0FF0FF0FF0FF0FF0) : T(0x0FF0FF0F);
        REQUIRE_FALSE(add_uint64_generic(opnd1, opnd2, 0, &result));
        REQUIRE(T(-1) == result);
    }

    {
        auto opnd1 = (T)(sizeof(T) == 8) ? T(0xF00F00F00F00F00F) : T(0xF00F00F0);
        auto opnd2 = (T)(sizeof(T) == 8) ? T(0x0FF0FF0FF0FF0FF0) : T(0x0FF0FF0F);
        REQUIRE(add_uint64_generic(opnd1, opnd2, 1, &result));
        REQUIRE(T(0) == result);
    }

}

TEST_CASE("XeHE IntArith AddUInt", "[AddUInt][cpu][XeHE]")
{

    UIntArith_AddUInt<uint64_t>();
    std::cout << "-------XeHE UIntArith AddUInt64 tests passed-------" << std::endl;

    UIntArith_AddUInt<uint32_t>();
    std::cout << "-------XeHE UIntArith AddUInt32 tests passed-------" << std::endl;
}


#if SEAL_COMPILER == SEAL_COMPILER_MSVC
#pragma optimize ("", on)
#elif SEAL_COMPILER == SEAL_COMPILER_GCC
#pragma GCC pop_options
#elif SEAL_COMPILER == SEAL_COMPILER_CLANG
#pragma clang optimize on
#endif

template <typename T>
void UIntArith_SubUInt(void)
{
    T result;
    REQUIRE_FALSE(sub_uint64(T(0), T(0), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE_FALSE(sub_uint64(T(1), T(1), 0, &result));
    REQUIRE(T(0) == result);
    REQUIRE_FALSE(sub_uint64(T(1), T(0), 1, &result));
    REQUIRE(T(0) == result);
    REQUIRE(sub_uint64(T(0), T(1), 1, &result));
    REQUIRE(T(-2) == result);
    REQUIRE(sub_uint64(T(1), T(1), 1, &result));
    REQUIRE(T(-1) == result);
    REQUIRE_FALSE(sub_uint64(T(-1), T(1), 0, &result));
    REQUIRE(T(-2) == result);
    REQUIRE(sub_uint64(T(1), T(-1), 0, &result));
    REQUIRE(T(2) == result);
    REQUIRE(sub_uint64(T(1), T(-1), 1, &result));
    REQUIRE(T(1) == result);
    REQUIRE(sub_uint64(T(2), T(-2), 0, &result));
    REQUIRE(T(4) == result);
    REQUIRE(sub_uint64(T(2), T(-2), 1, &result));
    REQUIRE(T(3) == result);
    {
        T opnd1 = (T)(sizeof(T) == 8) ? T(0xF00F00F00F00F00F) : T(0x0F00F00F);
        T opnd2 = (T)(sizeof(T) == 8) ? T(0x0FF0FF0FF0FF0FF0) : T(0x00FF0FF0);
        T res = (T)(sizeof(T) == 8) ? T(0xE01E01E01E01E01F) : T(0xE01E01F);
        REQUIRE_FALSE(sub_uint64(opnd1, opnd2, 0, &result));
        REQUIRE(res == result);
    }
    {
        T opnd1 = (T)(sizeof(T) == 8) ? T(0xF00F00F00F00F00F) : T(0xF00F00F);
        T opnd2 = (T)(sizeof(T) == 8) ? T(0x0FF0FF0FF0FF0FF0) : T(0x0FF0FF0);
        T res = (T)(sizeof(T) == 8) ? T(0xE01E01E01E01E01E) : T(0xE01E01E);
        REQUIRE_FALSE(sub_uint64(opnd1, opnd2, 1, &result));
        REQUIRE(res == result);
    }

}

TEST_CASE("XeHE UIntArith SubUInt", "[SubUInt][cpu][XeHE]")
{
    UIntArith_SubUInt<uint64_t>();
    std::cout << "-------XeHE UIntArith SubUInt64 tests passed-------" << std::endl;

    UIntArith_SubUInt<uint32_t>();
    std::cout << "-------XeHE UIntArith SubUInt32 tests passed-------" << std::endl;

}

template <typename T>
void UIntArith_AddUIntUInt(void)
{
    auto s_ptr = allocate_uint<T>(2);
    auto s_ptr2 = allocate_uint<T>(2);
    auto s_ptr3 = allocate_uint<T>(2);

    auto ptr = s_ptr.get();
    auto ptr2 = s_ptr2.get();
    auto ptr3 = s_ptr3.get();
    ptr[0] = 0;
    ptr[1] = 0;
    ptr2[0] = 0;
    ptr2[1] = 0;
    ptr3[0] = T(-1);
    ptr3[1] = T(-1);
    REQUIRE_FALSE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(0) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    ptr2[0] = 0;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);

    ptr[0] = T(-2);
    ptr[1] = T(-1);
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = T(-1);
    ptr3[1] = T(-1);
    REQUIRE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(0) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    ptr2[0] = T(-1);
    ptr2[1] = T(-1);
    ptr3[0] = 0;
    ptr3[1] = 0;

    REQUIRE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-2) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);
    REQUIRE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr.get()) != 0);
    REQUIRE(T(-2) == ptr[0]);
    REQUIRE(T(-1) == ptr[1]);

    ptr[0] = T(-1);
    ptr[1] = 0;
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(add_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(0) == ptr3[0]);
    REQUIRE(T(1) == ptr3[1]);

    ptr[0] = T(-1);
    ptr[1] = 5;
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(add_uint_uint(s_ptr.get(), 2, s_ptr2.get(), 1, false, 2, s_ptr3.get()) != 0);
    REQUIRE(T(0) ==  ptr3[0]);
    REQUIRE(T(6) ==  ptr3[1]);
    REQUIRE_FALSE(add_uint_uint(s_ptr.get(), 2, s_ptr2.get(), 1, true, 2, s_ptr3.get()) != 0);
    REQUIRE(T(1) == ptr3[0]);
    REQUIRE(T(6) == ptr3[1]);
}


TEST_CASE("XeHE UIntArith AddUIntUInt", "[AddUIntUInt][cpu][XeHE]")
{

    UIntArith_AddUIntUInt<uint64_t>();
    std::cout << "-------XeHE UIntArith AddUIntUInt64 tests passed-------" << std::endl;
    UIntArith_AddUIntUInt<uint32_t>();
    std::cout << "-------XeHE UIntArith AddUIntUInt32 tests passed-------" << std::endl;
}

template<typename T>
void UIntArith_SubUIntUInt(void)
{

    auto s_ptr = allocate_uint<T>(2);
    auto s_ptr2 = allocate_uint<T>(2);
    auto s_ptr3 = allocate_uint<T>(2);

    auto ptr = s_ptr.get();
    auto ptr2 = s_ptr2.get();
    auto ptr3 = s_ptr3.get();

    ptr[0] = 0;
    ptr[1] = 0;
    ptr2[0] = 0;
    ptr2[1] = 0;
    ptr3[0] = T(-1);
    ptr3[1] = T(-1);
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(0) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    ptr2[0] = 0;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-2) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);

    ptr[0] = 0;
    ptr[1] = 0;
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);
    REQUIRE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr.get()) != 0);
    REQUIRE(T(-1) == ptr[0]);
    REQUIRE(T(-1) == ptr[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    ptr2[0] = T(-1);
    ptr2[1] = T(-1);
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(0) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr.get()) != 0);
    REQUIRE(T(0) == ptr[0]);
    REQUIRE(T(0) == ptr[1]);

    ptr[0] = T(-2);
    ptr[1] = T(-1);
    ptr2[0] = T(-1);
    ptr2[1] = T(-1);
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(-1) == ptr3[1]);

    ptr[0] = 0;
    ptr[1] = 1;
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), s_ptr2.get(), 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);

    ptr[0] = 0;
    ptr[1] = 1;
    ptr2[0] = 1;
    ptr2[1] = 0;
    ptr3[0] = 0;
    ptr3[1] = 0;
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), 2, s_ptr2.get(), 1, false, 2, s_ptr3.get()) != 0);
    REQUIRE(T(-1) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);
    REQUIRE_FALSE(sub_uint_uint(s_ptr.get(), 2, s_ptr2.get(), 1, true, 2, s_ptr3.get()) != 0);
    REQUIRE(T(-2) == ptr3[0]);
    REQUIRE(T(0) == ptr3[1]);
}

TEST_CASE("XeHE UIntArith SubUIntUInt", "[SubUIntUInt][cpu][XeHE]")
{

    UIntArith_SubUIntUInt<uint64_t>();
    std::cout << "-------XeHE UIntArith SubUIntUInt64 tests passed-------" << std::endl;

    UIntArith_SubUIntUInt<uint32_t>();
    std::cout << "-------XeHE UIntArith SubUIntUInt32 tests passed-------" << std::endl;

}

template<typename T>
void UIntArith_AddUIntUInt2(void)
{
    auto s_ptr = allocate_uint<T>(2);
    auto s_ptr2 = allocate_uint<T>(2);

    auto ptr = s_ptr.get();
    auto ptr2 = s_ptr2.get();


    ptr[0] = T(0);
    ptr[1] = T(0);
    REQUIRE_FALSE(add_uint_uint64(s_ptr.get(), T(0), 2, s_ptr2.get()));
    REQUIRE(T(0) == ptr2[0]);
    REQUIRE(T(0) == ptr2[1]);

    ptr[0] = top_half_bits(T(-1));
    ptr[1] = T(0);
    REQUIRE_FALSE(add_uint_uint64(s_ptr.get(), bot_half_bits(T(-1)), 2, s_ptr2.get()));
    REQUIRE(T(-1) == ptr2[0]);
    REQUIRE(T(0) == ptr2[1]);

    ptr[0] = top_half_bits(T(-1));
    ptr[1] = top_half_bits(T(-1));
    auto arg2 = (sizeof(T) == 8) ? T(0x100000000) : T(0x10000);
    REQUIRE_FALSE(add_uint_uint64(s_ptr.get(), arg2, 2, s_ptr2.get()));
    REQUIRE(T(0) == ptr2[0]);
    REQUIRE((top_half_bits(T(-1)) + T(1)) == ptr2[1]);

    ptr[0] = T(-1);
    ptr[1] = T(-1);
    REQUIRE(add_uint_uint64(s_ptr.get(), T(1), 2, s_ptr2.get()));
    REQUIRE(T(0) == ptr2[0]);
    REQUIRE(T(0) == ptr2[1]);
}



TEST_CASE("XeHE UIntArith AddUIntUInt2", "[AddUIntUInt2][cpu][XeHE]")
{
    UIntArith_AddUIntUInt2<uint64_t>();
    std::cout << "-------XeHE UIntArith AddUIntUInt64_2 tests passed-------" << std::endl;

    UIntArith_AddUIntUInt2<uint32_t>();
    std::cout << "-------XeHE UIntArith AddUIntUInt32_2 tests passed-------" << std::endl;
}

#if 0
        TEST(UIntArith, SubUIntUInt64)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));

            ptr[0] = T(0);
            ptr[1] = T(0);
            REQUIRE_FALSE(sub_uint_uint64(ptr.get(), T(0), 2, ptr2.get()));
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);

            ptr[0] = T(0);
            ptr[1] = T(0);
            REQUIRE(sub_uint_uint64(ptr.get(), 1, 2, ptr2.get()));
            REQUIRE(T(-1), ptr2[0]);
            REQUIRE(T(-1), ptr2[1]);

            ptr[0] = 1;
            ptr[1] = T(0);
            REQUIRE(sub_uint_uint64(ptr.get(), 2, 2, ptr2.get()));
            REQUIRE(T(-1), ptr2[0]);
            REQUIRE(T(-1), ptr2[1]);

            ptr[0] = 0xFFFFFFFF0000000T(0);
            ptr[1] = T(0);
            REQUIRE_FALSE(sub_uint_uint64(ptr.get(), 0xFFFFFFFF, 2, ptr2.get()));
            REQUIRE(0xFFFFFFFE00000001, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);

            ptr[0] = 0xFFFFFFFF0000000T(0);
            ptr[1] = 0xFFFFFFFF0000000T(0);
            REQUIRE_FALSE(sub_uint_uint64(ptr.get(), 0x10000000T(0), 2, ptr2.get()));
            REQUIRE(0xFFFFFFFE0000000T(0), ptr2[0]);
            REQUIRE(0xFFFFFFFF0000000T(0), ptr2[1]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            REQUIRE_FALSE(sub_uint_uint64(ptr.get(), 1, 2, ptr2.get()));
            REQUIRE(T(-2), ptr2[0]);
            REQUIRE(T(-1), ptr2[1]);
        }

        TEST(UIntArith, IncrementUInt)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr1(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr1[0] = 0;
            ptr1[1] = 0;
            REQUIRE_FALSE(increment_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE_FALSE(increment_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(2), ptr1[0]);
            REQUIRE(T(0), ptr1[1]);

            ptr1[0] = T(-1);
            ptr1[1] = 0;
            REQUIRE_FALSE(increment_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(1, ptr2[1]);
            REQUIRE_FALSE(increment_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(1, ptr1[0]);
            REQUIRE(1, ptr1[1]);

            ptr1[0] = T(-1);
            ptr1[1] = 1;
            REQUIRE_FALSE(increment_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(2), ptr2[1]);
            REQUIRE_FALSE(increment_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(1, ptr1[0]);
            REQUIRE(T(2), ptr1[1]);

            ptr1[0] = T(-2);
            ptr1[1] = T(-1);
            REQUIRE_FALSE(increment_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(T(T(-1)), ptr2[0]);
            REQUIRE(T(T(-1)), ptr2[1]);
            REQUIRE(increment_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(0), ptr1[0]);
            REQUIRE(T(0), ptr1[1]);
            REQUIRE_FALSE(increment_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
        }

        TEST(UIntArith, DecrementUInt)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr1(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr1[0] = 2;
            ptr1[1] = 2;
            REQUIRE_FALSE(decrement_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(2), ptr2[1]);
            REQUIRE_FALSE(decrement_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(0), ptr1[0]);
            REQUIRE(T(2), ptr1[1]);
            REQUIRE_FALSE(decrement_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(T(T(-1)), ptr2[0]);
            REQUIRE(1, ptr2[1]);
            REQUIRE_FALSE(decrement_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(-2), ptr1[0]);
            REQUIRE(1, ptr1[1]);

            ptr1[0] = 2;
            ptr1[1] = 1;
            REQUIRE_FALSE(decrement_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(1, ptr2[0]);
            REQUIRE(1, ptr2[1]);
            REQUIRE_FALSE(decrement_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(0), ptr1[0]);
            REQUIRE(1, ptr1[1]);
            REQUIRE_FALSE(decrement_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(T(T(-1)), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE_FALSE(decrement_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(-2), ptr1[0]);
            REQUIRE(T(0), ptr1[1]);

            ptr1[0] = 2;
            ptr1[1] = 0;
            REQUIRE_FALSE(decrement_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE_FALSE(decrement_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(0), ptr1[0]);
            REQUIRE(T(0), ptr1[1]);
            REQUIRE(decrement_uint(ptr1.get(), 2, ptr2.get()) != 0);
            REQUIRE(T(T(-1)), ptr2[0]);
            REQUIRE(T(T(-1)), ptr2[1]);
            REQUIRE_FALSE(decrement_uint(ptr2.get(), 2, ptr1.get()) != 0);
            REQUIRE(T(-2), ptr1[0]);
            REQUIRE(T(T(-1)), ptr1[1]);
        }

        TEST(UIntArith, NegateUInt)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 1;
            ptr[1] = 0;
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(T(-1)), ptr[0]);
            REQUIRE(T(T(-1)), ptr[1]);
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(1, ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 2;
            ptr[1] = 0;
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(-2), ptr[0]);
            REQUIRE(T(T(-1)), ptr[1]);
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(2), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 0;
            ptr[1] = 1;
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(T(-1)), ptr[1]);
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(1, ptr[1]);

            ptr[0] = 0;
            ptr[1] = 2;
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(-2), ptr[1]);
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(2), ptr[1]);

            ptr[0] = 1;
            ptr[1] = 1;
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(T(-1)), ptr[0]);
            REQUIRE(T(-2), ptr[1]);
            negate_uint(ptr.get(), 2, ptr.get());
            REQUIRE(1, ptr[0]);
            REQUIRE(1, ptr[1]);
        }

        TEST(UIntArith, LeftShiftUInt)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            left_shift_uint(ptr.get(), 0, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            left_shift_uint(ptr.get(), 10, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            left_shift_uint(ptr.get(), 10, 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 0x5555555555555555;
            ptr[1] = 0xAAAAAAAAAAAAAAAA;
            left_shift_uint(ptr.get(), 0, 2, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            left_shift_uint(ptr.get(), 0, 2, ptr.get());
            REQUIRE(T(0x5555555555555555), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr[1]);
            left_shift_uint(ptr.get(), 1, 2, ptr2.get());
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0x5555555555555554), ptr2[1]);
            left_shift_uint(ptr.get(), 2, 2, ptr2.get());
            REQUIRE(T(0x5555555555555554), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr2[1]);
            left_shift_uint(ptr.get(), 64, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0x5555555555555555), ptr2[1]);
            left_shift_uint(ptr.get(), 65, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            left_shift_uint(ptr.get(), 127, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0x8000000000000000), ptr2[1]);

            left_shift_uint(ptr.get(), 2, 2, ptr.get());
            REQUIRE(T(0x5555555555555554), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr[1]);
            left_shift_uint(ptr.get(), 64, 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0x5555555555555554), ptr[1]);
        }

        TEST(UIntArith, LeftShiftUInt128)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            left_shift_uint128(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            left_shift_uint128(ptr.get(), 10, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            left_shift_uint128(ptr.get(), 10, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 0x5555555555555555;
            ptr[1] = 0xAAAAAAAAAAAAAAAA;
            left_shift_uint128(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            left_shift_uint128(ptr.get(), 0, ptr.get());
            REQUIRE(T(0x5555555555555555), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr[1]);
            left_shift_uint128(ptr.get(), 1, ptr2.get());
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0x5555555555555554), ptr2[1]);
            left_shift_uint128(ptr.get(), 2, ptr2.get());
            REQUIRE(T(0x5555555555555554), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr2[1]);
            left_shift_uint128(ptr.get(), 64, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0x5555555555555555), ptr2[1]);
            left_shift_uint128(ptr.get(), 65, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            left_shift_uint128(ptr.get(), 127, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0x8000000000000000), ptr2[1]);

            left_shift_uint128(ptr.get(), 2, ptr.get());
            REQUIRE(T(0x5555555555555554), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr[1]);
            left_shift_uint128(ptr.get(), 64, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0x5555555555555554), ptr[1]);
        }

        TEST(UIntArith, LeftShiftUInt192)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(3, pool));
            auto ptr2(allocate_uint(3, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr[2] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr2[2] = T(-1);
            left_shift_uint192(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr2[2] = T(-1);
            left_shift_uint192(ptr.get(), 10, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);
            left_shift_uint192(ptr.get(), 10, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);
            REQUIRE(T(0), ptr[2]);

            ptr[0] = 0x5555555555555555;
            ptr[1] = 0xAAAAAAAAAAAAAAAA;
            ptr[2] = 0xCDCDCDCDCDCDCDCD;
            left_shift_uint192(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            REQUIRE(T(0xCDCDCDCDCDCDCDCD), ptr2[2]);
            left_shift_uint192(ptr.get(), 0, ptr.get());
            REQUIRE(T(0x5555555555555555), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr[1]);
            REQUIRE(T(0xCDCDCDCDCDCDCDCD), ptr[2]);
            left_shift_uint192(ptr.get(), 1, ptr2.get());
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0x5555555555555554), ptr2[1]);
            REQUIRE(T(0x9B9B9B9B9B9B9B9B), ptr2[2]);
            left_shift_uint192(ptr.get(), 2, ptr2.get());
            REQUIRE(T(0x5555555555555554), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr2[1]);
            REQUIRE(T(0x3737373737373736), ptr2[2]);
            left_shift_uint192(ptr.get(), 64, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0x5555555555555555), ptr2[1]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[2]);
            left_shift_uint192(ptr.get(), 65, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            REQUIRE(T(0x5555555555555554), ptr2[2]);
            left_shift_uint192(ptr.get(), 191, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE(T(0x8000000000000000), ptr2[2]);

            left_shift_uint192(ptr.get(), 2, ptr.get());
            REQUIRE(T(0x5555555555555554), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr[1]);
            REQUIRE(T(0x3737373737373736), ptr[2]);

            left_shift_uint192(ptr.get(), 64, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0x5555555555555554), ptr[1]);
            REQUIRE(T(0xAAAAAAAAAAAAAAA9), ptr[2]);
        }

        TEST(UIntArith, RightShiftUInt)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            right_shift_uint(ptr.get(), 0, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            right_shift_uint(ptr.get(), 10, 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            right_shift_uint(ptr.get(), 10, 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 0x5555555555555555;
            ptr[1] = 0xAAAAAAAAAAAAAAAA;
            right_shift_uint(ptr.get(), 0, 2, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            right_shift_uint(ptr.get(), 0, 2, ptr.get());
            REQUIRE(T(0x5555555555555555), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr[1]);
            right_shift_uint(ptr.get(), 1, 2, ptr2.get());
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0x5555555555555555), ptr2[1]);
            right_shift_uint(ptr.get(), 2, 2, ptr2.get());
            REQUIRE(T(0x9555555555555555), ptr2[0]);
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr2[1]);
            right_shift_uint(ptr.get(), 64, 2, ptr2.get());
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            right_shift_uint(ptr.get(), 65, 2, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            right_shift_uint(ptr.get(), 127, 2, ptr2.get());
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);

            right_shift_uint(ptr.get(), 2, 2, ptr.get());
            REQUIRE(T(0x9555555555555555), ptr[0]);
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr[1]);
            right_shift_uint(ptr.get(), 64, 2, ptr.get());
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr[0]);
            REQUIRE(T(0), ptr[1]);
        }

        TEST(UIntArith, RightShiftUInt128)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            right_shift_uint128(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            right_shift_uint128(ptr.get(), 10, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            right_shift_uint128(ptr.get(), 10, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 0x5555555555555555;
            ptr[1] = 0xAAAAAAAAAAAAAAAA;
            right_shift_uint128(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            right_shift_uint128(ptr.get(), 0, ptr.get());
            REQUIRE(T(0x5555555555555555), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr[1]);
            right_shift_uint128(ptr.get(), 1, ptr2.get());
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0x5555555555555555), ptr2[1]);
            right_shift_uint128(ptr.get(), 2, ptr2.get());
            REQUIRE(T(0x9555555555555555), ptr2[0]);
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr2[1]);
            right_shift_uint128(ptr.get(), 64, ptr2.get());
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            right_shift_uint128(ptr.get(), 65, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            right_shift_uint128(ptr.get(), 127, ptr2.get());
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);

            right_shift_uint128(ptr.get(), 2, ptr.get());
            REQUIRE(T(0x9555555555555555), ptr[0]);
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr[1]);
            right_shift_uint128(ptr.get(), 64, ptr.get());
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr[0]);
            REQUIRE(T(0), ptr[1]);
        }

        TEST(UIntArith, RightShiftUInt192)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(3, pool));
            auto ptr2(allocate_uint(3, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr[2] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr2[2] = T(-1);
            right_shift_uint192(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr2[2] = T(-1);
            right_shift_uint192(ptr.get(), 10, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);
            right_shift_uint192(ptr.get(), 10, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);
            REQUIRE(T(0), ptr[2]);

            ptr[0] = 0x5555555555555555;
            ptr[1] = 0xAAAAAAAAAAAAAAAA;
            ptr[2] = 0xCDCDCDCDCDCDCDCD;

            right_shift_uint192(ptr.get(), 0, ptr2.get());
            REQUIRE(T(0x5555555555555555), ptr2[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[1]);
            REQUIRE(T(0xCDCDCDCDCDCDCDCD), ptr2[2]);
            right_shift_uint192(ptr.get(), 0, ptr.get());
            REQUIRE(T(0x5555555555555555), ptr[0]);
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr[1]);
            REQUIRE(T(0xCDCDCDCDCDCDCDCD), ptr[2]);
            right_shift_uint192(ptr.get(), 1, ptr2.get());
            REQUIRE(T(0x2AAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0xD555555555555555), ptr2[1]);
            REQUIRE(T(0x66E6E6E6E6E6E6E6), ptr2[2]);
            right_shift_uint192(ptr.get(), 2, ptr2.get());
            REQUIRE(T(0x9555555555555555), ptr2[0]);
            REQUIRE(T(0x6AAAAAAAAAAAAAAA), ptr2[1]);
            REQUIRE(T(0x3373737373737373), ptr2[2]);
            right_shift_uint192(ptr.get(), 64, ptr2.get());
            REQUIRE(T(0xAAAAAAAAAAAAAAAA), ptr2[0]);
            REQUIRE(T(0xCDCDCDCDCDCDCDCD), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);
            right_shift_uint192(ptr.get(), 65, ptr2.get());
            REQUIRE(T(0xD555555555555555), ptr2[0]);
            REQUIRE(T(0x66E6E6E6E6E6E6E6), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);
            right_shift_uint192(ptr.get(), 191, ptr2.get());
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            REQUIRE(T(0), ptr2[2]);

            right_shift_uint192(ptr.get(), 2, ptr.get());
            REQUIRE(T(0x9555555555555555), ptr[0]);
            REQUIRE(T(0x6AAAAAAAAAAAAAAA), ptr[1]);
            REQUIRE(T(0x3373737373737373), ptr[2]);
            right_shift_uint192(ptr.get(), 64, ptr.get());
            REQUIRE(T(0x6AAAAAAAAAAAAAAA), ptr[0]);
            REQUIRE(T(0x3373737373737373), ptr[1]);
            REQUIRE(T(0), ptr[2]);
        }

        TEST(UIntArith, HalfRoundUpUInt)
        {
            half_round_up_uint(nptr, 0, nptr);

            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            half_round_up_uint(ptr.get(), 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            half_round_up_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 1;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            half_round_up_uint(ptr.get(), 2, ptr2.get());
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            half_round_up_uint(ptr.get(), 2, ptr.get());
            REQUIRE(1, ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 2;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            half_round_up_uint(ptr.get(), 2, ptr2.get());
            REQUIRE(1, ptr2[0]);
            REQUIRE(T(0), ptr2[1]);
            half_round_up_uint(ptr.get(), 2, ptr.get());
            REQUIRE(1, ptr[0]);
            REQUIRE(T(0), ptr[1]);

            ptr[0] = 3;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            half_round_up_uint(ptr.get(), 2, ptr2.get());
            REQUIRE(T(2), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);

            ptr[0] = 4;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            half_round_up_uint(ptr.get(), 2, ptr2.get());
            REQUIRE(T(2), ptr2[0]);
            REQUIRE(T(0), ptr2[1]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            half_round_up_uint(ptr.get(), 2, ptr2.get());
            REQUIRE(T(0), ptr2[0]);
            REQUIRE(T(0x8000000000000000), ptr2[1]);
            half_round_up_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(0x8000000000000000), ptr[1]);
        }

        TEST(UIntArith, NotUInt)
        {
            not_uint(nptr, 0, nptr);

            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            ptr[0] = T(-1);
            ptr[1] = 0;
            not_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0), ptr[0]);
            REQUIRE(T(T(-1)), ptr[1]);

            ptr[0] = 0xFFFFFFFF00000000;
            ptr[1] = 0xFFFF0000FFFF0000;
            not_uint(ptr.get(), 2, ptr.get());
            REQUIRE(T(0x00000000FFFFFFFF), ptr[0]);
            REQUIRE(T(0x0000FFFF0000FFFF), ptr[1]);
        }

        TEST(UIntArith, AndUIntUInt)
        {
            and_uint_uint(nptr, nptr, 0, nptr);

            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            auto ptr3(allocate_uint(2, pool));
            ptr[0] = T(-1);
            ptr[1] = 0;
            ptr2[0] = 0;
            ptr2[1] = T(-1);
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            and_uint_uint(ptr.get(), ptr2.get(), 2, ptr3.get());
            REQUIRE(T(0), ptr3[0]);
            REQUIRE(T(0), ptr3[1]);

            ptr[0] = 0xFFFFFFFF00000000;
            ptr[1] = 0xFFFF0000FFFF0000;
            ptr2[0] = 0x0000FFFF0000FFFF;
            ptr2[1] = 0xFF00FF00FF00FF00;
            ptr3[0] = 0;
            ptr3[1] = 0;
            and_uint_uint(ptr.get(), ptr2.get(), 2, ptr3.get());
            REQUIRE(T(0x0000FFFF00000000), ptr3[0]);
            REQUIRE(T(0xFF000000FF000000), ptr3[1]);
            and_uint_uint(ptr.get(), ptr2.get(), 2, ptr.get());
            REQUIRE(T(0x0000FFFF00000000), ptr[0]);
            REQUIRE(T(0xFF000000FF000000), ptr[1]);
        }

        TEST(UIntArith, OrUIntUInt)
        {
            or_uint_uint(nptr, nptr, 0, nptr);

            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            auto ptr3(allocate_uint(2, pool));
            ptr[0] = T(-1);
            ptr[1] = 0;
            ptr2[0] = 0;
            ptr2[1] = T(-1);
            ptr3[0] = 0;
            ptr3[1] = 0;
            or_uint_uint(ptr.get(), ptr2.get(), 2, ptr3.get());
            REQUIRE(T(T(-1)), ptr3[0]);
            REQUIRE(T(T(-1)), ptr3[1]);

            ptr[0] = 0xFFFFFFFF00000000;
            ptr[1] = 0xFFFF0000FFFF0000;
            ptr2[0] = 0x0000FFFF0000FFFF;
            ptr2[1] = 0xFF00FF00FF00FF00;
            ptr3[0] = 0;
            ptr3[1] = 0;
            or_uint_uint(ptr.get(), ptr2.get(), 2, ptr3.get());
            REQUIRE(T(0xFFFFFFFF0000FFFF), ptr3[0]);
            REQUIRE(T(0xFFFFFF00FFFFFF00), ptr3[1]);
            or_uint_uint(ptr.get(), ptr2.get(), 2, ptr.get());
            REQUIRE(T(0xFFFFFFFF0000FFFF), ptr[0]);
            REQUIRE(T(0xFFFFFF00FFFFFF00), ptr[1]);
        }

        TEST(UIntArith, XorUIntUInt)
        {
            xor_uint_uint(nptr, nptr, 0, nptr);

            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(2, pool));
            auto ptr2(allocate_uint(2, pool));
            auto ptr3(allocate_uint(2, pool));
            ptr[0] = T(-1);
            ptr[1] = 0;
            ptr2[0] = 0;
            ptr2[1] = T(-1);
            ptr3[0] = 0;
            ptr3[1] = 0;
            xor_uint_uint(ptr.get(), ptr2.get(), 2, ptr3.get());
            REQUIRE(T(T(-1)), ptr3[0]);
            REQUIRE(T(T(-1)), ptr3[1]);

            ptr[0] = 0xFFFFFFFF00000000;
            ptr[1] = 0xFFFF0000FFFF0000;
            ptr2[0] = 0x0000FFFF0000FFFF;
            ptr2[1] = 0xFF00FF00FF00FF00;
            ptr3[0] = 0;
            ptr3[1] = 0;
            xor_uint_uint(ptr.get(), ptr2.get(), 2, ptr3.get());
            REQUIRE(T(0xFFFF00000000FFFF), ptr3[0]);
            REQUIRE(T(0x00FFFF0000FFFF00), ptr3[1]);
            xor_uint_uint(ptr.get(), ptr2.get(), 2, ptr.get());
            REQUIRE(T(0xFFFF00000000FFFF), ptr[0]);
            REQUIRE(T(0x00FFFF0000FFFF00), ptr[1]);
        }

        TEST(UIntArith, MultiplyUInt64Generic)
        {
            unsigned long long result[2];

            multiply_uint64_generic(T(0), T(0), result);
            REQUIRE(T(0), result[0]);
            REQUIRE(T(0), result[1]);
            multiply_uint64_generic(T(0), 1, result);
            REQUIRE(T(0), result[0]);
            REQUIRE(T(0), result[1]);
            multiply_uint64_generic(1, T(0), result);
            REQUIRE(T(0), result[0]);
            REQUIRE(T(0), result[1]);
            multiply_uint64_generic(1, 1, result);
            REQUIRE(1, result[0]);
            REQUIRE(T(0), result[1]);
            multiply_uint64_generic(0x10000000T(0), 0xFAFABABA, result);
            REQUIRE(0xFAFABABA0000000T(0), result[0]);
            REQUIRE(T(0), result[1]);
            multiply_uint64_generic(0x100000000T(0), 0xFAFABABA, result);
            REQUIRE(0xAFABABA00000000T(0), result[0]);
            REQUIRE(0xF, result[1]);
            multiply_uint64_generic(1111222233334444, 5555666677778888, result);
            REQUIRE(4140785562324247136, result[0]);
            REQUIRE(334670460471, result[1]);
        }
#endif

template<typename T>
void UIntArith_MultiplyUInt(void)
        {
            T result[2];

            multiply_uint64(T(0), T(0), result);
            REQUIRE(T(0) == result[0]);
            REQUIRE(T(0) == result[1]);
            multiply_uint64(T(0), 1, result);
            REQUIRE(T(0) == result[0]);
            REQUIRE(T(0) == result[1]);
            multiply_uint64(T(1), T(0), result);
            REQUIRE(T(0) == result[0]);
            REQUIRE(T(0) == result[1]);
            multiply_uint64(T(1), T(1), result);
            REQUIRE(T(1) == result[0]);
            REQUIRE(T(0) == result[1]);
            {
                auto opnd1 = (T)(sizeof(T) == 8) ? T(0x100000000) : T(0x10000);
                auto opnd2 = (T)(sizeof(T) == 8) ? T(0xFAFABABA) : T(0xFABA);
                auto res = (T)(sizeof(T) == 8) ? T(0xFAFABABA00000000) : T(0xFABA0000);
                multiply_uint64(opnd1, opnd2, result);
                REQUIRE(res == result[0]);
                REQUIRE(T(0) == result[1]);
            }

            {
                auto opnd1 = (T)(sizeof(T) == 8) ? T(0x1000000000) : T(0x100000);
                auto opnd2 = (T)(sizeof(T) == 8) ? T(0xFAFABABA) : T(0xFABA);
                auto res = (T)(sizeof(T) == 8) ? T(0xAFABABA000000000) : T(0xABA00000);
                multiply_uint64(opnd1, opnd2, result);
                REQUIRE(res == result[0]);
                REQUIRE(T(0xF) == result[1]);
            }

            {
                auto opnd1 = (T)(sizeof(T) == 8) ? T(1111222233334444) : T(11223344);
                auto opnd2 = (T)(sizeof(T) == 8) ? T(5555666677778888) : T(55667788);
                uint64_t res64;
                if (sizeof(T) == 4)
                {
                    res64 = uint64_t(11223344) * uint64_t(55667788);
                    //cout << T(bot_half_bits(res64)) << ", " << T((top_half_bits(res64) >> 32)) << endl;
                }

                auto res0 = (T)(sizeof(T) == 8) ? T(4140785562324247136) : T(bot_half_bits(res64));
                auto res1 = (T)(sizeof(T) == 8) ? T(334670460471) : T((top_half_bits(res64) >> 32));
                multiply_uint64(opnd1, opnd2, result);
                REQUIRE(res0 == result[0]);
                REQUIRE(res1 == result[1]);
            }

        }

TEST_CASE("XeHE UIntArith MultiplyUInt", "[MultiplyUInt][cpu][XeHE]")
{
    UIntArith_MultiplyUInt<uint64_t>();
    std::cout << "-------XeHE UIntArith MultiplyUInt64 tests passed-------" << std::endl;

    UIntArith_MultiplyUInt<uint32_t>();
    std::cout << "-------XeHE UIntArith MultiplyUInt32 tests passed-------" << std::endl;
}


template<typename T>
void UIntArith_MultiplyUIntUInt(void)
{
            auto s_ptr = allocate_uint<T>(2);
            auto s_ptr2 = allocate_uint<T>(2);
            auto s_ptr3 = allocate_uint<T>(4);
            auto ptr = s_ptr.get();
            auto ptr2 = s_ptr2.get();
            auto ptr3 = s_ptr3.get();

            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = 0;
            ptr2[1] = 0;
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            ptr3[2] = T(-1);
            ptr3[3] = T(-1);
            multiply_uint_uint<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(0) == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = 0;
            ptr2[1] = 0;
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            ptr3[2] = T(-1);
            ptr3[3] = T(-1);
            multiply_uint_uint<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(0) == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = 1;
            ptr2[1] = 0;
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_uint_uint<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(T(-1)) == ptr3[0]);
            REQUIRE(T(T(-1)) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = 0;
            ptr2[1] = 1;
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_uint_uint<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(0) == ptr3[0]);
            REQUIRE(T(T(-1)) == ptr3[1]);
            REQUIRE(T(T(-1)) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_uint_uint<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(1 == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);
            REQUIRE(T(-2) == ptr3[2]);
            REQUIRE(T(T(-1)) == ptr3[3]);

            ptr[0] = (T)(sizeof(T) == 8) ? T(9756571004902751654ul) : T(uint64_t(9756571004902751654ul) & 0x00000000ffffffff);
            ptr[1] = (T)(sizeof(T) == 8) ? T(731952007397389984) : T(uint64_t(9756571004902751654ul) >> 32);
            ptr2[0] = (T)(sizeof(T) == 8) ? T(701538366196406307) : T(uint64_t(701538366196406307) & 0x00000000ffffffff);
            ptr2[1] = (T)(sizeof(T) == 8) ? T(1699883529753102283) : T(uint64_t(701538366196406307) >> 32);
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_uint_uint<T>(ptr, ptr2, 2, ptr3);

            if (sizeof(T) == 8)
            {
                REQUIRE(T(9585656442714717618ul) == ptr3[0]);
                REQUIRE(T(1817697005049051848) == ptr3[1]);
                REQUIRE(T(14447416709120365380ul) == ptr3[2]);
                REQUIRE(T(67450014862939159) == ptr3[3]);
            }
            else if (sizeof(T) == 4)
            {
                REQUIRE(T(1727135154) == ptr3[0]);
                REQUIRE(T(2231834559) == ptr3[1]);
                REQUIRE(T(3188594246) == ptr3[2]);
                REQUIRE(T(86391109) == ptr3[3]);
            }

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_uint_uint<T>(ptr, 2, ptr2, 1, 2, ptr3);
            REQUIRE(1 == ptr3[0]);
            REQUIRE(T(T(-1)) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_uint_uint<T>(ptr, 2, ptr2, 1, 3, ptr3);
            REQUIRE(1 == ptr3[0]);
            REQUIRE(T(T(-1)) == ptr3[1]);
            REQUIRE(T(-2) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            ptr[0] = T(-1);
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = 0;
            ptr3[1] = 0;
            ptr3[2] = 0;
            ptr3[3] = 0;
            multiply_truncate_uint_uint<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(1 == ptr3[0]);
            REQUIRE(T(T(-1)) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);
        }

        TEST_CASE("XeHE UIntArith MultiplyUIntUInt", "[MultiplyUIntUint][cpu][XeHE]")
        {
            UIntArith_MultiplyUIntUInt<uint64_t>();
            std::cout << "-------XeHE UIntArith MultiplyUIntUInt64 tests passed-------" << std::endl;

            UIntArith_MultiplyUIntUInt<uint32_t>();
            std::cout << "-------XeHE UIntArith MultiplyUIntUint32 tests passed-------" << std::endl;
        }


#if 0
        TEST(UIntArith, MultiplyUIntUInt64)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto ptr(allocate_uint(3, pool));
            auto result(allocate_uint(4, pool));

            ptr[0] = 0;
            ptr[1] = 0;
            ptr[2] = 0;
            multiply_uint_uint64(ptr.get(), 3, T(0), 4, result.get());
            REQUIRE(T(0), result[0]);
            REQUIRE(T(0), result[1]);
            REQUIRE(T(0), result[2]);
            REQUIRE(T(0), result[3]);

            ptr[0] = 0xFFFFFFFFF;
            ptr[1] = 0xAAAAAAAAA;
            ptr[2] = 0x111111111;
            multiply_uint_uint64(ptr.get(), 3, T(0), 4, result.get());
            REQUIRE(T(0), result[0]);
            REQUIRE(T(0), result[1]);
            REQUIRE(T(0), result[2]);
            REQUIRE(T(0), result[3]);

            ptr[0] = 0xFFFFFFFFF;
            ptr[1] = 0xAAAAAAAAA;
            ptr[2] = 0x111111111;
            multiply_uint_uint64(ptr.get(), 3, 1, 4, result.get());
            REQUIRE(0xFFFFFFFFF, result[0]);
            REQUIRE(0xAAAAAAAAA, result[1]);
            REQUIRE(0x111111111, result[2]);
            REQUIRE(T(0), result[3]);

            ptr[0] = 0xFFFFFFFFF;
            ptr[1] = 0xAAAAAAAAA;
            ptr[2] = 0x111111111;
            multiply_uint_uint64(ptr.get(), 3, 0x1000T(0), 4, result.get());
            REQUIRE(0xFFFFFFFFF000T(0), result[0]);
            REQUIRE(0xAAAAAAAAA000T(0), result[1]);
            REQUIRE(0x111111111000T(0), result[2]);
            REQUIRE(T(0), result[3]);

            ptr[0] = 0xFFFFFFFFF;
            ptr[1] = 0xAAAAAAAAA;
            ptr[2] = 0x111111111;
            multiply_uint_uint64(ptr.get(), 3, 0x10000000T(0), 4, result.get());
            REQUIRE(0xFFFFFFFF0000000T(0), result[0]);
            REQUIRE(0xAAAAAAAA0000000F, result[1]);
            REQUIRE(0x111111110000000A, result[2]);
            REQUIRE(1, result[3]);

            ptr[0] = 5656565656565656;
            ptr[1] = 3434343434343434;
            ptr[2] = 1212121212121212;
            multiply_uint_uint64(ptr.get(), 3, 7878787878787878, 4, result.get());
            REQUIRE(889137003211615656T(0), result[0]);
            REQUIRE(127835914414679452, result[1]);
            REQUIRE(9811042505314082702, result[2]);
            REQUIRE(517709026347, result[3]);
        }

#endif
        template<typename T>
        void UIntArith_DivideUIntUInt(void)
        //TEST(UIntArith, DivideUIntUInt)
        {

            divide_uint_uint_inplace<T>(nullptr, nullptr, 0, nullptr);
            divide_uint_uint<T>(nullptr, nullptr, 0, nullptr, nullptr);

            auto s_ptr = allocate_uint<T>(4);
            auto s_ptr2 = allocate_uint<T>(4);
            auto s_ptr3 = allocate_uint<T>(4);
            auto s_ptr4 = allocate_uint<T>(4);
            auto ptr = s_ptr.get();
            auto ptr2 = s_ptr2.get();
            auto ptr3 = s_ptr3.get();
            auto ptr4 = s_ptr4.get();
            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = 0;
            ptr2[1] = 1;
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            divide_uint_uint_inplace<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(0) == ptr[0]);
            REQUIRE(T(0) == ptr[1]);
            REQUIRE(T(0) == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);

            ptr[0] = 0;
            ptr[1] = 0;
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            divide_uint_uint_inplace<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(0) == ptr[0]);
            REQUIRE(T(0) == ptr[1]);
            REQUIRE(T(0) == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);

            ptr[0] = T(-2);
            ptr[1] = T(-1);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            divide_uint_uint_inplace<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(-2) == ptr[0]);
            REQUIRE(T(T(-1)) == ptr[1]);
            REQUIRE(T(0) == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);

            ptr[0] = T(-1);
            ptr[1] = T(-1);
            ptr2[0] = T(-1);
            ptr2[1] = T(-1);
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            divide_uint_uint_inplace<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(0) == ptr[0]);
            REQUIRE(T(0) == ptr[1]);
            REQUIRE(1 == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);

            ptr[0] = 14;
            ptr[1] = 0;
            ptr2[0] = 3;
            ptr2[1] = 0;
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            divide_uint_uint_inplace<T>(ptr, ptr2, 2, ptr3);
            REQUIRE(T(2) == ptr[0]);
            REQUIRE(T(0) == ptr[1]);
            REQUIRE(T(4) == ptr3[0]);
            REQUIRE(T(0) == ptr3[1]);

            ptr[0] = 9585656442714717620ul;
            ptr[1] = 1817697005049051848;
            ptr[2] = 14447416709120365380ul;
            ptr[3] = 67450014862939159;
            ptr2[0] = 701538366196406307;
            ptr2[1] = 1699883529753102283;
            ptr2[2] = 0;
            ptr2[3] = 0;
            ptr3[0] = T(-1);
            ptr3[1] = T(-1);
            ptr3[2] = T(-1);
            ptr3[3] = T(-1);
            ptr4[0] = T(-1);
            ptr4[1] = T(-1);
            ptr4[2] = T(-1);
            ptr4[3] = T(-1);
            divide_uint_uint<T>(ptr, ptr2, 4, ptr3, ptr4);
            REQUIRE(T(2) ==  ptr4[0]);
            REQUIRE(T(0) == ptr4[1]);
            REQUIRE(T(0) == ptr4[2]);
            REQUIRE(T(0) == ptr4[3]);
            REQUIRE(T(9756571004902751654ul) == ptr3[0]);
            REQUIRE(T(731952007397389984) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);

            divide_uint_uint_inplace<T>(ptr, ptr2, 4, ptr3);
            REQUIRE(T(2) == ptr[0]);
            REQUIRE(T(0) == ptr[1]);
            REQUIRE(T(0) == ptr[2]);
            REQUIRE(T(0) == ptr[3]);
            REQUIRE(T(9756571004902751654ul) == ptr3[0]);
            REQUIRE(T(731952007397389984) == ptr3[1]);
            REQUIRE(T(0) == ptr3[2]);
            REQUIRE(T(0) == ptr3[3]);
        }

        TEST_CASE("XeHE UIntArith DivideUIntUInt", "[DivideUIntUInt][cpu][XeHE]")
        {
            UIntArith_DivideUIntUInt<uint64_t>();
            std::cout << "-------XeHE UIntArith  DivideUIntUIn64 tests passed-------" << std::endl;

            //UIntArith_MultiplyUIntUInt<uint32_t>();
            //std::cout << "-------XeHE UIntArith MultiplyUIntUint32 tests passed-------" << std::endl;
        }

#if 0
        TEST(UIntArith, DivideUInt128UInt64)
        {
            uint64_t input[2];
            uint64_t quotient[2];

            input[0] = 0;
            input[1] = 0;
            divide_uint128_uint64_inplace(input, 1, quotient);
            REQUIRE(T(0), input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(T(0), quotient[0]);
            REQUIRE(T(0), quotient[1]);

            input[0] = 1;
            input[1] = 0;
            divide_uint128_uint64_inplace(input, 1, quotient);
            REQUIRE(T(0), input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(1, quotient[0]);
            REQUIRE(T(0), quotient[1]);

            input[0] = 0x10101010;
            input[1] = 0x2B2B2B2B;
            divide_uint128_uint64_inplace(input, 0x100T(0), quotient);
            REQUIRE(0x1T(0), input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(0xB2B0000000010101, quotient[0]);
            REQUIRE(0x2B2B2, quotient[1]);

            input[0] = 1212121212121212;
            input[1] = 3434343434343434;
            divide_uint128_uint64_inplace(input, 5656565656565656, quotient);
            REQUIRE(5252525252525252, input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(11199808901895084909, quotient[0]);
            REQUIRE(T(0), quotient[1]);
        }

        TEST(UIntArith, DivideUInt192UInt64)
        {
            uint64_t input[3];
            uint64_t quotient[3];

            input[0] = 0;
            input[1] = 0;
            input[2] = 0;
            divide_uint192_uint64_inplace(input, 1, quotient);
            REQUIRE(T(0), input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(T(0), input[2]);
            REQUIRE(T(0), quotient[0]);
            REQUIRE(T(0), quotient[1]);
            REQUIRE(T(0), quotient[2]);

            input[0] = 1;
            input[1] = 0;
            input[2] = 0;
            divide_uint192_uint64_inplace(input, 1, quotient);
            REQUIRE(T(0), input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(T(0), input[2]);
            REQUIRE(1, quotient[0]);
            REQUIRE(T(0), quotient[1]);
            REQUIRE(T(0), quotient[2]);

            input[0] = 0x10101010;
            input[1] = 0x2B2B2B2B;
            input[2] = 0xF1F1F1F1;
            divide_uint192_uint64_inplace(input, 0x100T(0), quotient);
            REQUIRE(0x1T(0), input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(T(0), input[2]);
            REQUIRE(0xB2B0000000010101, quotient[0]);
            REQUIRE(0x1F1000000002B2B2, quotient[1]);
            REQUIRE(0xF1F1F, quotient[2]);

            input[0] = 1212121212121212;
            input[1] = 3434343434343434;
            input[2] = 5656565656565656;
            divide_uint192_uint64_inplace(input, 7878787878787878, quotient);
            REQUIRE(7272727272727272, input[0]);
            REQUIRE(T(0), input[1]);
            REQUIRE(T(0), input[2]);
            REQUIRE(17027763760347278414, quotient[0]);
            REQUIRE(13243816258047883211, quotient[1]);
            REQUIRE(T(0), quotient[2]);
        }

        TEST(UIntArith, ExponentiateUInt)
        {
            MemoryPool &pool = *global_variables::global_memory_pool;
            auto input(allocate_zero_uint(2, pool));
            auto result(allocate_zero_uint(8, pool));

            result[0] = 1, result[1] = 2, result[2] = 3, result[3] = 4;
            result[4] = 5, result[5] = 6, result[6] = 7, result[7] = 8;

            uint64_t exponent[2]{ 0, 0 };

            input[0] = 0xFFF;
            input[1] = 0;
            exponentiate_uint(input.get(), 2, exponent, 1, 1, result.get(), pool);
            REQUIRE(1, result[0]);
            REQUIRE(2, result[1]);

            exponentiate_uint(input.get(), 2, exponent, 1, 2, result.get(), pool);
            REQUIRE(1, result[0]);
            REQUIRE(T(0), result[1]);

            exponentiate_uint(input.get(), 1, exponent, 1, 4, result.get(), pool);
            REQUIRE(1, result[0]);
            REQUIRE(T(0), result[1]);
            REQUIRE(T(0), result[2]);
            REQUIRE(T(0), result[3]);

            input[0] = 123;
            exponent[0] = 5;
            exponentiate_uint(input.get(), 1, exponent, 2, 2, result.get(), pool);
            REQUIRE(28153056843, result[0]);
            REQUIRE(T(0), result[1]);

            input[0] = 1;
            exponent[0] = 1;
            exponent[1] = 1;
            exponentiate_uint(input.get(), 1, exponent, 2, 2, result.get(), pool);
            REQUIRE(1, result[0]);
            REQUIRE(T(0), result[1]);

            input[0] = 0;
            input[1] = 1;
            exponent[0] = 7;
            exponent[1] = 0;
            exponentiate_uint(input.get(), 2, exponent, 2, 8, result.get(), pool);
            REQUIRE(T(0), result[0]);
            REQUIRE(T(0), result[1]);
            REQUIRE(T(0), result[2]);
            REQUIRE(T(0), result[3]);
            REQUIRE(T(0), result[4]);
            REQUIRE(T(0), result[5]);
            REQUIRE(T(0), result[6]);
            REQUIRE(1, result[7]);

            input[0] = 121212;
            input[1] = 343434;
            exponent[0] = 3;
            exponent[1] = 0;
            exponentiate_uint(input.get(), 2, exponent, 2, 8, result.get(), pool);
            REQUIRE(1780889000200128, result[0]);
            REQUIRE(15137556501701088, result[1]);
            REQUIRE(42889743421486416, result[2]);
            REQUIRE(40506979898070504, result[3]);
            REQUIRE(T(0), result[4]);
            REQUIRE(T(0), result[5]);
            REQUIRE(T(0), result[6]);
            REQUIRE(T(0), result[7]);
        }

        TEST(UIntArith, ExponentiateUInt64)
        {
            REQUIRE(T(0), exponentiate_uint64(T(0), 1));
            REQUIRE(1, exponentiate_uint64(1, T(0)));
            REQUIRE(T(0), exponentiate_uint64(T(0), T(-1)));
            REQUIRE(1, exponentiate_uint64(T(-1), T(0)));
            REQUIRE(25, exponentiate_uint64(5, 2));
            REQUIRE(32, exponentiate_uint64(2, 5));
            REQUIRE(0x100000000000000T(0), exponentiate_uint64(0x1T(0), 15));
            REQUIRE(T(0), exponentiate_uint64(0x1T(0), 16));
            REQUIRE(12389286314587456613, exponentiate_uint64(123456789, 13));
        }

#endif