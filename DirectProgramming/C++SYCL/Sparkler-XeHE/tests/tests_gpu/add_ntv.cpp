/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

// TODO: enable (uncomment) catch benchmarking later, when needed
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "catch2/catch.hpp"
#include <array>
#include <iostream>
#include <iso646.h>
// #include "util/defines.h"



#include "../src/include/native/xe_uintarith_core.hpp"

#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

#ifdef BUILD_WITH_IGPU



template<typename T>
class kernel_add_ntv_uint;

template<typename T>
static void
add_ntv_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {


    // GPU part setting
    auto d_A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_Crry = cl::sycl::malloc_shared<T>(num_workitems, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_C(num_workitems);
    std::vector<T> h_Crry(num_workitems);
    std::vector<T> hd_C(num_workitems);
    std::vector<T> hd_Crry(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    for (int i = 0; i < num_workitems; i++) {
        d_A[i] = T(dis(gen));
        d_B[i] = T(dis(gen));
        h_A[i] = d_A[i];
        h_B[i] = d_B[i];
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Bin.size());
        for (int i = 0; i < Ain.size(); i++) {
            d_A[i] = Ain[i];
            d_B[i] = Bin[i];
            h_A[i] = d_A[i];
            h_B[i] = d_B[i];
        }
    }

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << "A[" << i << "] = " << d_A[i] << " B[" << i << "] = " << d_B[i] << "\n";
        }
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_add_ntv_uint<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];
            
                d_Crry[i] = xehe::native::add_uint<T>(d_A[i], d_B[i], d_C + i);

            });
        });

        queue.submit([&](sycl::handler& h) {
            // copy deviceArray to hostArray
            h.memcpy(hd_C.data(), d_C, hd_C.size()*sizeof(T));
        });

        queue.submit([&](sycl::handler& h) {
            // copy deviceArray to hostArray
            h.memcpy(hd_Crry.data(), d_Crry, hd_Crry.size()*sizeof(T));
        }).wait();

        if (!benchmark) {

#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << h_A[i] << " + " << h_B[i] << " = " << hd_C[i] << " with crry " << hd_Crry[i] << "\n";
        }
#endif


        for (int i = 0; i < range_size; i++)
        {   
            h_Crry[i] = xehe::native::add_uint<T>(h_A[i], h_B[i], h_C.data() + i);
        }
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << h_A[i] << " x " << h_B[i] << " = " << h_C[i] << " with carry " << h_Crry[i]  << "\n";
        }
#endif

        bool success = true;
        for (int i = 0; i < range_size && success; i++)
        {
#ifdef WIN32
            if (h_C[i] != hd_C[i] || h_Crry[i] != hd_Crry[i])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[i] << " d " << hd_C[i] << std::endl;
            }
#else
            REQUIRE((h_C[i] == hd_C[i] && h_Crry[i] == hd_Crry[i]));
#endif
        }

#ifdef WIN32
        if (success)
        {
            std::cout << "Success" << std::endl;
        }
#endif
    }

#ifndef WIN32    
    else {

        BENCHMARK("simple T bit add")
        {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int i = it[0];
                    T A, B, C, Crry;
                    A = d_A[i];
                    B = d_B[i];
                    A += B;
                    for (int j = 0; j < inner_loops; ++j)
                    {
                        Crry = xehe::native::add_uint<T>(A, C, &C);
                    }
                    d_Crry[i] = Crry;
                    d_C[i] = C;
                    });
                }).wait();
        };


    }

#endif

}



template<typename T>
class kernel_sub_ntv_uint;

template<typename T>
static void
sub_ntv_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {


    // GPU part setting
    auto d_A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_Brrw = cl::sycl::malloc_shared<T>(num_workitems, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_C(num_workitems);
    std::vector<T> h_Brrw(num_workitems);
    std::vector<T> hd_C(num_workitems);
    std::vector<T> hd_Brrw(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    for (int i = 0; i < num_workitems; i++) {
        h_A[i] = T(dis(gen));
        h_B[i] = T(dis(gen));
        d_A[i] = h_A[i];
        d_B[i] = h_B[i];
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Bin.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_A[i] = Ain[i];
            h_B[i] = Bin[i];
            d_A[i] = h_A[i];
            d_B[i] = h_B[i];
        }
    }

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << "A[" << i << "] = " << h_A[i] << " B[" << i << "] = " << h_B[i] << "\n";
        }
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_sub_ntv_uint<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];


            d_Brrw[i] = xehe::native::sub_uint<T>(d_A[i], d_B[i], d_C + i);

            });
        });


    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray
        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        });

    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray
        h.memcpy(hd_Brrw.data(), d_Brrw, hd_Brrw.size() * sizeof(T));
        }).wait();

        if (!benchmark) {
#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
            {
                std::cout << h_A[i] << " + " << h_B[i] << " = " << hd_C[i] << " with borrow " << hd_Brrw[i] << "\n";
            }
#endif

            for (int i = 0; i < range_size; i++)
            {
                h_Brrw[i] = xehe::native::sub_uint<T>(h_A[i], h_B[i], h_C.data() + i);
            }
#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
            {
                std::cout << h_A[i] << " x " << h_B[i] << " = " << h_C[i] << " with borrow " << h_Brrw[i] << "\n";
            }
#endif

            bool success = true;
            for (int i = 0; i < range_size && success; i++)
            {
#ifdef WIN32
                if (h_C[i] != hd_C[i] || h_Brrw[i] != hd_Brrw[i])
                {
                    success = false;
                    std::cout << "Failed at " << i << " h " << h_C[i] << " d " << hd_C[i] << std::endl;
                }
#else
                REQUIRE((h_C[i] == hd_C[i] && h_Brrw[i] == hd_Brrw[i]));
#endif
            }

#ifdef WIN32
            if (success)
            {
                std::cout << "Success" << std::endl;
            }
#endif
        }

#ifndef WIN32    
        else {

            BENCHMARK("simple T bit sub")
            {
                return queue.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        int i = it[0];
                        T A, B, C, Brrw;
                        A = d_A[i];
                        B = d_B[i];
                        A += B;
                        for (int j = 0; j < inner_loops; ++j)
                        {
                            Brrw = xehe::native::sub_uint<T>(A, C, &C);
                        }
                        d_Brrw[i] = Brrw;
                        d_C[i] = C;

                        });
                    }).wait();
            };


        }

#endif

}




void Basic_static_native_uint_add(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

#ifndef WIN32
    SECTION("add 32 bit")
#else
    std::cout << "add 32 bit" << std::endl;
#endif
    {

        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        add_ntv_test<uint32_t>(queue, Ain, Bin);
    }

#ifndef WIN32
    SECTION("add 64 bit")
#else
    std::cout << "add 64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        add_ntv_test<uint64_t>(queue, Ain, Bin);
    }


}



void Basic_static_native_uint_sub(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

#ifndef WIN32
    SECTION("sub 32 bit")
#else
    std::cout << "sub 32 bit" << std::endl;
#endif
    {

        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        sub_ntv_test<uint32_t>(queue, Ain, Bin);
    }

#ifndef WIN32
    SECTION("sub 64 bit")
#else
    std::cout << "sub 64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        sub_ntv_test<uint64_t>(queue, Ain, Bin);
    }


}



#ifndef WIN32

TEST_CASE("Basic static native uint add", "[gpu][uintarith][add]") {
    Basic_static_native_uint_add();
}


TEST_CASE("Basic static native uint sub", "[gpu][uintarith][sub]") {
    Basic_static_native_uint_sub();
}

#endif
/*****************************************************************
 *
 *   PERF TESTS
 *
 *****************************************************************/

void Basic_static_bench_native_uint_add(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    size_t num_workitems = 100000;
#ifndef WIN32
    SECTION("32 bit add: 100K threads")
#else
    std::cout << "32 bit add: " << num_workitems << " threads" << std::endl;
#endif
    {
        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        add_ntv_test<uint32_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
    }



#ifndef WIN32
    SECTION("64 bit add: 100K threads")
#else
    std::cout << "64 bit add: " << num_workitems << " threads" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        add_ntv_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
    }



}


void Basic_static_bench_native_uint_sub(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    size_t num_workitems = 100000;
#ifndef WIN32
    SECTION("32 bit sub: 100K threads")
#else
    std::cout << "32 bit sub: " << num_workitems << " threads" << std::endl;
#endif
    {
        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        sub_ntv_test<uint32_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
    }



#ifndef WIN32
    SECTION("64 bit sub: 100K threads")
#else
    std::cout << "64 bit sub: " << num_workitems << " threads" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        sub_ntv_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
    }



}



#ifndef WIN32
TEST_CASE("Basic static bench native uint add", "[gpu][uintarith][add]") {
    Basic_static_bench_native_uint_add();
}

TEST_CASE("Basic static bench native uint sub", "[gpu][uintarith][sub]") {
    Basic_static_bench_native_uint_sub();
}

#endif



/* ---------------------------------------------------------
//                                   MODULAR ARITHMETICS
 ----------------------------------------------------------*/

template<typename T>
class kernel_negate_mod_ntv;

template<typename T>
static void negate_mod_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Min, size_t num_workitems = 10, size_t range_size = 10,
    bool benchmark = false, int inner_loops = 100) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto M = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_C(num_workitems);
    std::vector<T> h_M(num_workitems);
    std::vector<T> hd_C(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

    for (int i = 0; i < num_workitems; i++) {

        // modulo
        h_M[i] = M[i] = dis(gen) % ((sizeof(T) == 8) ? 0x7fffffffffffffffUL : 0x7fffffff);
        h_A[i] = A[i] = dis(gen) % M[i];
    }

    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Min.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_M[i] = M[i] = Min[i];
            h_A[i] = A[i] = Ain[i] % M[i];
        }
    }

    if (!benchmark) {
#ifdef VERBOSE_TEST        
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " M[" << i << "] = " << M[i] << "\n";
        std::cout << "\n";
#endif        
    }

    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_negate_mod_ntv<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];
            d_C[i] = xehe::native::neg_mod(A[i], M[i]);
            });
        });



    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray

        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        }).wait();

        if (!benchmark) {


            for (int i = 0; i < range_size; i++)
            {
                //h_C[i] = xehe::util::negate_uint_mod<T>(A[i], M[i]);

                h_C[i] = (h_A[i] == 0) ? 0 : (h_M[i] - h_A[i]);

            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << " negate " << h_A[i] << " mod " << h_M[i] << " = " << hd_C[i] << " Host: " << h_C[i] << "\n";
#endif

            bool success = true;
            for (int i = 0; i < range_size && success; i++)
            {
#ifdef WIN32
                if (h_C[i] != hd_C[i])
                {
                    success = false;
                    std::cout << "Failed at " << i << " h " << h_C[i] << " d " << hd_C[i] << std::endl;
                }
#else
                REQUIRE(h_C[i] == hd_C[i]);
#endif
            }

#ifdef WIN32
            if (success)
            {
                std::cout << "Success" << std::endl;
            }
#endif

        }
#ifndef WIN32
        else {
            BENCHMARK("negate bench") {
                return queue.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        int i = it[0];
                        T tC = 0;
                        auto tA = A[i];
                        auto tM = M[i];
                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tC += tA;
                            auto tA2 = tC >> 1;
                            tC = xehe::native::neg_mod(tA2, tM);
                        }

                        d_C[i] = tC;

                        });
                    }).wait();
            };
        }
#endif
}

static
void Basic_negate_mod_test(bool benchmark = false, int inner_loop = 100)
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
    int n_timed_workers = 1000000;

    if (!benchmark)
    {
#ifdef WIN32

        std::cout << "Test: mod negate" << std::endl;

#endif
#ifndef WIN32
        SECTION("64bit binary")
#endif
        {
#ifdef WIN32
            std::cout << "64bit binary" << std::endl;
#endif
            uint64_t modulus = 2;
            std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1 };
            std::vector<uint64_t> Min{ modulus, modulus + 1, modulus, modulus + 1, modulus };

            negate_mod_test<uint64_t>(queue, Ain, Min, n_workers, n_workers);
        };
#ifndef WIN32
        SECTION("32 bit small")
#endif
        {
#ifdef WIN32
            std::cout << "32 bit small" << std::endl;
#endif
            uint32_t modulus = 10;
            std::vector<uint32_t> Ain{ 0, 0, 1, 1, 7, 6, 7 };
            std::vector<uint32_t> Min;
            for (int i = 0; i < Ain.size(); i++)
            {
                Min.push_back(modulus + i);
            }

            negate_mod_test<uint32_t>(queue, Ain, Min, n_workers, n_workers);
        };
#ifndef WIN32
        SECTION("64 bit #1")
#endif
        {
#ifdef WIN32
            std::cout << "64 bit #1" << std::endl;
#endif
            uint64_t modulus = 2305843009211596801ULL;
            std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1, (1152921504605798400), (1152921504605798401),
                                      (1152921504605798401), (2305843009211596800) };
            std::vector<uint64_t> Min;
            for (int i = 0; i < Ain.size(); i++)
            {
                Min.push_back(modulus + i);
            }

            negate_mod_test<uint64_t>(queue, Ain, Min, n_workers, n_workers);
        }

    }
#ifdef WIN32
    if (benchmark)
    {
        std::cout << "Bench: mod negate" << std::endl;
    }
#endif
#ifndef WIN32
    SECTION("32 bit big")
#endif
    {
#ifdef WIN32
        std::cout << "32 bit big" << std::endl;
#endif
        n_workers = (!benchmark) ? n_workers : n_timed_workers * 2;
        uint32_t modulus = 0xFFFFFFF0;
        std::vector<uint32_t> Ain{ 0xFFFFFFF1, 0, 1, 1, 7, 6, 7 };
        std::vector<uint32_t> Min;
        for (int i = 0; i < Ain.size(); i++)
        {
            Min.push_back(modulus + i);
        }

        negate_mod_test<uint32_t>(queue, Ain, Min, n_workers, n_workers, benchmark, inner_loop);
    };

#ifndef WIN32
    SECTION("64 bit #2")
#endif
    {
#ifdef WIN32
        std::cout << "64 bit #2" << std::endl;
#endif
        n_workers = (!benchmark) ? n_workers : n_timed_workers;
        uint64_t modulus = 2305843009213693941ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798401, 1152921504605798401,
                                  2305843009211596800 };
        std::vector<uint64_t> Min;
        for (int i = 0; i < Ain.size(); i++)
        {
            Min.push_back(modulus + i);
        }

        negate_mod_test<uint64_t>(queue, Ain, Min, n_workers, n_workers, benchmark, inner_loop);
    }
}


#ifndef WIN32
TEST_CASE("Basic mod negate test", "[gpu][uintarithmod][uint64][mod_neg]")
{
    Basic_negate_mod_test();
}

TEST_CASE("Basic bench mod negate test", "[gpu][uintarithmod][uint64][mod_neg]")
{
    Basic_negate_mod_test(true);
}

#endif



template<typename T>
class kernel_add_mod_ntv;


template<typename T>
static void
add_mod_test(cl::sycl::queue& queue, T modulus, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto mod = cl::sycl::malloc_shared<T>(1, queue);
    auto m = static_cast<T>(modulus);
    mod[0] = m;
    std::vector<T> h_C(num_workitems);
    std::vector<T> hd_C(num_workitems);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;



    for (int i = 0; i < num_workitems; i++) {
        h_A[i] = A[i] = dis(gen) % m;
        h_B[i] = B[i] = dis(gen) % m;
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Bin.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_A[i] = A[i] = Ain[i] % m;
            h_B[i] = B[i] = Bin[i] % m;
        }
    }

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }


    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_add_mod_ntv<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];

            d_C[i] = xehe::native::add_mod<T>(A[i], B[i], mod[0]);

            });
        });

    queue.wait();


    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray
        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        }).wait();

        if (!benchmark) {

            for (int i = 0; i < range_size; i++)
            {
                h_C[i] = (h_A[i] + h_B[i]) % m; // xehe::native::add_mod<T>(A[i], B[i], m);
            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << h_A[i] << " + " << h_B[i] << " mod " << m << " = " << hd_C[i] << "\n";
#endif
            bool success = true;
            for (int i = 0; i < range_size && success; i++)
            {
#ifdef WIN32
                if (h_C[i] != hd_C[i])
                {
                    success = false;
                    std::cout << "Failed at " << i << " h " << h_C[i] << " d " << hd_C[i] << std::endl;
                }
#else

                REQUIRE(h_C[i] == hd_C[i]);
#endif
            }

#ifdef WIN32
            if (success)
            {
                std::cout << "Success" << std::endl;
            }
#endif
        }
#ifndef WIN32
        else
        {
            BENCHMARK("simple add mod") {
                return queue.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        int i = it[0];
                        T tA, tB, tC = 0;
                        tA = A[i];
                        tB = B[i];
                        tA += tB;
                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tC = xehe::native::add_mod<T>(tA, tC, mod[0]);
                        }

                        d_C[i] = tC;
                        });
                    }).wait();
            };
        }
#endif
}









static void Basic_uint64_mod_add_test(void)
{
#ifdef WIN32
    std::cout << "Basic uint64 mod add test" << std::endl;
#endif
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
#ifndef WIN32
    SECTION("mod 2")
#else
    std::cout << "mod 2" << std::endl;
#endif
    {
        uint64_t mod = 2;
        std::vector<uint64_t> Ain{0, 0, 1, 1};
        std::vector<uint64_t> Bin{0, 1, 0, 1};
        add_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }
#ifndef WIN32
    SECTION("mod 10")
#else
    std::cout << "mod 10" << std::endl;
#endif
    {
        uint64_t mod = 10;
        std::vector<uint64_t> Ain{0, 0, 1, 1, 3, 4, 6, 7};
        std::vector<uint64_t> Bin{0, 1, 0, 1, 8, 3, 7, 7};
        add_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }

#ifndef WIN32
    SECTION("mod 2305843009211596801ULL")
#else
    std::cout << "mod 2305843009211596801ULL"  << std::endl;
#endif
    {
        uint64_t mod = 2305843009211596801ULL;
        std::vector<uint64_t> Ain{0, 0, 1, 1, 1152921504605798400ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7};
        std::vector<uint64_t> Bin{0, 1, 0, 1, 1152921504605798401ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7};
        add_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }
}


static void Basic_uint32_mod_add_test(void)
{
#ifdef WIN32
    std::cout << "Basic uint32 mod add test" << std::endl;
#endif
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
#ifndef WIN32
    SECTION("mod 2")
#else
    std::cout << "mod 2"  << std::endl;
#endif
    {
        uint32_t mod = 2;
        std::vector<uint32_t> Ain{0, 0, 1, 1};
        std::vector<uint32_t> Bin{0, 1, 0, 1};
        add_mod_test<uint32_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }

#ifndef WIN32
    SECTION("mod 10")
#else
    std::cout << "mod 10"  << std::endl;
#endif
    {
        uint32_t mod = 10;
        std::vector<uint32_t> Ain{0, 0, 1, 1, 3, 4, 6, 7};
        std::vector<uint32_t> Bin{0, 1, 0, 1, 8, 3, 7, 7};
        add_mod_test<uint32_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }

}

static void Basic_uint_mod_add_test(void)
{
    Basic_uint32_mod_add_test();
    Basic_uint64_mod_add_test();
}


static void Basic_bench_mod_add_test(void)
{
#ifdef WIN32
    std::cout << "Basic bench mod add test" << std::endl;
#endif
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100000;

#ifndef WIN32
    SECTION("mod 32bit")
#else
    std::cout << "mod 32bit" << std::endl;
#endif
    {
        uint32_t mod = 10;
        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 3, 4, 6, 7 };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 8, 3, 7, 7 };
        add_mod_test<uint32_t>(queue, mod, Ain, Bin, n_workers, n_workers, true);
    }


#ifndef WIN32
    SECTION("mod 64bit")
#else
    std::cout << "mod 64bit" << std::endl;
#endif
    {
        uint64_t mod = 2305843009211596801ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 1152921504605798400ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 1152921504605798401ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        add_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers, true);
    }

}

#ifndef WIN32
TEST_CASE("Basic uint mod add test", "[gpu][uintarithmod][uint][mod_add]")
{
    Basic_uint_mod_add_test();
}


TEST_CASE("Basic bench mod add test", "[gpu][uintarithmod][uint][mod_add]")
{
    Basic_bench_mod_add_test();
}
#endif


/*----------------------------------------------------------------------------------------------------
   mod(sub)
-----------------------------------------------------------------------------------------------------*/

template<typename T>
class kernel_sub_mod_ntv;


template<typename T>
static void
sub_mod_test(cl::sycl::queue& queue, T modulus, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto mod = cl::sycl::malloc_shared<T>(1, queue);
    auto m = static_cast<T>(modulus);
    mod[0] = m;
    std::vector<T> h_C(num_workitems);
    std::vector<T> hd_C(num_workitems);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;



    for (int i = 0; i < num_workitems; i++) {
        h_A[i] = A[i] = dis(gen) % m;
        h_B[i] = B[i] = dis(gen) % m;
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Bin.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_A[i] = A[i] = Ain[i] % m;
            h_B[i] = B[i] = Bin[i] % m;
        }
    }

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }

        queue.submit([&](sycl::handler& h) {
            h.parallel_for<kernel_sub_mod_ntv<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
            {
                int i = it[0];

                d_C[i] = xehe::native::sub_mod<T>(A[i], B[i], mod[0]);

                });
            });

        queue.submit([&](sycl::handler& h) {
            // copy deviceArray to hostArray
            h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
            }).wait();

    if (!benchmark) {
        for (int i = 0; i < range_size; i++)
        {
            //h_C[i] = uint64_t(A[i] - B[i]) % m; // incorrect
            h_C[i] = xehe::native::sub_mod<T>(h_A[i], h_B[i], m);
        }
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << h_A[i] << " - " << h_B[i] << " mod " << m << " = " << hd_C[i] << "\n";
#endif
        bool success = true;
        for (int i = 0; i < range_size && success; i++)
        {
#ifdef WIN32
            if (h_C[i] != hd_C[i])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
            }
#else

            REQUIRE(h_C[i] == hd_C[i]);
#endif
        }

#ifdef WIN32
        if (success)
        {
            std::cout << "Success" << std::endl;
        }
#endif
    }
#ifndef WIN32
    else
    {
        BENCHMARK("simple sub mod") {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int i = it[0];
                    T tA, tB, tC = 0;
                    tA = A[i];
                    tB = B[i];
                    tA += tB;
                    for (int j = 0; j < inner_loops; ++j)
                    {
                        tC = xehe::native::sub_mod<T>(tA, tC, mod[0]);
                    }

                    d_C[i] = tC;
                    });
                }).wait();
        };
    }
#endif
}



static void Basic_uint64_mod_sub_test(void)
{
#ifdef WIN32
    std::cout << "Basic uint64 mod sub test" << std::endl;
#endif
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
#ifndef WIN32
    SECTION("mod 2")
#else
    std::cout << "mod 2" << std::endl;
#endif
    {
        uint64_t mod = 2;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1 };
        sub_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }
#ifndef WIN32
    SECTION("mod 10")
#else
    std::cout << "mod 10" << std::endl;
#endif
    {
        uint64_t mod = 10;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 3, 4, 6, 7 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 8, 3, 7, 7 };
        sub_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }

#ifndef WIN32
    SECTION("mod 2305843009211596801ULL")
#else
    std::cout << "mod 2305843009211596801ULL" << std::endl;
#endif
    {
        uint64_t mod = 2305843009211596801ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 1152921504605798400ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 1152921504605798401ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        sub_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }
}



static void Basic_uint32_mod_sub_test(void)
{
#ifdef WIN32
    std::cout << "Basic uint32 mod sub test" << std::endl;
#endif
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
#ifndef WIN32
    SECTION("mod 2")
#else
    std::cout << "mod 2" << std::endl;
#endif
    {
        uint32_t mod = 2;
        std::vector<uint32_t> Ain{ 0, 0, 1, 1 };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1 };
        sub_mod_test<uint32_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }

#ifndef WIN32
    SECTION("mod 10")
#else
    std::cout << "mod 10" << std::endl;
#endif
    {
        uint32_t mod = 10;
        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 3, 4, 6, 7 };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 8, 3, 7, 7 };
        sub_mod_test<uint32_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }

}

static void Basic_uint_mod_sub_test(void)
{
    Basic_uint32_mod_sub_test();
    Basic_uint64_mod_sub_test();
}


static void Basic_bench_mod_sub_test(void)
{
#ifdef WIN32
    std::cout << "Basic bench mod add test" << std::endl;
#endif
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100000;

#ifndef WIN32
    SECTION("mod 32bit")
#else
    std::cout << "mod 32bit" << std::endl;
#endif
    {
        uint32_t mod = 10;
        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 3, 4, 6, 7 };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 8, 3, 7, 7 };
        sub_mod_test<uint32_t>(queue, mod, Ain, Bin, n_workers, n_workers, true);
    }


#ifndef WIN32
    SECTION("mod 64bit")
#else
    std::cout << "mod 64bit" << std::endl;
#endif
    {
        uint64_t mod = 2305843009211596801ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 1152921504605798400ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 1152921504605798401ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        sub_mod_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers, true);
    }



}


#ifndef WIN32


TEST_CASE("Basic uint mod sub test", "[gpu][uintarithmod][uint][mod_sub]")
{
    Basic_uint32_mod_sub_test();
    Basic_uint64_mod_sub_test();
}


TEST_CASE("Basic bench mod sub test", "[gpu][uintarithmod][uint][mod_sub]")
{
    Basic_bench_mod_sub_test();
}

#endif



#endif
