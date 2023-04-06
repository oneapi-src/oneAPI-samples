/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "catch2/catch.hpp"
#include <array>
#include <iostream>
#include <iso646.h>




#include "../src/include/native/xe_uintarith_core.hpp"

#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

#ifdef BUILD_WITH_IGPU



template<typename T>
class kernel_mul_ntv_uint;

template<typename T>
class kernel_mul_ntv_setup;

template<typename T>
static void
mul_ntv_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {


    // GPU part setting
    auto d_A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(2 * num_workitems, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_C(2 * num_workitems);
    std::vector<T> hd_C(2 * num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

    for (int i = 0; i < num_workitems; i++) {
        d_A[i] = dis(gen);
        d_B[i] = dis(gen);
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
            h.parallel_for<kernel_mul_ntv_uint<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
            {
                int i = it[0];

                d_C[2 * i] = xehe::native::mul_uint<T>(d_A[i], d_B[i], d_C + 2 * i + 1);

                });
            });




        queue.submit([&](sycl::handler& h) {
            // copy deviceArray to hostArray
            h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
            }).wait();
    
    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " x " << d_B[i] << " = " << hd_C[i * 2 + 1] << hd_C[i * 2] << "\n";
        }
#endif

        for (int i = 0; i < range_size; i++)
        {
            h_C[2 * i] = xehe::native::mul_uint<T>(h_A[i], h_B[i], h_C.data() + 2 * i + 1);
        }
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << h_A[i] << " x " << h_B[i] << " = " << h_C[i * 2 + 1] << h_C[i * 2] << "\n";
        }
#endif

        bool success = true;
        for (int i = 0; i < range_size && success; i++)
        {
#ifdef WIN32
            if (h_C[2 * i] != hd_C[2 * i] || h_C[2 * i + 1] != hd_C[2 * i + 1])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[2 * i + 1] << h_C[2 * i] << " d " << hd_C[2 * i + 1] << hd_C[2 * i] << std::endl;
            }
#else
            REQUIRE((h_C[2 * i] == hd_C[2 * i] && h_C[2 * i + 1] == hd_C[2 * i + 1]));
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

        BENCHMARK("simple T bit mul")
        {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int i = it[0];
                    T A, B, C[2];
                    A = d_A[i];
                    B = d_B[i];
                    A += B;
                    for (int i = 0; i < inner_loops; ++i)
                    {
                        C[0] = xehe::native::mul_uint<T>(A, C[1], C + 1);
                    }
                    d_C[2 * i] = C[0];
                    d_C[2 * i + 1] = C[1];
                    });
                }).wait();
        };


    }

#endif

}



void Basic_static_native_uint_mul(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    size_t num_workitems = 1000;
#ifdef WIN32
    std::cout << "Test: native_uint_mul" << std::endl;
#endif

#ifndef WIN32
    SECTION("mul 32 bit")
#else
    std::cout << "mul 32 bit" << std::endl;
#endif
    {

        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        mul_ntv_test<uint32_t>(queue, Ain, Bin, num_workitems, num_workitems);
    }

#ifndef WIN32
    SECTION("mul 64 bit")
#else
    std::cout << "mul 64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000,  0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA, 0x1ffffffff };
        mul_ntv_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems);
    }


}





#ifndef WIN32

TEST_CASE("Basic static native uint multiply", "[gpu][uintarith][mul]") {
    Basic_static_native_uint_mul();
}


#endif


/*****************************************************************
 *
 *   PERF TESTS
 *
 *****************************************************************/

void Basic_static_bench_native_uint_mul(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    size_t num_workitems = 100000;

#ifdef WIN32
    std::cout << "Bench: native_uint_mul" << std::endl;
#endif
#ifndef WIN32
    SECTION("32 bit mul: 10M threads")
#else
    std::cout << "32 bit mul: " << num_workitems*2 << " threads" << std::endl;
#endif
    {
        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        mul_ntv_test<uint32_t>(queue, Ain, Bin, (num_workitems*2), (num_workitems*2), true);
    }



#ifndef WIN32
    SECTION("64 bit mul: 10M threads")
#else
    std::cout << "64 bit mul: " << num_workitems << " threads" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        mul_ntv_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
    }


}


#ifndef WIN32
TEST_CASE("Basic static bench native uint multiply", "[gpu][uintarith][mul]") {
    Basic_static_bench_native_uint_mul();
}

#endif




/* ---------------------------------------------------------
//                                   MODULAR ARITHMETICS
 ----------------------------------------------------------*/

 
template<typename T>
class kernel_mul_op_inv_mod;


template<typename T>
static void
mul_op_inv_mod_test(cl::sycl::queue& queue, T modulus, std::vector<T>& Ain, std::vector<T>& Bin,
    size_t num_workitems = 10, size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {
    auto m = modulus;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B_inv_mod = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto mod = cl::sycl::malloc_shared<T>(1, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_B_inv_mod(num_workitems);
    std::vector<T> h_C(num_workitems);
    std::vector<T> hd_C(num_workitems);


    for (int i = 0; i < num_workitems; i++) {
        h_A[i] = dis(gen) % m;
        h_B[i] = dis(gen) % m;
        A[i] = h_A[i];
        B[i] = h_B[i];
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Bin.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_A[i] = Ain[i] % m;
            h_B[i] = Bin[i] % m;
            A[i] = h_A[i];
            B[i] = h_B[i];
        }
    }
    for (int i = 0; i < num_workitems; i++) {

        // op2*2^BitCount/modulus
        T num[2]{ 0, B[i] };
        T quo2[2];
        xehe::native::div_uint2<T>(num, m, quo2);
        h_B_inv_mod[i] = quo2[0];
        B_inv_mod[i] = h_B_inv_mod[i];
    }
    // upload modulus
    mod[0] = m;

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << h_A[i] << " B[" << i << "] = " << h_B[i] << "\n";
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {
        // NOTE: I will keep printout for later reference - how to print out of kernel
        /*
        sycl::stream out(1024, 256, h);
        */
        h.parallel_for<kernel_mul_op_inv_mod<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];

            d_C[i] = xehe::native::mul_quotent_mod(A[i], B[i], mod[0], B_inv_mod[i]);
            });
        });



    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray
        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        }).wait();



        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
            {
                h_C[i] = xehe::native::mul_quotent_mod(h_A[i], h_B[i], m, h_B_inv_mod[i]);
            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << h_A[i] << " x " << h_B[i] << " mod " << m << " = " << hd_C[i] << " Host: " << h_C[i] << "\n";
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
            BENCHMARK("mul_op_inv_mod")
            {
                return queue.submit([&](sycl::handler& h) {
                    // NOTE: I will keep printout for later reference - how to print out of kernel
                    /*
                    sycl::stream out(1024, 256, h);
                    */
                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {

                        int i = it[0];
                        T tC = 0;
                        auto tA = A[i];
                        auto tB = B[i];
                        auto tB_inv_mod = B_inv_mod[i];
                        tA += tB;
                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tC = xehe::native::mul_quotent_mod(tA, tC, mod[0], tB_inv_mod);
                        }

                        d_C[i] = tC;


                        });
                    }).wait();
            };

        }
#endif
}


static
void Basic_mul_inv_mod_test(bool benchmark = false, int inner_loop = 100)
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
    int n_timed_workers = 1000000;

    if (!benchmark)
    {

#ifdef WIN32
        std::cout << "Test: mul_inv_mod" << std::endl;
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
            std::vector<uint64_t> Bin{ 0, 1, 0, 1, modulus - 1 };

            mul_op_inv_mod_test<uint64_t>(queue, modulus, Ain, Bin, n_workers, n_workers);
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
            std::vector<uint32_t> Bin{ 0, 1, 0, 1, 7, 7, 6 };

            mul_op_inv_mod_test<uint32_t>(queue, modulus, Ain, Bin, n_workers, n_workers);
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
            std::vector<uint64_t> Bin{ 0, 1, 0, 1, modulus - 1, (1152921504605798401), (1152921504605798400),
                                      (1152921504605798401), (2305843009211596800) };

            mul_op_inv_mod_test<uint64_t>(queue, modulus, Ain, Bin, n_workers, n_workers);
        };

    }

#ifdef WIN32
    if (benchmark)
    {
        std::cout << "Bench: mul_inv_mod" << std::endl;
    }
#endif
#ifndef WIN32
    SECTION("32 bit big")
#endif
    {
#ifdef WIN32
        std::cout << "32 bit big" << std::endl;
#endif
        uint32_t modulus = 0xFFFFFFF0;
        std::vector<uint32_t> Ain{ 0xFFFFFFF1, 0, 1, 1, 7, 6, 7 };
        std::vector<uint32_t> Bin{ 0xFFFFFFF1, 1, 0, 1, 7, 7, 6 };

        n_workers = (!benchmark) ? n_workers : n_timed_workers*2;
        mul_op_inv_mod_test<uint32_t>(queue, modulus, Ain, Bin, n_workers, n_workers, benchmark, inner_loop);

    };

#ifndef WIN32
    SECTION("64 bit #2")
#endif
    {
#ifdef WIN32
        std::cout << "64 bit #2"  << std::endl;
#endif
        uint64_t modulus = 2305843009213693951ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798401, 1152921504605798401,
                                  2305843009211596800 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798400, 1152921504605798401,
                                  2305843009211596800 };

        n_workers = (!benchmark) ? n_workers : n_timed_workers;
        mul_op_inv_mod_test<uint64_t>(queue, modulus, Ain, Bin, n_workers, n_workers, benchmark, inner_loop);
    }
}



#ifndef WIN32
TEST_CASE("Basic mul inv mod test", "[gpu][uintarithmod][uint64][mul_mod]")
{
    Basic_mul_inv_mod_test();
}

TEST_CASE("Basic bench mul inv mod test", "[gpu][uintarithmod][uint64][mul_mod]")
{
    Basic_mul_inv_mod_test(true);
}


#endif



template<typename T>
class kernel_burret_red1;


template<typename T>
static void
burret_red1_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Min,
    size_t num_workitems = 10, size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto M = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto M_inv = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);

    std::vector<T> h_M(num_workitems);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_M_inv(num_workitems);;
    std::vector<T> h_C(num_workitems);
    std::vector<T> hd_C(num_workitems);


    for (int i = 0; i < num_workitems; i++) {
        h_A[i] = dis(gen);
        // modulo
        h_M[i] = dis(gen) % ((sizeof(T) == 8) ? 0x7fffffffffffffffUL : 0x7fffffff);
        A[i] = h_A[i];
        M[i] = h_M[i];
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Min.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_A[i] = Ain[i];
            h_M[i] = Min[i];
            A[i] = h_A[i];
            M[i] = h_M[i];
        }
    }

    // inv modulo
    for (int i = 0; i < num_workitems; i++) {

        h_M_inv[i] = xehe::native::mod_inverse1(M[i]); // 2^64 / mod
        M_inv[i] = h_M_inv[i];
    }


    if (!benchmark) {


#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << h_A[i] << " M[" << i << "] = " << h_M[i] << "\n";
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {

        h.parallel_for<kernel_burret_red1<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];

            d_C[i] = xehe::native::barrett_reduce(A[i], M[i], M_inv[i]);
            });
        }).wait();



    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray 
        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
            {
                h_C[i] = h_A[i] % h_M[i];
            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << h_A[i] << " mod " << h_M[i] << " = " << hd_C[i] << " Host: " << h_C[i] << "\n";
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
            BENCHMARK("barret reduction 1")
            {

                return queue.submit([&](sycl::handler& h) {

                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {

                        int i = it[0];
                        T tC = 0;
                        auto tA = A[i];
                        auto tM = M[i];
                        auto tM_inv = M_inv[i];

                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tC += tA;
                            auto tA2 = tC >> 1;
                            tC = xehe::native::barrett_reduce(tA2, tM, tM_inv);
                        }

                        d_C[i] = tC;

                        });
                    }).wait();
            };

        }
#endif
}

static
void Basic_barret_red1_test(bool benchmark = false, int inner_loop = 100)
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
    int n_timed_workers = 1000000;

    if (!benchmark)
    {
#ifdef WIN32

        std::cout << "Test: barret_red1" << std::endl;

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

            burret_red1_test<uint64_t>(queue, Ain, Min, n_workers, n_workers);
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

            burret_red1_test<uint32_t>(queue, Ain, Min, n_workers, n_workers);
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

            burret_red1_test<uint64_t>(queue, Ain, Min, n_workers, n_workers);
        }

    }
#ifdef WIN32
    if (benchmark)
    {
        std::cout << "Bench: barret_red1" << std::endl;
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

        burret_red1_test<uint32_t>(queue, Ain, Min, n_workers, n_workers, benchmark, inner_loop);
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

        burret_red1_test<uint64_t>(queue, Ain, Min, n_workers, n_workers, benchmark, inner_loop);
    }
}


#ifndef WIN32
TEST_CASE("Basic barret red1 test", "[gpu][uintarithmod][uint64][barret_red1]")
{
    Basic_barret_red1_test();
}

TEST_CASE("Basic bench barret red1 test", "[gpu][uintarithmod][uint64][barret_red1]")
{
    Basic_barret_red1_test(true);
}


#endif



template<typename T>
class kernel_burret_red2;


template<typename T>
static void
burret_red2_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, std::vector<T>& Min,
    size_t num_workitems = 10, size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

    auto A = cl::sycl::malloc_shared<T>(2 * num_workitems, queue);
    auto M = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto M_inv = cl::sycl::malloc_shared<T>(2 * num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    std::vector<T> h_M(num_workitems);
    std::vector<T> h_A(2 * num_workitems);
    std::vector<T> h_M_inv(2 * num_workitems);;
    std::vector<T> h_C(num_workitems);
    std::vector<T> hd_C(num_workitems);


    for (int i = 0; i < num_workitems; i++) {
        auto a = dis(gen);
        auto b = dis(gen);
        T in[2];
        in[0] = xehe::native::mul_uint<T>(a, b, in + 1);
        h_A[2 * i] = in[0];
        h_A[2 * i + 1] = in[1];

        // modulo
        h_M[i] = dis(gen) % ((sizeof(T) == 8) ? 0x7fffffffffffffffUL : 0x7fffffff);
        A[2 * i] = h_A[2 * i];
        A[2 * i + 1] = h_A[2 * i + 1];

        // modulo
        M[i] = h_M[i];

    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Min.size());
        for (int i = 0; i < Ain.size(); i++) {
            T in[2];
            in[0] = xehe::native::mul_uint<T>(Ain[i], Bin[i], in + 1);
            h_A[2 * i] = in[0];
            h_A[2 * i + 1] = in[1];
            h_M[i] = Min[i];
            A[2 * i] = h_A[2 * i];
            A[2 * i + 1] = h_A[2 * i + 1];
            M[i] = h_M[i];

        }
    }

    // inv modulo
    for (int i = 0; i < num_workitems; i++) {

        // 2^128
        M_inv[2 * i] = xehe::native::mod_inverse2(M[i], &M_inv[2 * i + 1]);
    }


    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << h_A[2 * i + 1] << h_A[2 * i] << " M[" << i << "] = " << h_M[i] << "\n";
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {

        h.parallel_for<kernel_burret_red2<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];
            // reduction T2
            d_C[i] = xehe::native::barrett_reduce2(&A[2 * i], M[i], &M_inv[2 * i]);

            });
        });



    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray
        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
            {
                h_C[i] = xehe::native::mod_reduce_generic2<T>(&h_A[2 * i], h_M[i]);
            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
            {
                std::cout << h_A[i] << " mod " << h_M[i] << " = " << hd_C[i] << " Host: " << h_C[i] << "\n";
            }
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
            BENCHMARK("barret reduction 2")
            {

                return queue.submit([&](sycl::handler& h) {

                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {

                        int i = it[0];
                        T tC = 0, tA[2], tM_inv[2];
                        tA[0] = A[2 * i];
                        tA[1] = A[2 * i + 1];
                        auto tM = M[i];
                        tM_inv[0] = M_inv[2 * i];
                        tM_inv[1] = M_inv[2 * i + 1];

                        // reduction T2
                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tA[1] = tC;
                            tA[0] += tC;
                            tC = xehe::native::barrett_reduce2(tA, tM, tM_inv);
                        }


                        d_C[i] = tC;

                        });
                    }).wait();
            };

        }
#endif
}


static
void Basic_barret_red2_test(bool benchmark = false, int inner_loop = 100)
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
    int n_timed_workers = 1000000;

    if (!benchmark)
    {
#ifdef WIN32

        std::cout << "Test: barret_red2" << std::endl;

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
            std::vector<uint64_t> Bin{ 0, 0, 1, 1, modulus - 1 };
            std::vector<uint64_t> Min{ modulus, modulus + 1, modulus, modulus + 1, modulus };

            burret_red2_test<uint64_t>(queue, Ain, Bin, Min, n_workers, n_workers);
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
            std::vector<uint32_t> Bin{ 0, 0, 1, 1, 7, 6, 7 };
            std::vector<uint32_t> Min;
            for (int i = 0; i < Ain.size(); i++)
            {
                Min.push_back(modulus + i);
            }

            burret_red2_test<uint32_t>(queue, Ain, Bin, Min, n_workers, n_workers);
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
            std::vector<uint64_t> Bin{ 0, 0, 1, 1, modulus - 1, (1152921504605798400), (1152921504605798401),
                                      (1152921504605798401), (2305843009211596800) };
            std::vector<uint64_t> Min;
            for (int i = 0; i < Ain.size(); i++)
            {
                Min.push_back(modulus + i);
            }

            burret_red2_test<uint64_t>(queue, Ain, Bin, Min, n_workers, n_workers);
        }

    }
#ifdef WIN32
    if (benchmark)
    {
        std::cout << "Bench: barret_red2" << std::endl;
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
        std::vector<uint32_t> Bin{ 0xFFFFFFF1, 0, 1, 1, 7, 6, 7 };
        std::vector<uint32_t> Min;
        for (int i = 0; i < Ain.size(); i++)
        {
            Min.push_back(modulus + i);
        }

        burret_red2_test<uint32_t>(queue, Ain, Bin, Min, n_workers, n_workers, benchmark, inner_loop);
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
        std::vector<uint64_t> Bin{ 0, 0, 1, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798401, 1152921504605798401,
                                  2305843009211596800 };
        std::vector<uint64_t> Min;
        for (int i = 0; i < Ain.size(); i++)
        {
            Min.push_back(modulus + i);
        }

        burret_red2_test<uint64_t>(queue, Ain, Bin, Min, n_workers, n_workers, benchmark, inner_loop);
    }
}


#ifndef WIN32
TEST_CASE("Basic barret red2 test", "[gpu][uintarithmod][uint64][barret_red2]")
{
    Basic_barret_red2_test();
}

TEST_CASE("Basic bench barret red2 test", "[gpu][uintarithmod][uint64][barret_red2]")
{
    Basic_barret_red2_test(true);
}
#endif


template<typename T>
class kernel_mod_mul_inv2;


template<typename T>
static void
mod_mul_inv2_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, std::vector<T>& Min,
    size_t num_workitems = 10, size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto M = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto M_inv = cl::sycl::malloc_shared<T>(2 * num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    std::vector<T> h_M(num_workitems);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_M_inv(2 * num_workitems);;
    std::vector<T> h_C(num_workitems);
    std::vector<T> hd_C(num_workitems);


    for (int i = 0; i < num_workitems; i++) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
        // modulo
        h_M[i] = dis(gen) % ((sizeof(T) == 8) ? 0x7fffffffffffffffUL : 0x7fffffff);
        A[i] = h_A[i];
        B[i] = h_B[i];
        // modulo
        M[i] = h_M[i];
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Min.size());
        for (int i = 0; i < Ain.size(); i++) {
            h_A[i] = Ain[i];
            h_B[i] = Bin[i];
            h_M[i] = Min[i];
            A[i] = h_A[i];
            B[i] = h_B[i];
            M[i] = h_M[i];
        }
    }

    // inv modulo
    for (int i = 0; i < num_workitems; i++) {

        T quo3[3];
        T pow3[3]{ 0, 0, 1 }; // 2^128
        xehe::native::div_uint3<T>(pow3, h_M[i], quo3);
        h_M_inv[2 * i] = quo3[0];
        h_M_inv[2 * i + 1] = quo3[1];
        M_inv[2 * i] = h_M_inv[2 * i];
        M_inv[2 * i + 1] = h_M_inv[2 * i + 1];
    }


    if (!benchmark) {


#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << h_A[i] << " B[" << i << "] = " << h_B[i] << " M[" << i << "] = " << h_M[i] << "\n";
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {

        h.parallel_for<kernel_mod_mul_inv2<T>>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
        {
            int i = it[0];

            // mod(mul2)
            d_C[i] = xehe::native::mul_mod(A[i], B[i], M[i], &M_inv[2 * i]);

            });
        });



    queue.submit([&](sycl::handler& h) {
        // copy deviceArray to hostArray
        h.memcpy(hd_C.data(), d_C, hd_C.size() * sizeof(T));
        }).wait();

        if (!benchmark) {

            for (int i = 0; i < range_size; i++)
            {
                // prepare T2 value
                T in[2];
                in[0] = xehe::native::mul_uint<T>(h_A[i], h_B[i], in + 1);
                h_C[i] = xehe::native::mod_reduce_generic2<T>(in, h_M[i]);
            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << h_A[i] << " mod " << h_M[i] << " = " << hd_C[i] << " Host: " << h_C[i] << "\n";
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
            BENCHMARK("mod mul 2")
            {

                return queue.submit([&](sycl::handler& h) {

                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {

                        int i = it[0];
                        T tC = 0, tA, tB, tM_inv[2];
                        tA = A[i];
                        tB = A[i];
                        auto tM = M[i];
                        tM_inv[0] = M_inv[2 * i];
                        tM_inv[1] = M_inv[2 * i + 1];
                        tA += tB;
                        // reduction T2
                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tC = xehe::native::mul_mod(tA, tC, tM, tM_inv);
                        }


                        d_C[i] = tC;

                        });
                    }).wait();
            };

        }
#endif
}

static
void Basic_mod_mul2_test(bool benchmark = false, int inner_loop = 100)
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
    int n_timed_workers = 1000000;

    if (!benchmark)
    {
#ifdef WIN32

        std::cout << "Test: mod mul2" << std::endl;

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
            std::vector<uint64_t> Bin{ 0, 0, 1, 1, modulus - 1 };
            std::vector<uint64_t> Min{ modulus, modulus + 1, modulus, modulus + 1, modulus };

            mod_mul_inv2_test<uint64_t>(queue, Ain, Bin, Min, n_workers, n_workers);
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
            std::vector<uint32_t> Bin{ 0, 0, 1, 1, 7, 6, 7 };
            std::vector<uint32_t> Min;
            for (int i = 0; i < Ain.size(); i++)
            {
                Min.push_back(modulus + i);
            }

            mod_mul_inv2_test<uint32_t>(queue, Ain, Bin, Min, n_workers, n_workers);
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
            std::vector<uint64_t> Bin{ 0, 0, 1, 1, modulus - 1, (1152921504605798400), (1152921504605798401),
                                      (1152921504605798401), (2305843009211596800) };
            std::vector<uint64_t> Min;
            for (int i = 0; i < Ain.size(); i++)
            {
                Min.push_back(modulus + i);
            }

            mod_mul_inv2_test<uint64_t>(queue, Ain, Bin, Min, n_workers, n_workers);
        }

    }
#ifdef WIN32
    if (benchmark)
    {
        std::cout << "Bench: mod mul2" << std::endl;
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
        std::vector<uint32_t> Bin{ 0xFFFFFFF1, 0, 1, 1, 7, 6, 7 };
        std::vector<uint32_t> Min;
        for (int i = 0; i < Ain.size(); i++)
        {
            Min.push_back(modulus + i);
        }

        mod_mul_inv2_test<uint32_t>(queue, Ain, Bin, Min, n_workers, n_workers, benchmark, inner_loop);
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
        std::vector<uint64_t> Bin{ 0, 0, 1, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798401, 1152921504605798401,
                                  2305843009211596800 };
        std::vector<uint64_t> Min;
        for (int i = 0; i < Ain.size(); i++)
        {
            Min.push_back(modulus + i);
        }

        mod_mul_inv2_test<uint64_t>(queue, Ain, Bin, Min, n_workers, n_workers, benchmark, inner_loop);
    }
}


#ifndef WIN32
TEST_CASE("Basic mod mul2 test", "[gpu][uintarithmod][uint64][mod_mul2]")
{
    Basic_mod_mul2_test();
}

TEST_CASE("Basic bench mod mul2 test", "[gpu][uintarithmod][uint64][mod_mul2]")
{
    Basic_mod_mul2_test(true);
}
#endif


#endif //BUILD_WITH_IGPU
