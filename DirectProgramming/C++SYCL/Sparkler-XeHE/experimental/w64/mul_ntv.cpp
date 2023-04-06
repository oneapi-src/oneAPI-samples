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


// to remove exceptions
#ifdef XeHE_DEBUG
#undef XeHE_DEBUG
#endif
#include "../src/include/native/xe_uintarith.hpp"
#include "../src/include/native/xe_uintarith_core.hpp"
#include "../src/include/native/xe_uintarith_w64.hpp"

#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

#ifdef BUILD_WITH_IGPU

#include "dpcpp_utils.h"
#include "lib_utils.h"
#include <CL/sycl.hpp>


template<typename T>
class kernel_mul_ntv_uint;

template<typename T>
class kernel_mul_ntv_setup;

template<typename T>
static void
mul_ntv_test(cl::sycl::queue &queue, std::vector<T> &Ain, std::vector<T> &Bin, size_t num_workitems = 10,
         size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {


    // GPU part setting
    auto d_A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(2 * num_workitems, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_C(2 * num_workitems);
    
   

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

    queue.submit([&](sycl::handler &h) {
        h.parallel_for<kernel_mul_ntv_uint<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            d_C[2*i] = xehe::native::mul_uint<T>(d_A[i], d_B[i], d_C +2*i+1);

        });
    });

    queue.wait();

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " x " << d_B[i] << " = " << d_C[i*2 + 1] << d_C[i * 2] << "\n";
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
            if (h_C[2*i] != d_C[2*i] || h_C[2 * i + 1] != d_C[2 * i + 1])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[2 * i + 1] << h_C[2*i] << " d " << d_C[2 * i + 1]<< d_C[2*i] << std :: endl;
            }
#else
            REQUIRE((h_C[2*i] == d_C[2*i] && h_C[2 * i + 1] == d_C[2 * i + 1]));
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
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];
                    T A, B, C[2];
                    A = d_A[i];
                    B = d_B[i];
                    A += B;
                    for(int i = 0; i < inner_loops; ++i)
                    {
                      C[0] = xehe::native::mul_uint<T>(A, C[0], C + 1);
                    }
                    d_C[2*i] = C[0];
                    d_C[2*i+1] = C[1];
                    });
                }).wait();
        };


    }
    
#endif

}

template<typename T>
class kernel_mul_ntv32_uint;

template<typename T>
class kernel_mul_ntv32_setup;

template<typename T>
static void
mul_ntv32_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {


    // GPU part setting
    auto d_A = cl::sycl::malloc_shared<uint64_t>(num_workitems, queue);
    auto d_B = cl::sycl::malloc_shared<uint64_t>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<uint64_t>(2 * num_workitems, queue);
    std::vector<uint64_t> h_A(num_workitems);
    std::vector<uint64_t> h_B(num_workitems);
    std::vector<uint64_t> h_C(2 * num_workitems);



    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    for (int i = 0; i < num_workitems; i++) {

        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
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
            std::cout << "D: A[" << i << "] = " << d_A[i]
                << " B[" << i << "] = " << d_B[i] << "\n";
            std::cout << "H: A[" << i << "] = " << h_A[i] << " B[" << i << "] = " << h_B[i] << "\n";
        }
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_mul_ntv32_uint<uint64_t>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];



            xehe::w64_t A, B;
            xehe::w128_t C;
            //auto pA = (xehe::w64_t*)&d_A[i];
            //A.part[0] = pA->part[0];
            //A.part[1] = pA->part[1];
            XEHE_READ_W64(d_A, i, 0, A);
            //auto pB = (xehe::w64_t*)&d_B[i];
            //B.part[0] = pB->part[0];
            //B.part[1] = pB->part[1];
            XEHE_READ_W64(d_B, i, 0, B);


            C = xehe::native::mul_uint(A, B);

            XEHE_WRITE_W128(d_C, (2 * i), 0, C);
            });
        });

    queue.wait();

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " x " << d_B[i]
                << " = " << d_C[i*2+1] << d_C[i*2] << "\n";
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
            if (h_C[2 * i] != d_C[i * 2] || h_C[2 * i + 1] != d_C[i * 2 + 1])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[2 * i + 1] << h_C[2 * i]
                    << " d " << d_C[i * 2 + 1] << d_C[i * 2] << std::endl;
            }
#else
            REQUIRE((h_C[2 * i] == d_C[i * 2] && h_C[2 * i + 1] == d_C[i * 2 + 1]));
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
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];


                    xehe::w64_t A, B;
                    xehe::w128_t C;
                    C.part[0].part[0] = C.part[0].part[1] = C.part[1].part[0] = C.part[1].part[1] = 1;
                    XEHE_READ_W64(d_A, i, 0, A);
                    XEHE_READ_W64(d_B, i, 0, B);

                    A.part[0] += B.part[0];
                    A.part[1] += B.part[1];

                    for(int i = 0; i < inner_loops; ++i)
                    {
                        auto rC = ((i & 1) == 0) ? C.part[0] : C.part[1];
                        C = xehe::native::mul_uint(A, rC);
                    }

                    XEHE_WRITE_W128(d_C, (2 * i), 0, C);
                    


                    });
                }).wait();
        };



    }

#endif

}


void Basic_static_native_uint_mul(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

#ifndef WIN32
    SECTION("mul 32 bit")
#else
    std::cout << "mul 32 bit" << std::endl;
#endif
    {

        std::vector<uint32_t> Ain{ 0, 0, 1, 1, 0xFABA };
        std::vector<uint32_t> Bin{ 0, 1, 0, 1, 0xABA00000 };
        mul_ntv_test<uint32_t>(queue, Ain, Bin);
    }

#ifndef WIN32
    SECTION("mul 64 bit")
#else
    std::cout << "mul 64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        mul_ntv_test<uint64_t>(queue, Ain, Bin);
    }


}



void Basic_static_native32_uint_mul(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();


#ifndef WIN32
    SECTION("mul w64 bit")
#else
    std::cout << "mul w64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        mul_ntv32_test<uint64_t>(queue, Ain, Bin);
    }


}



#ifndef WIN32

TEST_CASE("Basic static native uint multiply", "[gpu][uintarith]") {
    Basic_static_native_uint_mul();
}

TEST_CASE("Basic static native32 uint multiply", "[gpu][uintarith]") {
    Basic_static_native32_uint_mul();
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

#ifndef WIN32
    SECTION("w64 mul: 10M threads")
#else
    std::cout << "w64 mul: " << num_workitems << " threads" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        mul_ntv32_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
    }

}


#ifndef WIN32
TEST_CASE("Basic static bench native uint multiply", "[gpu][uintarith]") {
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
        auto m  = modulus;

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;

        auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto B_inv_mod = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto mod = cl::sycl::malloc_shared<T>(1, queue);
        std::vector<T> h_C(num_workitems);


        for (int i = 0; i < num_workitems; i++) {
            A[i] = dis(gen) % m;
            B[i] = dis(gen) % m;
        }
        if (Ain.size() > 0 and Ain.size() < num_workitems) {
            assert(Ain.size() == Bin.size());
            for (int i = 0; i < Ain.size(); i++) {
                A[i] = Ain[i] % m;
                B[i] = Bin[i] % m;
            }
        }
        for (int i = 0; i < num_workitems; i++) {

            // op2*2^BitCount/modulus
            T num[2]{ 0, B[i] };
            T quo2[2];
            xehe::native::div_uint2<T>(num, m, quo2);
            B_inv_mod[i] = quo2[0];
        }
        // upload modulus
        mod[0] = m;

        if (!benchmark) {
#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
            std::cout << "\n";
#endif
        }

        queue.submit([&](sycl::handler& h) {
            // NOTE: I will keep printout for later reference - how to print out of kernel
            /*
            sycl::stream out(1024, 256, h);
            */
            h.parallel_for<kernel_mul_op_inv_mod<T>>(sycl::range<1>(range_size), [=](auto it) {
                int i = it[0];

                d_C[i] = xehe::native::mul_quotent_mod(A[i], B[i], mod[0], B_inv_mod[i]);
                });
            }).wait();

            if (!benchmark) {
                for (int i = 0; i < range_size; i++)
                {
                    h_C[i] = xehe::native::mul_quotent_mod(A[i], B[i], m, B_inv_mod[i]);
                }

#ifdef VERBOSE_TEST
                for (int i = 0; i < num_workitems; i++)
                    std::cout << A[i] << " x " << B[i] << " mod " << m << " = " << d_C[i] << " Host: " << h_C[i] << "\n";
#endif
                bool success = true;
                for (int i = 0; i < range_size && success; i++)
                {
#ifdef WIN32
                    if (h_C[i] != d_C[i])
                    {
                        success = false;
                        std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
                    }
#else
                    REQUIRE(h_C[i] == d_C[i]);
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
                        h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {

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

template<typename T>
class kernel_mulw64_op_inv_mod;


template<typename T>
static void
mulw64_op_inv_mod_test(cl::sycl::queue& queue, T modulus, std::vector<T>& Ain, std::vector<T>& Bin,
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
    std::vector<T> h_C(num_workitems);


    for (int i = 0; i < num_workitems; i++) {
        A[i] = dis(gen) % m;
        B[i] = dis(gen) % m;
    }
    if (Ain.size() > 0 and Ain.size() < num_workitems) {
        assert(Ain.size() == Bin.size());
        for (int i = 0; i < Ain.size(); i++) {
            A[i] = Ain[i] % m;
            B[i] = Bin[i] % m;
        }
    }
    for (int i = 0; i < num_workitems; i++) {

        // op2*2^BitCount/modulus
        T num[2]{ 0, B[i] };
        T quo2[2];
        xehe::native::div_uint2<T>(num, m, quo2);
        B_inv_mod[i] = quo2[0];
    }
    // upload modulus
    mod[0] = m;

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }

    queue.submit([&](sycl::handler& h) {
        // NOTE: I will keep printout for later reference - how to print out of kernel
        /*
        sycl::stream out(1024, 256, h);
        */
        h.parallel_for<kernel_mulw64_op_inv_mod<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];
            xehe::w64_t tA, tB, tB_inv_mod;
            xehe::w64_t tC, tmod;
            tmod.part[0] = ((xehe::w64_t*)mod)->part[0];
            tmod.part[1] = ((xehe::w64_t*)mod)->part[1];
            XEHE_READ_W64(A, i, 0, tA);
            XEHE_READ_W64(B, i, 0, tB);
            XEHE_READ_W64(B_inv_mod, i, 0, tB_inv_mod);

            tC = xehe::native::mul_quotent_mod<xehe::w64_t>(tA, tB, tmod, tB_inv_mod);


            XEHE_WRITE_W64(d_C, i, 0, tC);

            });
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
            {
                h_C[i] = xehe::native::mul_quotent_mod(A[i], B[i], m, B_inv_mod[i]);
            }

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " x " << B[i] << " mod " << m << " = " << d_C[i] << " Host: " << h_C[i] << "\n";
#endif
            bool success = true;
            for (int i = 0; i < range_size && success; i++)
            {
#ifdef WIN32
                if (h_C[i] != d_C[i])
                {
                    success = false;
                    std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
                }
#else
                REQUIRE(h_C[i] == d_C[i]);
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
                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                        int i = it[0];
                        xehe::w64_t tA, tB, tB_inv_mod;
                        xehe::w64_t tC, tmod;
                        tmod.part[0] = ((xehe::w64_t*)mod)->part[0];
                        tmod.part[1] = ((xehe::w64_t*)mod)->part[1];
                        XEHE_READ_W64(A, i, 0, tA);
                        XEHE_READ_W64(B, i, 0, tB);
                        XEHE_READ_W64(B_inv_mod, i, 0, tB_inv_mod);
                        xehe::native::add_uint<xehe::w64_t>(tA, tB, &tA);

                        for (int j = 0; j < inner_loops; ++j)
                        {
                            tC = xehe::native::mul_quotent_mod<xehe::w64_t>(tA, tC, tmod, tB_inv_mod);
                        }

                        XEHE_WRITE_W64(d_C, i, 0, tC);
                        });
                    }).wait();
            };

        }
#endif
}

static
void Basic_mulw64_inv_mod_test(bool benchmark = false, int inner_loop = 100)
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    int n_workers = 100;
    int n_timed_workers = 1000000;

    if (!benchmark)
    {
#ifndef WIN32
        SECTION("w64 binary")
#endif
        {
#ifdef WIN32
            std::cout << "w64 binary" << std::endl;
#endif
            uint64_t modulus = 2;
            std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1 };
            std::vector<uint64_t> Bin{ 0, 1, 0, 1, modulus - 1 };

            mulw64_op_inv_mod_test<uint64_t>(queue, modulus, Ain, Bin, n_workers, n_workers);
};
#ifndef WIN32
        SECTION("w64 #1")
#endif
        {
#ifdef WIN32
            std::cout << "w64 #1" << std::endl;
#endif
            uint64_t modulus = 2305843009211596801ULL;
            std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1, (1152921504605798400), (1152921504605798401),
                                      (1152921504605798401), (2305843009211596800) };
            std::vector<uint64_t> Bin{ 0, 1, 0, 1, modulus - 1, (1152921504605798401), (1152921504605798400),
                                      (1152921504605798401), (2305843009211596800) };

            mulw64_op_inv_mod_test<uint64_t>(queue, modulus, Ain, Bin, n_workers, n_workers);
        };

    }

#ifndef WIN32
    SECTION("w64 #2")
#endif
    {
#ifdef WIN32
        std::cout << "w64 #2" << std::endl;
#endif
        uint64_t modulus = 2305843009213693951ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798401, 1152921504605798401,
                                  2305843009211596800 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, modulus - 1, 0xFFFFFFFFFFFFFFF1, 1152921504605798400, 1152921504605798401,
                                  2305843009211596800 };

        n_workers = (!benchmark) ? n_workers : n_timed_workers;
        mulw64_op_inv_mod_test<uint64_t>(queue, modulus, Ain, Bin, n_workers, n_workers, benchmark, inner_loop);
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

TEST_CASE("Basic mulw64 inv mod test", "[gpu][uintarithmod][w64][mul_mod]")
{
    Basic_mulw64_inv_mod_test();
}

TEST_CASE("Basic bench mulw64 inv mod test", "[gpu][uintarithmod][w64][mul_mod]")
{
    Basic_mulw64_inv_mod_test(true);
}

#endif




#if 0 //def BUILD_WITH_SEAL


template<typename T>
class kernel_mul_op_mod_quot;


template<typename T>
static void mul_op_mod_quot_test(cl::sycl::queue& queue, uint64_t modulus, T A0, T B0, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false) {

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C2 = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto h_C = cl::sycl::malloc_shared<T>(num_workitems, queue);

    A[0] = A0;

    for (int i = 1; i < num_workitems; i++) {
        A[i] = dis(gen) % m;
    }
    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << "\n";
        std::cout << "\n";
#endif
    }

    auto d_m = cl::sycl::malloc_device<T>(1, queue);
    auto d_mul_mod_operand = cl::sycl::malloc_shared<MUMO<T>>(1, queue);
    queue.memcpy(d_m, &m, sizeof(T) * 1);
    queue.memcpy(d_mul_mod_operand, &y, sizeof(MUMO<T>) * 1);

    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_mul_mod_operand<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            d_C[i] = xehe::util::multiply_uint_mod<T>(A[i], *d_mul_mod_operand, *d_m);
            });
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
                h_C[i] = xehe::util::multiply_uint_mod<T>(A[i], y, m);

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " mod " << m << " = " << d_C[i] << " Host: " << h_C[i] << "\n";
#endif
            for (int i = 0; i < range_size; i++)
                REQUIRE(h_C[i] == d_C[i]);
        }
        else {
            BENCHMARK("basic mul_mod operand")
            {
                return queue.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                        int i = it[0];

                        d_C[i] = xehe::util::multiply_uint_mod<T>(A[i], *d_mul_mod_operand, *d_m);
                        });
                    }).wait();
            };

            BENCHMARK("simple mul_mod operand setup")
            {
                return queue.submit([&](sycl::handler& h) {
                    h.parallel_for<kernel_mul_mod_operand_setup<T>>(sycl::range<1>(range_size), [=](auto it) {
                        int i = it[0];
                        xehe::w64_t Ai;
                        Ai.value64b = A[i];
                        xehe::w64_t operand;
                        operand.value64b = d_mul_mod_operand->operand;
                        xehe::w64_t dm;
                        dm.value64b = *d_m;
                        d_C[i] = Ai.values32b[0] + operand.values32b[0] + dm.values32b[0];

                        });
                    }).wait();
            };
        }


        queue.submit([&](sycl::handler& h) {
            h.parallel_for<kernel_mul_mod_lazy<T>>(sycl::range<1>(range_size), [=](auto it) {
                int i = it[0];

                d_C[i] = xehe::util::multiply_uint_mod_lazy<T>(A[i], *d_mul_mod_operand, *d_m);
                });
            }).wait();

            if (!benchmark) {
                for (int i = 0; i < range_size; i++)
                    h_C[i] = xehe::util::multiply_uint_mod_lazy<T>(A[i], y, m);

#ifdef VERBOSE_TEST
                for (int i = 0; i < num_workitems; i++)
                    std::cout << A[i] << " mod " << m << " = " << d_C[i] << " Host: " << h_C[i] << "\n";
#endif
                for (int i = 0; i < range_size; i++)
                    REQUIRE(h_C[i] == d_C[i]);
            }
            else {
                BENCHMARK("simple mul_mod_lazy")
                {
                    return queue.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                            int i = it[0];

                            d_C[i] = xehe::util::multiply_uint_mod_lazy<T>(A[i], *d_mul_mod_operand, *d_m);
                            });
                        }).wait();
                };

                BENCHMARK("simple mul_mod_lazy setup")
                {
                    return queue.submit([&](sycl::handler& h) {
                        h.parallel_for<kernel_mul_mod_lazy_setup<T>>(sycl::range<1>(range_size), [=](auto it) {
                            int i = it[0];

                            xehe::w64_t Ai;
                            Ai.value64b = A[i];
                            xehe::w64_t operand;
                            operand.value64b = d_mul_mod_operand->operand;
                            xehe::w64_t dm;
                            dm.value64b = *d_m;
                            d_C[i] = Ai.values32b[0] + operand.values32b[0] + dm.values32b[0];

                            });
                        }).wait();
                };
            }

}



template<typename T>
class kernel_mul_mod32;

template<typename T>
static void mul_mod32_test(cl::sycl::queue &queue, size_t num_workitems = 10) {

    T m;
    m = 0xFFFFFFFF;

    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto h_C = cl::sycl::malloc_shared<T>(num_workitems, queue);

    A[0] = 0xFFFFFFF1;
    B[0] = 0xFFFFFFF1;

    for (int i = 1; i < num_workitems; i++) {
        A[i] = rand() % m;
        B[i] = rand() % m;
    }

#ifdef VERBOSE_TEST
    for (int i = 0; i < num_workitems; i++)
        std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
    std::cout << "\n";
#endif

    queue.submit([&](sycl::handler &h) {
        h.parallel_for<kernel_mul_mod32<T>>(sycl::range<1>(num_workitems), [=](auto it) {
            int i = it[0];
            d_C[i] = A[i] * B[i] % m;
        });
    }).wait();

    for (int i = 0; i < num_workitems; i++)
        h_C[i] = (A[i] * B[i]) % m;

#ifdef VERBOSE_TEST
    for (int i = 0; i < num_workitems; i++)
        std::cout << A[i] << " x " << B[i] << " mod " << m << " = " << d_C[i] << " Host: " << h_C[i] << " H2: "
                  << (A[i] * B[i]) % m << "\n";
#endif
    for (int i = 0; i < num_workitems; i++)
        REQUIRE(h_C[i] == d_C[i]);
}

TEST_CASE("Basic 32b mul mod test", "[gpu][uint32][mul]")
{
    // NOTE: this test is test the compiler capability to automatically expand the width, when multiplying 32b x 32b
    // As of now, it neither works on CPU nor on GPU -- room for compiler improvement!
    //
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    mul_mod32_test<uint32_t>(queue);
}


template<typename T>
class kernel_div2_mod;

template<typename T>
class kernel_div2_mod_setup;

template<typename T>
static void div2_mod_test(cl::sycl::queue &queue, T modulus, std::vector<T> Ain, size_t num_workitems = 10,
                          size_t range_size = 10, bool benchmark = false) {
    if (sizeof(T) == 8) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;

        auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto h_C = cl::sycl::malloc_shared<T>(num_workitems, queue);


        for (int i = 0; i < num_workitems; i++) {
            A[i] = dis(gen) % modulus;
        }
        if (Ain.size() > 0 and Ain.size() < num_workitems) {
            for (int i = 0; i < Ain.size(); i++) {
                A[i] = Ain[i];
            }
        }

        if (!benchmark) {
#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << "A[" << i << "] = " << A[i] << "\n";
            std::cout << "\n";
#endif
        }

        auto d_m = cl::sycl::malloc_device<T>(1, queue);
        queue.memcpy(d_m, &modulus, sizeof(T) * 1);

        queue.submit([&](sycl::handler &h) {
            h.parallel_for<kernel_div2_mod<T>>(sycl::range<1>(range_size), [=](auto it) {
                int i = it[0];

                d_C[i] = xehe::util::div2_uint_mod<T>(A[i], *d_m);
            });
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
                h_C[i] = xehe::util::div2_uint_mod<T>(A[i], modulus);

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " div 2 mod " << modulus << " = " << d_C[i] << " Host: " << h_C[i] << "\n";
#endif
            for (int i = 0; i < range_size; i++)
                REQUIRE(h_C[i] == d_C[i]);
        } else {

            BENCHMARK("simple div2 mod")
                        {
                            return queue.submit([&](sycl::handler &h) {
                                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                                    int i = it[0];

                                    d_C[i] = xehe::util::div2_uint_mod<T>(A[i], *d_m);
                                });
                            }).wait();
                        };

            BENCHMARK("simple div2 mod setup")
                        {
                            return queue.submit([&](sycl::handler &h) {
                                h.parallel_for<kernel_div2_mod_setup<T>>(sycl::range<1>(range_size), [=](auto it) {
                                    int i = it[0];

                                    xehe::w64_t Ai;
                                    Ai.value64b = A[i];
                                    xehe::w64_t dm;
                                    dm.value64b = *d_m;
                                    d_C[i] = Ai.values32b[0] + dm.values32b[0];
                                });
                            }).wait();
                        };

        }

    } else {
        FAIL();
    }
}

template<typename T>
class kernel_mul_add_mod;

template<typename T>
class kernel_mul_add_mod_setup;

template<typename T>
static void mul_add_mod_test(cl::sycl::queue &queue, uint64_t modulus, std::vector<T> &Ain, std::vector<T> &Bin,
                             std::vector<T> &Cin, size_t num_workitems = 10, size_t range_size = 10,
                             bool benchmark = false) {
    if (sizeof(T) == 8) {
        seal::Modulus seal_mod(modulus);
        T m = seal_mod.value();
        const T *const_ratio = nullptr;
        const_ratio = seal_mod.const_ratio().data();

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;

        auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto C = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto d_R = cl::sycl::malloc_shared<T>(num_workitems, queue);
        auto h_R = cl::sycl::malloc_shared<T>(num_workitems, queue);

        for (int i = 0; i < num_workitems; i++) {
            A[i] = dis(gen) % m;
            B[i] = dis(gen) % m;
            C[i] = dis(gen) % m;
        }
        if (Ain.size() > 0 && Ain.size() <= num_workitems) {
            for (int i = 0; i < Ain.size(); i++) {
                A[i] = Ain[i];
                B[i] = Bin[i];
                C[i] = Cin[i];
            }
        }
        if (!benchmark) {
#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " " << B[i] << " " << C[i] << "\n";
#endif
        }

        auto d_m = cl::sycl::malloc_device<T>(1, queue);
        auto d_const_ratio = cl::sycl::malloc_device<T>(3, queue);
        queue.memcpy(d_m, &m, sizeof(T) * 1);
        queue.memcpy(d_const_ratio, const_ratio, sizeof(T) * 3);

        queue.submit([&](sycl::handler &h) {
            h.parallel_for<kernel_mul_add_mod<T>>(sycl::range<1>(range_size), [=](auto it) {
                int i = it[0];
                d_R[i] = xehe::util::multiply_add_uint_mod(A[i], B[i], C[i], *d_m, d_const_ratio);
            });
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
                h_R[i] = xehe::util::multiply_add_uint_mod<T>(A[i], B[i], C[i], m, const_ratio);

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " * " << B[i] << " + " << C[i] << " mod " << m << " = " << d_R[i] << " Host: "
                          << h_R[i] << "\n";
#endif
            for (int i = 0; i < range_size; i++)
                REQUIRE(h_R[i] == d_R[i]);

        } else {
            BENCHMARK("simple mul_add_mod") {
                                                queue.submit([&](sycl::handler &h) {
                                                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                                                        int i = it[0];
                                                        d_R[i] = xehe::util::multiply_add_uint_mod(A[i], B[i], C[i],
                                                                                                   *d_m, d_const_ratio);
                                                    });
                                                }).wait();
                                            };

            BENCHMARK("simple mul_add_mod setup")
                        {
                            return queue.submit([&](sycl::handler &h) {
                                h.parallel_for<kernel_mul_add_mod_setup<T>>(sycl::range<1>(range_size), [=](auto it) {
                                    int i = it[0];

                                    xehe::w64_t Ai;
                                    Ai.value64b = A[i];
                                    xehe::w64_t Bi;
                                    Bi.value64b = B[i];
                                    xehe::w64_t Ci;
                                    Ci.value64b = C[i];
                                    xehe::w64_t dc;
                                    dc.value64b = d_const_ratio[0];
                                    xehe::w64_t dm;
                                    dm.value64b = *d_m;
                                    d_R[i] = Ai.values32b[0] + Bi.values32b[0] + Ci.values32b[0] + dc.values32b[0] +
                                             dm.values32b[0];

                                });
                            }).wait();
                        };
        }

    } else {
        FAIL();
    }
}

template<typename ValueType>
class MulAddModKernel {
public:

    MulAddModKernel(ValueType *R, const ValueType *A, const ValueType *B, const ValueType *C, const ValueType *m,
                    const ValueType *const_ratio) {
        d_A = A;
        d_B = B;
        d_C = C;
        d_R = R;
        d_m = m; // 1
        d_const_ratio = const_ratio; // 3
    }

    void operator()(cl::sycl::id<1> i) const {
        d_R[i] = xehe::util::multiply_add_uint_mod(d_A[i], d_B[i], d_C[i], *d_m, d_const_ratio);

    }

protected:
    const ValueType *d_A;
    const ValueType *d_B;
    const ValueType *d_C;
    ValueType *d_R;
    const ValueType *d_m;
    const ValueType *d_const_ratio;
};

template<typename ValueType>
cl::sycl::event
mul_add_mod_gpu(cl::sycl::queue &q, size_t range_size, ValueType *R, const ValueType *A, const ValueType *B,
                const ValueType *C, const ValueType *m, const ValueType *const_ratio) {

    cl::sycl::event event = q.submit([&](cl::sycl::handler &h) {
        h.parallel_for(range_size, MulAddModKernel<ValueType>(R, A, B, C, m, const_ratio));
    });
    event.wait();
    return event;
}

template<typename T>
class kernel_mul_add_mod_lambda;

template<typename T>
static void
mul_add_mod_explicit_mem_test(cl::sycl::queue &queue, uint64_t modulus, std::vector<T> &Ain, std::vector<T> &Bin,
                              std::vector<T> &Cin, size_t num_workitems = 10, size_t range_size = 10,
                              bool benchmark = false) {
    if (sizeof(T) == 8) {
        seal::Modulus seal_mod(modulus);
        T m = seal_mod.value();
        const T *const_ratio = nullptr;
        const_ratio = seal_mod.const_ratio().data();

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;

        auto d_A = cl::sycl::malloc_device<T>(num_workitems, queue);
        auto d_B = cl::sycl::malloc_device<T>(num_workitems, queue);
        auto d_C = cl::sycl::malloc_device<T>(num_workitems, queue);
        auto d_R = cl::sycl::malloc_device<T>(num_workitems, queue);

        auto h_R = cl::sycl::malloc_shared<T>(num_workitems, queue);

        std::vector<T> A(num_workitems);
        std::vector<T> B(num_workitems);
        std::vector<T> C(num_workitems);
        std::vector<T> R(num_workitems);

        for (int i = 0; i < num_workitems; i++) {
            A[i] = dis(gen) % m;
            B[i] = dis(gen) % m;
            C[i] = dis(gen) % m;
        }
        if (Ain.size() > 0 && Ain.size() <= num_workitems) {
            for (int i = 0; i < Ain.size(); i++) {
                A[i] = Ain[i];
                B[i] = Bin[i];
                C[i] = Cin[i];
            }
        }
        if (!benchmark) {
#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " " << B[i] << " " << C[i] << "\n";
#endif
        }

        queue.submit([&](sycl::handler &h) {
            h.memcpy(d_A, &A[0], num_workitems * sizeof(T));
        }).wait();

        queue.submit([&](sycl::handler &h) {
            h.memcpy(d_B, &B[0], num_workitems * sizeof(T));
        }).wait();

        queue.submit([&](sycl::handler &h) {
            h.memcpy(d_C, &C[0], num_workitems * sizeof(T));
        }).wait();

        auto d_m = cl::sycl::malloc_device<T>(1, queue);
        auto d_const_ratio = cl::sycl::malloc_device<T>(3, queue);
        queue.memcpy(d_m, &m, sizeof(T) * 1);
        queue.memcpy(d_const_ratio, const_ratio, sizeof(T) * 3);

        bool lambda_flag = false;
        if (lambda_flag) {
            queue.submit([&](sycl::handler &h) {
                h.parallel_for<kernel_mul_add_mod_lambda<T>>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];
                    d_R[i] = xehe::util::multiply_add_uint_mod(d_A[i], d_B[i], d_C[i], *d_m, d_const_ratio);
                });
            }).wait();
        } else {
#ifdef DEBUG_BENCHMARKING
            double min_time_ns = DBL_MAX;
            for (int i = 0; i < 10; i++) {
                auto Q_event = mul_add_mod_gpu<T>(queue, range_size, d_R, d_A, d_B, d_C, d_m, d_const_ratio);
                auto end = Q_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
                auto start = Q_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
                double exec_time_ns = end - start;
                std::cout << "Execution time (iteration " << i << ") [sec]: "
                          << (double) exec_time_ns * 1.0E-9 << "\n";
                min_time_ns = std::min(min_time_ns, exec_time_ns);
                std::cout << "min time: " << (double) min_time_ns * 1.0E-9 << std::endl;
            }
#else
            mul_add_mod_gpu<T>(queue, range_size, d_R, d_A, d_B, d_C, d_m, d_const_ratio);
#endif
        }


        queue.submit([&](sycl::handler &h) {
            h.memcpy(&R[0], d_R, num_workitems * sizeof(T));
        }).wait();

        if (!benchmark) {
            for (int i = 0; i < range_size; i++)
                h_R[i] = xehe::util::multiply_add_uint_mod<T>(A[i], B[i], C[i], m, const_ratio);

#ifdef VERBOSE_TEST
            for (int i = 0; i < num_workitems; i++)
                std::cout << A[i] << " * " << B[i] << " + " << C[i] << " mod " << m << " = " << d_R[i] << " Host: "
                          << h_R[i] << "\n";
#endif
            for (int i = 0; i < range_size; i++)
                REQUIRE(h_R[i] == R[i]);

        } else {
            BENCHMARK("simple mul_add_mod") {
                                                queue.submit([&](sycl::handler &h) {
                                                    h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                                                        int i = it[0];
                                                        d_R[i] = xehe::util::multiply_add_uint_mod(d_A[i], d_B[i],
                                                                                                   d_C[i],
                                                                                                   *d_m, d_const_ratio);
                                                    });
                                                }).wait();
                                            };
                // NOTE: this code was developed to verify that we can still use benchmarking utils from catch2
                // if the input size is large, all overheads become negligible and BENCHMARK can be used
#ifdef DEBUG_BENCHMARKING
                {
                    std::vector<int> v;
                    auto num_samples = 20;
                    for (int i = 0; i < num_samples; i++) {
                        auto start = std::chrono::high_resolution_clock::now();

                        queue.submit([&](sycl::handler &h) {
                            h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                                int i = it[0];
                                d_R[i] = xehe::util::multiply_add_uint_mod(d_A[i], d_B[i],
                                                                           d_C[i],
                                                                           *d_m, d_const_ratio);
                            });
                        }).wait();

                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                        v.push_back(duration.count());

                        //cout << duration.count()<< endl;

                    }
                    std::cout << std::endl;

                    for (int i = 0; i < num_samples; i++)
                        std::cout << v[i] << std::endl;

                    auto offset = 5;
                    std::vector<int> mean_v(v.begin() + offset, v.end());
                    double sum = std::accumulate(mean_v.begin(), mean_v.end(), 0.0);
                    double mean = sum / (mean_v.size());

                    double sq_sum = std::inner_product(mean_v.begin(), mean_v.end(), mean_v.begin(), 0.0);
                    double stdev = std::sqrt(sq_sum / mean_v.size() - mean * mean);
                    std::cout << "mean: " << (int) mean << std::endl;
                    std::cout << "stdev: " << stdev << std::endl;
                }
#endif


            BENCHMARK("simple mul_add_mod kernel")
                        {
                            return mul_add_mod_gpu<T>(queue, range_size, d_R, d_A, d_B, d_C, d_m, d_const_ratio);

                        };
                // NOTE: this code was developed to verify that we can still use benchmarking utils from catch2
                // if the input size is large, all overheads become negligible and BENCHMARK can be used
#ifdef DEBUG_BENCHMARKING
                std::vector<int> v;
                auto num_samples = 20;
                for (int i = 0; i < num_samples; i++) {
                    auto start = std::chrono::high_resolution_clock::now();

                    mul_add_mod_gpu<T>(queue, range_size, d_R, d_A, d_B, d_C, d_m, d_const_ratio);

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    v.push_back(duration.count());

                    //cout << duration.count()<< endl;

                }
                std::cout << std::endl;

                for (int i = 0; i < num_samples; i++)
                    std::cout << v[i] << std::endl;

                auto offset = 5;
                std::vector<int> mean_v(v.begin() + offset, v.end());
                double sum = std::accumulate(mean_v.begin(), mean_v.end(), 0.0);
                double mean = sum / (mean_v.size());

                double sq_sum = std::inner_product(mean_v.begin(), mean_v.end(), mean_v.begin(), 0.0);
                double stdev = std::sqrt(sq_sum / mean_v.size() - mean * mean);
                std::cout << "mean: " << (int) mean << std::endl;
                std::cout << "stdev: " << stdev << std::endl;
#endif

            BENCHMARK("simple mul_add_mod setup")
                        {
                            return queue.submit([&](sycl::handler &h) {
                                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                                    int i = it[0];

                                    xehe::w64_t Ai;
                                    Ai.value64b = d_A[i];
                                    xehe::w64_t Bi;
                                    Bi.value64b = d_B[i];
                                    xehe::w64_t Ci;
                                    Ci.value64b = d_C[i];
                                    xehe::w64_t dc;
                                    dc.value64b = d_const_ratio[0];
                                    xehe::w64_t dm;
                                    dm.value64b = *d_m;
                                    d_R[i] = Ai.values32b[0] + Bi.values32b[0] + Ci.values32b[0] + dc.values32b[0] +
                                             dm.values32b[0];

                                });
                            }).wait();
                        };
        }

        cl::sycl::free(d_A, queue);
        cl::sycl::free(d_B, queue);
        cl::sycl::free(d_C, queue);
        cl::sycl::free(d_R, queue);

    } else {
        FAIL();
    }
}




TEST_CASE("64b mul mod operand test with uintarithmod", "[gpu][uintarithmod][uint64][mul]")
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    SECTION("64 bit values") {
        mul_mod_operand_test<uint64_t>(queue, 2305843009211596801ULL, 2305843009211596800ULL, 2305843009211596800ULL);
    }
}


TEST_CASE("64b div by 2 mod test with uintarithmod", "[gpu][uintarithmod][uint64][div2]")
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    SECTION("32 bit #1")
    {
        uint64_t modulus = 3;
        std::vector<uint64_t> Ain{0, 1, modulus - 1};

        div2_mod_test<uint64_t>(queue, modulus, Ain);
    }

    SECTION("32 bit #2")
    {
        uint64_t modulus = 17;
        std::vector<uint64_t> Ain{0, 1, 3, 5, 7, 8, modulus - 1};

        div2_mod_test<uint64_t>(queue, modulus, Ain);
    }

    SECTION("64 bit")
    {
        uint64_t modulus = 0xFFFFFFFFFFFFFFFULL;
        std::vector<uint64_t> Ain{0, 1, 3, modulus - 1};

        div2_mod_test<uint64_t>(queue, modulus, Ain);
    }
}

TEST_CASE("64b muladd mod test with uintarithmod", "[gpu][uintarithmod][uint64][mul][add]")
{

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    SECTION("32 bit inputs")
    {
        uint64_t modulus = 7;
        std::vector<uint64_t> Ain{0, 1, 0, 0, 3, modulus - 1};
        std::vector<uint64_t> Bin{0, 0, 1, 0, 4, modulus - 1};
        std::vector<uint64_t> Cin{0, 0, 0, 1, 5, modulus - 1};
        mul_add_mod_test<uint64_t>(queue, modulus, Ain, Bin, Cin);
    }

    SECTION("64 bit inputs")
    {
        uint64_t modulus = 0x1FFFFFFFFFFFFFFFULL;
        std::vector<uint64_t> Ain{0, 1, 0, 0, 3, modulus - 1};
        std::vector<uint64_t> Bin{0, 0, 1, 0, 4, modulus - 1};
        std::vector<uint64_t> Cin{0, 0, 0, 1, 5, modulus - 1};
        mul_add_mod_test<uint64_t>(queue, modulus, Ain, Bin, Cin);
    }

}

TEST_CASE("64b muladd mod test with explicit mem management", "[gpu][mem][uint64][mul][add]")
{

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();

    SECTION("32 bit inputs")
    {
        uint64_t modulus = 7;
        std::vector<uint64_t> Ain{0, 1, 0, 0, 3, modulus - 1};
        std::vector<uint64_t> Bin{0, 0, 1, 0, 4, modulus - 1};
        std::vector<uint64_t> Cin{0, 0, 0, 1, 5, modulus - 1};
        mul_add_mod_explicit_mem_test<uint64_t>(queue, modulus, Ain, Bin, Cin);
    }

    SECTION("64 bit inputs")
    {
        uint64_t modulus = 0x1FFFFFFFFFFFFFFFULL;
        std::vector<uint64_t> Ain{0, 1, 0, 0, 3, modulus - 1};
        std::vector<uint64_t> Bin{0, 0, 1, 0, 4, modulus - 1};
        std::vector<uint64_t> Cin{0, 0, 0, 1, 5, modulus - 1};
        mul_add_mod_explicit_mem_test<uint64_t>(queue, modulus, Ain, Bin, Cin);
    }

}



#endif // BUILD WITH SEAL



#endif
