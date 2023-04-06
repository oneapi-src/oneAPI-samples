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
        h.parallel_for<kernel_add_ntv_uint<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            
            d_Crry[i] = xehe::native::add_uint<T>(d_A[i], d_B[i], d_C + i);

            });
        });

    queue.wait();

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " + " << d_B[i] << " = " << d_C[i] << " with crry " << d_Crry[i] << "\n";
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
            if (h_C[i] != d_C[i] || h_Crry[i] != d_Crry[i])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
            }
#else
            REQUIRE((h_C[i] == d_C[i] && h_Crry[i] == d_Crry[i]));
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
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
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
class kernel_add_ntv_w64;

template<typename T>
static void
add_ntvw64_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {


    // GPU part setting
    // GPU part setting
    auto d_A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_Crry = cl::sycl::malloc_shared<T>(num_workitems, queue);
    std::vector<T> h_A(num_workitems);
    std::vector<T> h_B(num_workitems);
    std::vector<T> h_C(num_workitems);
    std::vector<T> h_Crry(num_workitems);


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
        h.parallel_for<kernel_add_ntv_w64<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            xehe::w64_t A, B;
            xehe::w64_t C, Crry;
            Crry.part[1] = Crry.part[0] = 0;
            XEHE_READ_W64(d_A, i, 0, A);
            XEHE_READ_W64(d_B, i, 0, B);

            Crry = xehe::native::add_uint<xehe::w64_t>(A, B, &C);

            XEHE_WRITE_W64(d_C, i, 0, C);
            XEHE_WRITE_W64(d_Crry, i, 0, Crry);


            });
        });

    queue.wait();

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " + " << d_B[i] << " = " << d_C[i] << " with carry " << d_Crry[i] << "\n";
        }
#endif

        for (int i = 0; i < range_size; i++)
        {
            h_Crry[i] = xehe::native::add_uint<T>(h_A[i], h_B[i], h_C.data() + i);
        }
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << h_A[i] << " x " << h_B[i] << " = " << h_C[i] << " with crry " << h_Crry[i] << "\n";
        }
#endif

        bool success = true;
        for (int i = 0; i < range_size && success; i++)
        {
#ifdef WIN32
            if (h_C[i] != d_C[i] || h_Crry[i] != d_Crry[i])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
    }
#else
            REQUIRE((h_C[i] == d_C[i] && h_Crry[i] == d_Crry[i]));
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
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];

                    xehe::w64_t A, B;
                    xehe::w64_t C, Crry;
                    Crry.part[1] = Crry.part[0] = 0;
                    XEHE_READ_W64(d_A, i, 0, A);
                    XEHE_READ_W64(d_B, i, 0, B);
                    xehe::native::add_uint<xehe::w64_t>(A, B, &A);
                    for (int j = 0; j < inner_loops; ++j)
                    {
                        Crry = xehe::native::add_uint<xehe::w64_t>(A, C, &C);
                    }

                    XEHE_WRITE_W64(d_C, i, 0, C);
                    XEHE_WRITE_W64(d_Crry, i, 0, Crry);

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
        h.parallel_for<kernel_sub_ntv_uint<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];


            d_Brrw[i] = xehe::native::sub_uint<T>(d_A[i], d_B[i], d_C + i);

            });
        });

    queue.wait();

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " + " << d_B[i] << " = " << d_C[i] << " with borrow " << d_Brrw[i] << "\n";
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
            if (h_C[i] != d_C[i] || h_Brrw[i] != d_Brrw[i])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
            }
#else
            REQUIRE((h_C[i] == d_C[i] && h_Brrw[i] == d_Brrw[i]));
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
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
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


template<typename T>
class kernel_sub_ntv_w64;

template<typename T>
static void
sub_ntvw64_test(cl::sycl::queue& queue, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
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
        h.parallel_for<kernel_sub_ntv_w64<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            xehe::w64_t A, B;
            xehe::w64_t C, Brrw;
            Brrw.part[1] = Brrw.part[0] = 0;
            XEHE_READ_W64(d_A, i, 0, A);
            XEHE_READ_W64(d_B, i, 0, B);

            Brrw = xehe::native::sub_uint<xehe::w64_t>(A, B, &C);

            XEHE_WRITE_W64(d_C, i, 0, C);
            XEHE_WRITE_W64(d_Brrw, i, 0, Brrw);


            });
        });

    queue.wait();

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
        {
            std::cout << d_A[i] << " + " << d_B[i] << " = " << d_C[i] << " with borrow " << d_Brrw[i] << "\n";
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
            if (h_C[i] != d_C[i] || h_Brrw[i] != d_Brrw[i])
            {
                success = false;
                std::cout << "Failed at " << i << " h " << h_C[i] << " d " << d_C[i] << std::endl;
            }
#else
            REQUIRE((h_C[i] == d_C[i] && h_Brrw[i] == d_Brrw[i]));
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
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];

                    xehe::w64_t A, B;
                    xehe::w64_t C, Brrw;
                    Brrw.part[1] = Brrw.part[0] = 0;
                    XEHE_READ_W64(d_A, i, 0, A);
                    XEHE_READ_W64(d_B, i, 0, B);
                    xehe::native::sub_uint<xehe::w64_t>(A, B, &A);
                    for (int j = 0; j < inner_loops; ++j)
                    {
                        Brrw = xehe::native::sub_uint<xehe::w64_t>(A, C, &C);
                    }

                    XEHE_WRITE_W64(d_C, i, 0, C);
                    XEHE_WRITE_W64(d_Brrw, i, 0, Brrw);

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

void Basic_static_native_w64_add(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();


#ifndef WIN32
    SECTION("add w64 bit")
#else
    std::cout << "add w64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        add_ntvw64_test<uint64_t>(queue, Ain, Bin);
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

void Basic_static_native_w64_sub(void) {

    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();


#ifndef WIN32
    SECTION("sub w64 bit")
#else
    std::cout << "sub w64 bit" << std::endl;
#endif
    {
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
        sub_ntvw64_test<uint64_t>(queue, Ain, Bin);
    }


}


#ifndef WIN32

TEST_CASE("Basic static native uint add", "[gpu][uintarith]") {
    Basic_static_native_uint_add();
}

TEST_CASE("Basic static native w64 add", "[gpu][uintarith]") {
    Basic_static_native_w64_add();
}

TEST_CASE("Basic static native uint sub", "[gpu][uintarith]") {
    Basic_static_native_uint_sub();
}

TEST_CASE("Basic static native w64 sub", "[gpu][uintarith]") {
    Basic_static_native_w64_sub();
}

#endif
/*****************************************************************
 *
 *   PERF TESTS
 *
 *****************************************************************/
void Basic_static_bench_native_w64_add(void) {
    {
        xehe::dpcpp::Context ctx;
        auto queue = ctx.queue();

        size_t num_workitems = 100000;

#ifndef WIN32
        SECTION("w64 bit add: 100K threads")
#else
        std::cout << "w64 bit add: " << num_workitems << " threads" << std::endl;
#endif
        {
            std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
            std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
            add_ntvw64_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
        }


    }
}

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

void Basic_static_bench_native_w64_sub(void) {
    {
        xehe::dpcpp::Context ctx;
        auto queue = ctx.queue();

        size_t num_workitems = 100000;

#ifndef WIN32
        SECTION("w64 bit sub: 100K threads")
#else
        std::cout << "w64 bit sub: " << num_workitems << " threads" << std::endl;
#endif
        {
            std::vector<uint64_t> Ain{ 0, 0, 1, 1, 0x1000000000 };
            std::vector<uint64_t> Bin{ 0, 1, 0, 1, 0xFAFABABA };
            sub_ntvw64_test<uint64_t>(queue, Ain, Bin, num_workitems, num_workitems, true);
        }


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
TEST_CASE("Basic static bench native uint add", "[gpu][uintarith]") {
    Basic_static_bench_native_uint_add();
}
TEST_CASE("Basic static bench native w64 add", "[gpu][uintarith]") {
    Basic_static_bench_native_w64_add();
}

TEST_CASE("Basic static bench native uint sub", "[gpu][uintarith]") {
    Basic_static_bench_native_uint_sub();
}
TEST_CASE("Basic static bench native w64 sub", "[gpu][uintarith]") {
    Basic_static_bench_native_w64_sub();
}

#endif



/* ---------------------------------------------------------
//                                   MODULAR ARITHMETICS
 ----------------------------------------------------------*/


template<typename T>
class kernel_add_mod_ntv;


template<typename T>
static void
add_mod_test(cl::sycl::queue &queue, T modulus, std::vector<T> &Ain, std::vector<T> &Bin, size_t num_workitems = 10,
             size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto mod = cl::sycl::malloc_shared<T>(1, queue);
    auto m = static_cast<T>(modulus);
    mod[0] = m;
    std::vector<T> h_C(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

   

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

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }



    queue.submit([&](sycl::handler &h) {
        h.parallel_for<kernel_add_mod_ntv<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            d_C[i] = xehe::native::add_mod<T>(A[i], B[i], mod[0]);

        });
    });

    queue.wait();

    if (!benchmark) {
        for (int i = 0; i < range_size; i++)
        {
            h_C[i] = (A[i] + B[i]) % m; // xehe::native::add_mod<T>(A[i], B[i], m);
        }

#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << A[i] << " + " << B[i] << " mod " << m << " = " << d_C[i] << "\n";
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
    else
    {
        BENCHMARK("simple add mod") {
                                        return queue.submit([&](sycl::handler &h) {
                                            h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
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

//#define VERBOSE_TEST

template<typename T>
class kernel_add_mod_w64;


template<typename T>
static void
add_mod_w64_test(cl::sycl::queue& queue, T modulus, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto mod = cl::sycl::malloc_shared<T>(1, queue);
    auto m = static_cast<T>(modulus);
    mod[0] = m;
    std::vector<T> h_C(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;



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

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }



    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_add_mod_w64<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            xehe::w64_t tA, tB;
            xehe::w64_t tC, tmod;
            tmod.part[0] = ((xehe::w64_t*)mod)->part[0];
            tmod.part[1] = ((xehe::w64_t*)mod)->part[1];
            XEHE_READ_W64(A, i, 0, tA);
            XEHE_READ_W64(B, i, 0, tB);
            tC = xehe::native::add_mod<xehe::w64_t>(tA, tB, tmod);

            XEHE_WRITE_W64(d_C, i, 0, tC);
            });
        });

    queue.wait();

    if (!benchmark) {
        for (int i = 0; i < range_size; i++)
        {
            h_C[i] = (A[i] + B[i]) % m; 
            /*
            xehe::w64_t tA, tB;
            xehe::w64_t tC, tmod;
            tA.value64b = A[i];
            tB.value64b = B[i];
            tmod.value64b = m;
            tC = xehe::native::add_mod<xehe::w64_t>(tA, tB, tmod);
            */

        }

#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << A[i] << " + " << B[i] << " mod " << m << " = " << d_C[i] << "\n";
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
    else
    {
        BENCHMARK("simple add mod") {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];

                    xehe::w64_t tA, tB;
                    xehe::w64_t tC, tmod;
                    tmod.part[0] = ((xehe::w64_t*)mod)->part[0];
                    tmod.part[1] = ((xehe::w64_t*)mod)->part[1];
                    XEHE_READ_W64(A, i, 0, tA);
                    XEHE_READ_W64(B, i, 0, tB);
                    xehe::native::add_uint<xehe::w64_t>(tA, tB, &tA);

                    for (int j = 0; j < inner_loops; ++j)
                    {
                        tC = xehe::native::add_mod<xehe::w64_t>(tA, tC, tmod);
                    }
                    XEHE_WRITE_W64(d_C, i, 0, tC);

                    });
                }).wait();
        };
    }
#endif
}



#if 0


template<typename T>
class kernel_negate_mod;

template<typename T>
static void negate_mod_test(cl::sycl::queue &queue, T mod, T A0, size_t num_workitems = 10, size_t range_size = 10,
                            bool benchmark = false) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto h_C = cl::sycl::malloc_shared<T>(num_workitems, queue);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;

    auto m = static_cast<T>(mod);

    A[0] = A0;

    for (int i = 1; i < num_workitems; i++) {
        A[i] = dis(gen) % m;
    }

#ifdef VERBOSE_TEST
    if (!benchmark) {
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << "\n";
        std::cout << "\n";
    }
#endif

    auto d_m = cl::sycl::malloc_device<T>(1, queue);
    queue.memcpy(d_m, &m, sizeof(T) * 1);

    queue.submit([&](sycl::handler &h) {
        h.parallel_for<kernel_negate_mod<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            d_C[i] = xehe::util::negate_uint_mod<T>(A[i], *d_m);

        });
    }).wait();

    if (!benchmark) {
        for (int i = 0; i < range_size; i++)
            h_C[i] = xehe::util::negate_uint_mod<T>(A[i], m);

#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << " negate " << A[i] << " mod " << m << " = " << d_C[i] << " Host: " << h_C[i] << "\n";
#endif

        for (int i = 0; i < num_workitems; i++)
            REQUIRE(h_C[i] == d_C[i]);

    } else {
        BENCHMARK("simple negate") {
                                       return queue.submit([&](sycl::handler &h) {
                                           h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                                               int i = it[0];

                                               d_C[i] = xehe::util::negate_uint_mod<T>(A[i], *d_m);

                                           });
                                       }).wait();
                                   };
    }


//    BENCHMARK_ADVANCED("advanced negate uint mod")(Catch::Benchmark::Chronometer meter) {
//            auto d_m = cl::sycl::malloc_device<T>(1, queue);
//            queue.memcpy(d_m, &m, sizeof(T) * 1);
//
//            meter.measure([] {
//            return queue.submit([&](sycl::handler &h) {
//            h.parallel_for<kernel_negate_mod<T>>(sycl::range<1>(num_workitems), [=](auto it) {
//                int i = it[0];
//                d_C[i] = xehe::util::negate_uint_mod<T>(A[i], *d_m);
//
//            });
//        }).wait(); });
//    };
}

#endif



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


static void Basic_w64_mod_add_test(void)
{
#ifdef WIN32
    std::cout << "Basic w64 mod add test" << std::endl;
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
        add_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
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
        add_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
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
        add_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }
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

#ifndef WIN32
    SECTION("mod w64")
#else
    std::cout << "mod w64" << std::endl;
#endif
    {
        uint64_t mod = 2305843009211596801ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 1152921504605798400ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 1152921504605798401ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        add_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers, true);
    }

}

#ifndef WIN32
TEST_CASE("Basic uint64 mod add test", "[gpu][uintarithmod][uint64][mod_add]")
{
    Basic_uint64_mod_add_test();
}

TEST_CASE("Basic uint32 mod add test", "[gpu][uintarithmod][uint32][mod_add]")
{
    Basic_uint32_mod_add_test();
}

TEST_CASE("Basic w64 mod add test", "[gpu][uintarithmod][uint64][mod_add]")
{
    Basic_w64_mod_add_test();
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


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;



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

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }



    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_sub_mod_ntv<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            d_C[i] = xehe::native::sub_mod<T>(A[i], B[i], mod[0]);

            });
        });

    queue.wait();

    if (!benchmark) {
        for (int i = 0; i < range_size; i++)
        {
            //h_C[i] = uint64_t(A[i] - B[i]) % m; // incorrect
            h_C[i] = xehe::native::sub_mod<T>(A[i], B[i], m);
        }

#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << A[i] << " - " << B[i] << " mod " << m << " = " << d_C[i] << "\n";
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
    else
    {
        BENCHMARK("simple sub mod") {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
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

//#define VERBOSE_TEST

template<typename T>
class kernel_sub_mod_w64;


template<typename T>
static void
sub_mod_w64_test(cl::sycl::queue& queue, T modulus, std::vector<T>& Ain, std::vector<T>& Bin, size_t num_workitems = 10,
    size_t range_size = 10, bool benchmark = false, uint32_t inner_loops = 100) {
    auto A = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto B = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto d_C = cl::sycl::malloc_shared<T>(num_workitems, queue);
    auto mod = cl::sycl::malloc_shared<T>(1, queue);
    auto m = static_cast<T>(modulus);
    mod[0] = m;
    std::vector<T> h_C(num_workitems);


    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis;



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

    if (!benchmark) {
#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << "\n";
        std::cout << "\n";
#endif
    }



    queue.submit([&](sycl::handler& h) {
        h.parallel_for<kernel_sub_mod_w64<T>>(sycl::range<1>(range_size), [=](auto it) {
            int i = it[0];

            xehe::w64_t tA, tB;
            xehe::w64_t tC, tmod;
            tmod.part[0] = ((xehe::w64_t*)mod)->part[0];
            tmod.part[1] = ((xehe::w64_t*)mod)->part[1];
            XEHE_READ_W64(A, i, 0, tA);
            XEHE_READ_W64(B, i, 0, tB);
            tC = xehe::native::sub_mod<xehe::w64_t>(tA, tB, tmod);

            XEHE_WRITE_W64(d_C, i, 0, tC);
            });
        });

    queue.wait();

    if (!benchmark) {
        for (int i = 0; i < range_size; i++)
        {
            //h_C[i] = (A[i] - B[i]) % m;
            h_C[i] = xehe::native::sub_mod<T>(A[i], B[i], m);

        }

#ifdef VERBOSE_TEST
        for (int i = 0; i < num_workitems; i++)
            std::cout << A[i] << " - " << B[i] << " mod " << m << " = " << d_C[i] << "\n";
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
    else
    {
        BENCHMARK("simple sub mod w64") {
            return queue.submit([&](sycl::handler& h) {
                h.parallel_for<>(sycl::range<1>(range_size), [=](auto it) {
                    int i = it[0];

                    xehe::w64_t tA, tB;
                    xehe::w64_t tC, tmod;
                    tmod.part[0] = ((xehe::w64_t*)mod)->part[0];
                    tmod.part[1] = ((xehe::w64_t*)mod)->part[1];
                    XEHE_READ_W64(A, i, 0, tA);
                    XEHE_READ_W64(B, i, 0, tB);
                    xehe::native::add_uint<xehe::w64_t>(tA, tB, &tA);

                    for (int j = 0; j < inner_loops; ++j)
                    {
                        tC = xehe::native::sub_mod<xehe::w64_t>(tA, tC, tmod);
                    }
                    XEHE_WRITE_W64(d_C, i, 0, tC);

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


static void Basic_w64_mod_sub_test(void)
{
#ifdef WIN32
    std::cout << "Basic w64 mod sub test" << std::endl;
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
        sub_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
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
        sub_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
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
        sub_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers);
    }
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

#ifndef WIN32
    SECTION("mod w64")
#else
    std::cout << "mod w64" << std::endl;
#endif
    {
        uint64_t mod = 2305843009211596801ULL;
        std::vector<uint64_t> Ain{ 0, 0, 1, 1, 1152921504605798400ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        std::vector<uint64_t> Bin{ 0, 1, 0, 1, 1152921504605798401ULL, 1152921504605798401ULL, 2305843009211596800ULL,
                                  7 };
        sub_mod_w64_test<uint64_t>(queue, mod, Ain, Bin, n_workers, n_workers, true);
    }

}




#ifndef WIN32


TEST_CASE("Basic uint64 mod sub test", "[gpu][uintarithmod][uint64][mod_sub]")
{
    Basic_uint64_mod_sub_test();
}

TEST_CASE("Basic uint32 mod sub test", "[gpu][uintarithmod][uint32][mod_sub]")
{
    Basic_uint32_mod_sub_test();
}

TEST_CASE("Basic w64 mod sub test", "[gpu][uintarithmod][uint64][mod_sub]")
{
    Basic_w64_mod_add_test();
}

TEST_CASE("Basic bench mod sub test", "[gpu][uintarithmod][uint][mod_sub]")
{
    Basic_bench_mod_sub_test();
}

#endif




#if 0


TEST_CASE("Basic 64b negate test with uintarithmod", "[gpu][uintarithmod][uint64][negate]")
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    negate_mod_test<uint64_t>(queue, 2305843009211596801ULL, 1ULL);
}

TEST_CASE("Basic 32b negate test with uintarithmod", "[gpu][uintarithmod][uint32][negate]")
{
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    negate_mod_test<uint32_t>(queue, 0xFFFFFFFF, 1ULL, 10);
}

/*
 * *************************************************************
 *               perf benchmarks
 * *************************************************************
 */

TEST_CASE("basic negate mod", "[gpu][perf][negate]") {
    xehe::dpcpp::Context ctx;
    auto queue = ctx.queue();
    size_t num_workitems = 10000000;

    SECTION("32bit simple negate mod: 10M threads", "[gpu][bench][negate][uint32]") {
        negate_mod_test<uint32_t>(queue, 0xFFFFFFFF, 1ULL, num_workitems, num_workitems, true);
    }

    SECTION("32bit simple negate mod: 1 thread", "[gpu][bench][negate][uint32]") {
        negate_mod_test<uint32_t>(queue, 0xFFFFFFFF, 1ULL, num_workitems, 1, true);
    }

    SECTION("64bit simple negate mod: 10M threads", "[gpu][bench][negate][uint64]") {
        negate_mod_test<uint64_t>(queue, 0xFFFFFFFF, 1ULL, num_workitems, num_workitems, true);
    }

    SECTION("64bit simple negate mod: 1 thread", "[gpu][bench][negate][uint64]") {
        negate_mod_test<uint64_t>(queue, 0xFFFFFFFF, 1ULL, num_workitems, 1, true);
    }
}


#endif // if 0

#endif
