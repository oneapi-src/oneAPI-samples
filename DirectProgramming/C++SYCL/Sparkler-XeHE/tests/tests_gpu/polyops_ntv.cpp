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
#include <vector>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <cstdint>
#include <cstdint>
#include <random>
#include <iso646.h>
#include <numeric>


#include "../src/include/native/xe_polyops.hpp"

#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

#ifdef BUILD_WITH_IGPU



#ifdef BUILD_WITH_SEAL

#include "seal/util/defines.h"
#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"
#include "seal/util/polycore.h"
#include "seal/util/polyarithsmallmod.h"
#endif




#define CALL_CPU_VARIANT false

using namespace xehe;
using namespace std;

namespace xehetest {
    namespace util {

        template<typename T>
        void XeHETests_Poly_Ops(int coeff_count_power, int q_base, size_t shift, bool benchmark = false, bool use_USM=true) {
            std::cout << "***************************************************************************" << std::endl;
            std::cout << "Poly Ops " << ((sizeof(T) == 8) ? "64" : "32")
                << "bit"
                << " Poly " << (1<<coeff_count_power)
                << " RNS " << q_base
                << " shift " << shift
                << " benchmark: " << ((benchmark)? "yes" : "no")
                << ((use_USM)? "    USM" : " buffer") <<  ": yes"
                << std::endl;
            std::cout << "***************************************************************************" << std::endl;


            dpcpp::Context ctx;
            auto queue = ctx.queue();


            size_t q_base_sz = q_base;

            size_t n = (size_t(1) << coeff_count_power);
            auto xe_modulus = cl::sycl::malloc_shared<T>(q_base_sz, queue);
            auto xe_mod_inv = cl::sycl::malloc_shared<T>(q_base_sz, queue);
            auto xe_mod_inv2 = cl::sycl::malloc_shared<T>(2 * q_base_sz, queue);

            auto poly = cl::sycl::malloc_shared<T>(n * q_base_sz, queue);
            auto poly2 = cl::sycl::malloc_shared<T>(n * q_base_sz, queue);
            auto poly_res = cl::sycl::malloc_shared<T>(n * q_base_sz, queue);


            std::vector<T> h_poly(n * q_base_sz);
            std::vector<T> h_poly2(n * q_base_sz);
            std::vector<T> hd_poly_res(n * q_base_sz);
            std::vector<T> h_poly_res(n * q_base_sz);

            std::random_device rd;
            std::vector<T> xe_mod_h(q_base_sz);
            std::vector<T> xe_mod_inv_h(q_base_sz);
            std::vector<T> xe_mod_inv2_h(2 * q_base_sz);


#ifdef BUILD_WITH_SEAL
            int prime_bit_size = (sizeof(T) * 8 - 4);
            auto gen_modulus = seal::util::get_primes(n, prime_bit_size, q_base_sz);
            auto pool = seal::MemoryManager::GetPool();
            SEAL_ALLOCATE_ZERO_GET_RNS_ITER(seal_poly, n, q_base_sz, pool);
            SEAL_ALLOCATE_ZERO_GET_RNS_ITER(seal_poly2, n, q_base_sz, pool);
            SEAL_ALLOCATE_ZERO_GET_RNS_ITER(seal_res, n, q_base_sz, pool);


            for (int j = 0; j < q_base_sz; ++j) {
                xe_mod_h[j] = T(gen_modulus[j].value());
            }

#endif

            for (int j = 0; j < q_base_sz; ++j) {
                xe_modulus[j] = xe_mod_h[j];
                // inverse

                xe_mod_inv_h[j] = xehe::native::mod_inverse1(xe_mod_h[j]); // 2^64 / mod
                xe_mod_inv2_h[2 * j] = xehe::native::mod_inverse2(xe_mod_h[j], &xe_mod_inv2_h[2 * j + 1]); // 2^128 / mod

                xe_mod_inv[j] = xe_mod_inv_h[j];
                xe_mod_inv2[2 * j] = xe_mod_inv2_h[2 * j];
                xe_mod_inv2[2 * j + 1] = xe_mod_inv2_h[2 * j + 1];



                for (int i = 0; i < n; ++i) {
                    h_poly[j * n + i] = static_cast<T>(rd()) % xe_mod_h[j];
                    h_poly2[j * n + i] = static_cast<T>(rd()) % xe_mod_h[j];

                    seal_poly[j][i] = h_poly[j * n + i];
                    seal_poly2[j][i] = h_poly2[j * n + i];
                }

            }
            std::memcpy(poly, h_poly.data(), h_poly.size() * sizeof(T));
            std::memcpy(poly2, h_poly2.data(), h_poly2.size() * sizeof(T));
            cl::sycl::buffer<T> poly_buf{h_poly};
            cl::sycl::buffer<T> poly2_buf{h_poly2};
            cl::sycl::buffer<T> poly_res_buf{cl::sycl::range<1>{(n * q_base_sz)} };
            cl::sycl::buffer<T> xe_mod_buf{ xe_mod_h };
            cl::sycl::buffer<T> xe_mod_inv_buf{ xe_mod_inv_h };
            cl::sycl::buffer<T> xe_mod_inv2_buf{ xe_mod_inv2_h };



#ifndef WIN32
            SECTION("PolyCoeffMod")
#else
            std::cout << "PolyCoeffMod" << std::endl;
#endif
            {
                if (!benchmark) {

#ifdef BUILD_WITH_SEAL
                    seal::util::modulo_poly_coeffs(seal_poly, q_base_sz, gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }
#endif
                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));
#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_mod<T>(1, q_base_sz, n, poly, xe_modulus, xe_mod_inv,
                        hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_mod<T>(queue, 1, q_base_sz, n, poly, xe_modulus, xe_mod_inv,
                            poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                        }).wait();
                    }
                    else
                    {
                        //const sycl::property_list rd_only{ read_only };
                        xehe::native::poly_coeff_mod<T>(queue, 1, q_base_sz, n, poly_buf, xe_mod_buf, xe_mod_inv_buf,
                            poly_res_buf);
                        queue.submit([&](sycl::handler& h) {
                            auto aRes =  poly_res_buf.template get_access<cl::sycl::access::mode::read>(h);
                            // copy hostArray to deviceArray
                            h.copy(aRes, hd_poly_res.data());
                        }).wait();
                    }

#endif


                        bool success = true;
                        for (int j = 0; j < q_base_sz && success; ++j) {
                            for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                                if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                                {
                                    success = false;
                                    std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;
                                }
#else
                                REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                            }
                        }
#ifdef WIN32                
                        if (success) {
                            std::cout << "PolyCoeffMod success" << std::endl;
                        }
#endif
                }
#ifndef WIN32
                else {
                    BENCHMARK("PolyCoeffMod benchmark")
                    {
                        return xehe::native::poly_coeff_mod<T>(queue, 1, q_base_sz, n, poly, xe_modulus, xe_mod_inv,
                            poly_res);
                    };
                }
#endif
            }

#ifndef WIN32
            SECTION("PolyCoeffNegMod")
#else
            std::cout << "PolyCoeffNegMod" << std::endl;
#endif

            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::negate_poly_coeffmod(seal_poly, q_base_sz, gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }


#endif
                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_neg_mod(1, q_base_sz, n, poly, xe_modulus, hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_neg_mod<T>(queue, 1, q_base_sz, n, poly, xe_modulus, poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                        }).wait();
                    }
                    else
                    {
                        //const sycl::property_list rd_only{ read_only };
                        xehe::native::poly_coeff_neg_mod<T>(queue, 1, q_base_sz, n, poly_buf, xe_mod_buf,
                            poly_res_buf);

                        queue.submit([&](sycl::handler& h) {
                            auto aRes = poly_res_buf.template get_access<cl::sycl::access::mode::read>(h);
                            // copy hostArray to deviceArray
                            h.copy(aRes, hd_poly_res.data());
                        }).wait();
                    }
#endif

                        bool success = true;
                        for (int j = 0; j < q_base_sz && success; ++j) {
                            for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                                if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                                {
                                    success = false;
                                    std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << poly_res[j * n + i] << std::endl;
                                }
#else
                                REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                            }
                        }
#ifdef WIN32                
                        if (success) {
                            std::cout << "PolyCoeffNegMod success" << std::endl;
                        }
#endif

                }
#ifndef WIN32
                else {
                    BENCHMARK("PolyCoeffNegMod benchmark")
                    {
                        return xehe::native::poly_coeff_neg_mod(queue, 1, q_base_sz, n, poly, xe_modulus,
                            poly_res);
                    };
                }
#endif
            }


#ifndef WIN32
            SECTION("PolyCoeffAddMod")
#else
            std::cout << "PolyCoeffAddMod" << std::endl;
#endif
            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::add_poly_coeffmod(seal_poly, seal_poly2, q_base_sz, gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }

#endif

                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_add_mod(1, q_base_sz, n, poly, poly2, xe_modulus, hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_add_mod<T>(queue, 1, 1, 1, q_base_sz, n, poly, poly2, xe_modulus, poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                         }).wait();
                    }
                    else
                    {
                        xehe::native::poly_coeff_add_mod<T>(queue, 1, q_base_sz, n,
                            poly_buf, poly2_buf,
                            xe_mod_buf,
                            poly_res_buf);

                        queue.submit([&](sycl::handler& h) {
                            auto aRes = poly_res_buf.template get_access<cl::sycl::access::mode::read>(h);
                            // copy hostArray to deviceArray
                            h.copy(aRes, hd_poly_res.data());
                        }).wait();
                    }
#endif

                        bool success = true;
                        for (int j = 0; j < q_base_sz && success; ++j) {
                            for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                                if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                                {
                                    success = false;
                                    std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;
                                }
#else
                                REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                            }
                        }

#ifdef WIN32 
                        if (success) {
                            std::cout << "PolyCoeffAddMod success" << std::endl;
                        }
#endif
                }
#ifndef WIN32 
                else {
                    BENCHMARK("PolyCoeffAddMod benchmark")
                    {
                        return xehe::native::poly_coeff_add_mod(queue, 1, 1, 1, q_base_sz, n, poly, poly2,
                            xe_modulus, poly_res);
                    };

                }
#endif
            }


#ifndef WIN32
            SECTION("PolyCoeffSubMod")
#else
            std::cout << "PolyCoeffSubMod" << std::endl;
#endif
            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::sub_poly_coeffmod(seal_poly, seal_poly2, q_base_sz, gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }

#endif

                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_sub_mod<T>(1, q_base_sz, n, poly, poly2, xe_modulus, hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_sub_mod(queue, 1, 1, 1, q_base_sz, n, poly, poly2, xe_modulus, poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                        }).wait();
                    }
                    else
                    {
                        xehe::native::poly_coeff_sub_mod<T>(queue, 1, q_base_sz, n,
                            poly_buf, poly2_buf,
                            xe_mod_buf,
                            poly_res_buf);

                        auto host_acc = poly_res_buf.template get_access<cl::sycl::access::mode::read>();
                        std::memcpy(hd_poly_res.data(), host_acc.get_pointer(), poly_res_buf.get_size());
                    }


#endif

                        bool success = true;
                        for (int j = 0; j < q_base_sz && success; ++j) {
                            for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                                if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                                {
                                    success = false;
                                    std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;
                                }
#else
                                REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                            }
                        }

#ifdef WIN32 
                        if (success) {
                            std::cout << "PolyCoeffSubMod success" << std::endl;
                        }
#endif
                }
#ifndef WIN32 
                else {
                    BENCHMARK("PolyCoeffSubMod benchmark")
                    {
                        return xehe::native::poly_coeff_sub_mod(queue, 1, 1, 1, q_base_sz, n, poly, poly2,
                            xe_modulus, poly_res);
                    };

                }
#endif
            }


#ifndef WIN32
            SECTION("PolyCoeffAddScalarMod")
#else
            std::cout << "PolyCoeffAddScalarMod" << std::endl;
#endif
            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::add_poly_scalar_coeffmod(seal_poly, q_base_sz, seal_poly2[0][0], gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }
#endif
                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_add_scalar_mod<T>(1, q_base_sz, n, poly, &poly2[0], xe_modulus, hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_add_scalar_mod<T>(queue, 1, q_base_sz, n, poly, &poly2[0], xe_modulus, poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                        }).wait();
                    }
                    else
                    {
                        auto scalar = h_poly2[0];
                        xehe::native::poly_coeff_add_scalar_mod<T>(queue, 1, q_base_sz, n,
                            poly_buf,
                            scalar,
                            xe_mod_buf,
                            poly_res_buf);
                        // host memory access
                        auto host_acc = poly_res_buf.template get_access<cl::sycl::access::mode::read>();
                        std::memcpy(hd_poly_res.data(), host_acc.get_pointer(), poly_res_buf.get_size());

                    }
#endif

                        bool success = true;
                        for (int j = 0; j < q_base_sz && success; ++j) {
                            for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                                if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                                {
                                    success = false;
                                    std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;

                                }

#else
                                REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                            }
                        }

#ifdef WIN32 
                        if (success) {
                            std::cout << "PolyCoeffAddScalarMod success" << std::endl;
                        }
#endif
                }
#ifndef WIN32
                else {
                    BENCHMARK("PolyCoeffAddScalarMod benchmark")
                    {
                        return xehe::native::poly_coeff_add_scalar_mod(queue, 1, q_base_sz, n, poly,
                            &poly2[0], xe_modulus, poly_res);


                    };
                }
#endif
            }


#ifndef WIN32
            SECTION("PolyCoeffSubScalarMod")
#else
            std::cout << "PolyCoeffSubScalarMod" << std::endl;
#endif
            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::sub_poly_scalar_coeffmod(seal_poly, q_base_sz, seal_poly2[0][0], gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }
#endif
                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_sub_scalar_mod(1, q_base_sz, n, poly, &poly2[0], xe_modulus, hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_sub_scalar_mod(queue, 1, q_base_sz, n, poly, &poly2[0], xe_modulus, poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                        }).wait();
                    }
                    else
                    {
                        auto scalar = h_poly2[0];
                        xehe::native::poly_coeff_sub_scalar_mod<T>(queue, 1, q_base_sz, n,
                            poly_buf,
                            scalar,
                            xe_mod_buf,
                            poly_res_buf);
                        // host memory access
                        auto host_acc = poly_res_buf.template get_access<cl::sycl::access::mode::read>();
                        std::memcpy(hd_poly_res.data(), host_acc.get_pointer(), poly_res_buf.get_size());

                    }

#endif

                    bool success = true;
                    for (int j = 0; j < q_base_sz && success; ++j) {
                        for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                            if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                            {
                                success = false;
                                std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;
                            }
#else
                            REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                        }
                    }

#ifdef WIN32 
                    if (success) {
                        std::cout << "PolyCoeffSubScalarMod success" << std::endl;
                    }
#endif
                }
#ifndef WIN32
                else {
                    BENCHMARK("PolyCoeffSubScalarMod benchmark")
                    {
                        return xehe::native::poly_coeff_sub_scalar_mod(queue, 1, q_base_sz, n, poly,
                            &poly2[0], xe_modulus, poly_res);


                    };
                }
#endif
            }

        
#ifndef WIN32
            SECTION("PolyCoeffMulScalarMod")
#else
            std::cout << "PolyCoeffMulScalarMod" << std::endl;
#endif
            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::multiply_poly_scalar_coeffmod(seal_poly, q_base_sz, seal_poly2[0][0], gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }
#endif
                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_mul_scalar_mod<T>(1, q_base_sz, n, poly, &poly2[0], xe_modulus, xe_mod_inv, hd_poly_res.data());
#else
                    if (use_USM)
                    {
                        xehe::native::poly_coeff_mul_scalar_mod<T>(queue, 1, q_base_sz, n, poly, &poly2[0], xe_modulus, xe_mod_inv, poly_res);
                        queue.submit([&](sycl::handler& h) {
                            // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                            }).wait();
                    }
                    else
                    {
                        auto scalar = h_poly2[0];
                        xehe::native::poly_coeff_mul_scalar_mod<T>(queue, 1, q_base_sz, n,
                            poly_buf,
                            scalar,
                            xe_mod_buf,
                            xe_mod_inv_buf,
                            poly_res_buf);
                        // host memory access
                        auto host_acc = poly_res_buf.template get_access<cl::sycl::access::mode::read>();
                        std::memcpy(hd_poly_res.data(), host_acc.get_pointer(), poly_res_buf.get_size());
                    }


#endif

                    bool success = true;
                    for (int j = 0; j < q_base_sz && success; ++j) {
                        for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                            if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                            {
                                success = false;
                                std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;
                            }
#else
                            REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                        }
                    }

#ifdef WIN32 
                    if (success) {
                        std::cout << "PolyCoeffMulScalarMod success" << std::endl;
                    }
#endif
                }
#ifndef WIN32
                else {
                    BENCHMARK("PolyCoeffMulScalarMod benchmark")
                    {
                        return xehe::native::poly_coeff_mul_scalar_mod(queue, 1, q_base_sz, n, poly,
                            &poly2[0], xe_modulus, xe_mod_inv, poly_res);


                    };
                }
#endif
            }


#ifndef WIN32
            SECTION("PolyCoeffProdMod")
#else
            std::cout << "PolyCoeffProdMod" << std::endl;
#endif
            {
                if (!benchmark) {
#ifdef BUILD_WITH_SEAL
                    seal::util::dyadic_product_coeffmod(seal_poly, seal_poly2, q_base_sz, gen_modulus, seal_res);
                    for (int j = 0; j < q_base_sz; ++j) {
                        for (size_t i = 0; i < n; i++) {
                            h_poly_res[j * n + i] = seal_res[j][i];
                        }
                    }
#endif

                    // clear the output for comparison
                    memset(poly_res, 0, q_base_sz * n * sizeof(T));

#if CALL_CPU_VARIANT
                    xehe::native::poly_coeff_prod_mod<T>(1, q_base_sz, n, poly, poly2, xe_modulus, xe_mod_inv2, hd_poly_res.data());
#else
                    if (use_USM)
                    {

                        xehe::native::poly_coeff_prod_mod<T>(queue, 1, q_base_sz, n, poly, poly2, xe_modulus, xe_mod_inv2, poly_res);
                        queue.submit([&](sycl::handler& h) {
                        // copy hostArray to deviceArray
                            h.memcpy(hd_poly_res.data(), poly_res, hd_poly_res.size() * sizeof(T));
                        }).wait();
                    }
                    else
                    {
                        xehe::native::poly_coeff_prod_mod<T>(queue, 1, q_base_sz, n,
                            poly_buf, poly2_buf,
                            xe_mod_buf,
                            xe_mod_inv2_buf,
                            poly_res_buf);

                        auto host_acc = poly_res_buf.template get_access<cl::sycl::access::mode::read>();
                        std::memcpy(hd_poly_res.data(), host_acc.get_pointer(), poly_res_buf.get_size());
                    }
#endif


                        bool success = true;
                        for (int j = 0; j < q_base_sz && success; ++j) {
                            for (size_t i = 0; i < n && success; i++) {
#ifdef WIN32
                                if (hd_poly_res[j * n + i] != h_poly_res[j * n + i])
                                {
                                    success = false;
                                    std::cout << "Failed at " << j << ", " << i << " h " << h_poly_res[j * n + i] << " d " << hd_poly_res[j * n + i] << std::endl;
                                }
#else
                                REQUIRE(hd_poly_res[j * n + i] == h_poly_res[j * n + i]);
#endif
                            }

                        }
#ifdef WIN32
                        if (success) {
                            std::cout << "PolyCoeffProdMod success" << std::endl;
                        }
#endif
                }
#ifndef WIN32
                else {
                    BENCHMARK("PolyCoeffProdMod benchmark")
                    {
                        return xehe::native::poly_coeff_prod_mod<T>(queue, 1, q_base_sz, n, poly, poly2, xe_modulus, xe_mod_inv2, poly_res);
                    };
                }
#endif
            }



        }

#ifndef WIN32
        TEST_CASE("PolySmallModOps Test", "[gpu][PolySmallModOps]")
        {
            XeHETests_Poly_Ops<uint32_t>(13, 4, (size_t(1) << 12), false, false);
            XeHETests_Poly_Ops<uint64_t>(13, 4, (size_t(1) << 12), false, false);
            XeHETests_Poly_Ops<uint32_t>(13, 4, (size_t(1) << 12), false, true);
            XeHETests_Poly_Ops<uint64_t>(13, 4, (size_t(1) << 12), false, true);
            //std::cout << "-------XeHE Poly_ModOps32 tests passed-------" << std::endl;
        }

        TEST_CASE("PolySmallModOps Benchmark", "[gpu][PolySmallModOps][perf]")
        {
            int coeff_count_power = 13;
            int q_base = 4;
            size_t shift = (size_t(1) << 12);
            bool benchmark_flag = true;
            bool USM_flag = false;

            {
                // 32bit : 2 times more primes
                int q_base32 = q_base*2;
                USM_flag = false;

                SECTION("32 bit, Buffers")
                {
                    XeHETests_Poly_Ops<uint32_t>(coeff_count_power, q_base32, shift, benchmark_flag, USM_flag);
                }
                
                USM_flag = true;
                SECTION("32 bit, USM")
                {
                    XeHETests_Poly_Ops<uint32_t>(coeff_count_power, q_base32, shift, benchmark_flag, USM_flag);
                }

            }
            {

                USM_flag = false;
                SECTION("64 bit, Buffers")
                {
                    XeHETests_Poly_Ops<uint64_t>(coeff_count_power, q_base, shift, benchmark_flag, USM_flag);

                }
                USM_flag = true;
                SECTION("64 bit, USM")
                {
                    XeHETests_Poly_Ops<uint64_t>(coeff_count_power, q_base, shift, benchmark_flag, USM_flag);

                }
            }


        }
#endif // #ifndef WIN32




    } // namespace util
} // namespace xehetest



#endif //BUILD_WITH_IGPU