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


#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

#ifdef BUILD_WITH_IGPU

#include "dpcpp_utils.h"
#include <CL/sycl.hpp>
// to remove exceptions

#undef XeHE_DEBUG


#ifdef BUILD_WITH_SEAL

#include "seal/util/defines.h"
#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"
#include "seal/util/ntt.h"
#include "seal/util/numth.h"
#include "seal/util/polycore.h"

#endif

#include "util/common.h"
#include "util/xe_uintarith.h"
#include "util/xe_uintarithmod.h"
#include "util/uintarithsmallmod.h"
#include "util/xe_uintcore.h"
#include "util/xe_ntt.h"
#include "util/xe_ntt_relin.h"
#include "util/xe_intt_rescale.h"

using namespace xehe;
using namespace xehe::util;
using namespace std;


namespace xehetest {
    namespace util {

        // GPU RNS NTT correctness test, compared to CPU RNS NTT
        template <typename T>
        void NTT_NegacyclicRNS_NTTTest(void)
        {
            xehe::dpcpp::Context ctx;
            auto queue = ctx.queue();

            int poly_num = 3;
            int coeff_count_power = 2;
            size_t n = (size_t(1) << coeff_count_power);
            std::vector<T> xe_modulus = {113, 97, 73};
            size_t q_base_sz = xe_modulus.size();
            auto h_xe_modulus = cl::sycl::malloc_shared<T>(q_base_sz, queue);
            memcpy(h_xe_modulus, xe_modulus.data(), q_base_sz * sizeof(T));

            std::vector<MultiplyUIntModOperand<T >> roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_degree_modulo(q_base_sz);
            auto h_roots = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(n * q_base_sz, queue);
            auto h_inv_roots = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(n * q_base_sz, queue);
            auto h_inv_degree_modulo = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(q_base_sz, queue);


#ifdef BUILD_WITH_SEAL
            auto pool = seal::MemoryManager::GetPool();
            seal::util::Pointer<seal::util::RNSTool> rns_tool;
            seal::util::NTTTables ntt[]{{coeff_count_power, seal::Modulus(xe_modulus[0])},
                                        {coeff_count_power, seal::Modulus(xe_modulus[1])},
                                        {coeff_count_power, seal::Modulus(xe_modulus[2])}};

            for (int j = 0; j < q_base_sz; ++j) {
                for (int i = 0; i < n; i++) {
                    roots[j * n + i].operand = ntt[j].get_from_root_powers()[i].operand;
                    roots[j * n + i].quotient = ntt[j].get_from_root_powers()[i].quotient;

                    inv_roots[j * n + i].operand = ntt[j].get_from_inv_root_powers()[i].operand;
                    inv_roots[j * n + i].quotient = ntt[j].get_from_inv_root_powers()[i].quotient;
                }
                inv_degree_modulo[j].operand = ntt[j].inv_degree_modulo().operand;
                inv_degree_modulo[j].quotient = ntt[j].inv_degree_modulo().quotient;
            }

            memcpy(h_roots, roots.data(), n * q_base_sz * sizeof(MultiplyUIntModOperand<T>));
            memcpy(h_inv_roots, inv_roots.data(), n * q_base_sz * sizeof(MultiplyUIntModOperand<T>));
            memcpy(h_inv_degree_modulo, inv_degree_modulo.data(), q_base_sz * sizeof(MultiplyUIntModOperand<T>));
#endif


            std::vector<T> in_rns(n * q_base_sz * poly_num);
            auto h_in_rns = cl::sycl::malloc_shared<T>(n * q_base_sz * poly_num, queue);
            std::vector<T> in_normal(n * q_base_sz * poly_num);
            // random test
            random_device rd;
            for (T i = 0; i < n * q_base_sz * poly_num; ++i)
            {
                T temp = static_cast<T>(rd()) % xe_modulus[(i >> coeff_count_power) % q_base_sz];
                in_rns[i] = temp;
                in_normal[i] = temp;
            }
            memcpy(h_in_rns, in_rns.data(), n * q_base_sz * poly_num * sizeof(T));

            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(queue, poly_num, q_base_sz,
                    coeff_count_power,
                    h_in_rns,
                    h_xe_modulus,
                    h_roots);
            memcpy(in_rns.data(), h_in_rns, n * q_base_sz * poly_num * sizeof(T));
            //cpu called to compare
            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_num, q_base_sz,
                    coeff_count_power,
                    in_normal.data(),
                    xe_modulus.data(),
                    roots.data());

            for (size_t i = 0; i < n * q_base_sz * poly_num; i++)
            {
                REQUIRE(in_rns[i] == in_normal[i]);
            }


        }

        TEST_CASE("GPU Negacyclic RNS NTT Test", "[gpu][RNS_NTT]")
        {
            NTT_NegacyclicRNS_NTTTest<uint64_t>();
            std::cout << "-------XeHE NTT NegacyclicRNS_NTT_Testt64 tests passed-------" << std::endl;
            //NTT_NegacyclicRNS_NTTTest<uint32_t>();
            //std::cout << "-------XeHE NTT NegacyclicRNS_NTT_Testt32 tests passed-------" << std::endl;
        }

        // GPU RNS Inverse NTT correctness test, compared to CPU RNS Inverse NTT
        template <typename T>
        void NTT_InverseNegacyclicRNS_NTTTest(void)
        {
            xehe::dpcpp::Context ctx;
            auto queue = ctx.queue();

            int poly_num = 3;
            int coeff_count_power = 2;
            size_t n = (size_t(1) << coeff_count_power);
            std::vector<T> xe_modulus = {113, 97, 73};
            size_t q_base_sz = xe_modulus.size();
            auto h_xe_modulus = cl::sycl::malloc_shared<T>(q_base_sz, queue);
            memcpy(h_xe_modulus, xe_modulus.data(), q_base_sz * sizeof(T));

            std::vector<MultiplyUIntModOperand<T >> roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_degree_modulo(q_base_sz);
            auto h_roots = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(n * q_base_sz, queue);
            auto h_inv_roots = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(n * q_base_sz, queue);
            auto h_inv_degree_modulo = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(q_base_sz, queue);


#ifdef BUILD_WITH_SEAL
            auto pool = seal::MemoryManager::GetPool();
            seal::util::Pointer<seal::util::RNSTool> rns_tool;
            seal::util::NTTTables ntt[]{{coeff_count_power, seal::Modulus(xe_modulus[0])},
                                        {coeff_count_power, seal::Modulus(xe_modulus[1])},
                                        {coeff_count_power, seal::Modulus(xe_modulus[2])}};

            for (int j = 0; j < q_base_sz; ++j) {
                for (int i = 0; i < n; i++) {
                    roots[j * n + i].operand = ntt[j].get_from_root_powers()[i].operand;
                    roots[j * n + i].quotient = ntt[j].get_from_root_powers()[i].quotient;

                    inv_roots[j * n + i].operand = ntt[j].get_from_inv_root_powers()[i].operand;
                    inv_roots[j * n + i].quotient = ntt[j].get_from_inv_root_powers()[i].quotient;
                }
                inv_degree_modulo[j].operand = ntt[j].inv_degree_modulo().operand;
                inv_degree_modulo[j].quotient = ntt[j].inv_degree_modulo().quotient;
            }

            memcpy(h_roots, roots.data(), n * q_base_sz * sizeof(MultiplyUIntModOperand<T>));
            memcpy(h_inv_roots, inv_roots.data(), n * q_base_sz * sizeof(MultiplyUIntModOperand<T>));
            memcpy(h_inv_degree_modulo, inv_degree_modulo.data(), q_base_sz * sizeof(MultiplyUIntModOperand<T>));
#endif

            std::vector<T> in_rns(n * q_base_sz * poly_num);
            auto h_in_rns = cl::sycl::malloc_shared<T>(n * q_base_sz * poly_num, queue);
            std::vector<T> in_normal(n * q_base_sz * poly_num);
            // random test
            random_device rd;
            for (T i = 0; i < n * q_base_sz * poly_num; ++i)
            {
                T temp = static_cast<T>(rd()) % xe_modulus[(i >> coeff_count_power) % q_base_sz];
                in_rns[i] = temp;
                in_normal[i] = temp;
            }
            memcpy(h_in_rns, in_rns.data(), n * q_base_sz * poly_num * sizeof(T));

            inverse_ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(queue,
                    poly_num,
                    q_base_sz,
                    coeff_count_power,
                    h_in_rns, h_xe_modulus,
                    h_inv_roots,
                    h_inv_degree_modulo);
            memcpy(in_rns.data(), h_in_rns, n * q_base_sz * poly_num * sizeof(T));
            //cpu called to compare
            inverse_ntt_negacyclic_harvey <T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_num, q_base_sz,
                    coeff_count_power,
                    in_normal.data(),
                    xe_modulus.data(),
                    inv_roots.data(),
                    inv_degree_modulo.data());

            for (size_t i = 0; i < n * q_base_sz * poly_num; i++)
            {
                REQUIRE(in_rns[i] == in_normal[i]);
            }

        }

        TEST_CASE("GPU Negacyclic RNS Inverse NTT Test", "[gpu][RNS_NTT_Inverse]")
        {
            NTT_InverseNegacyclicRNS_NTTTest<uint64_t>();
            std::cout << "-------XeHE NTT NegacyclicRNS_NTTInverse_Testt64 tests passed-------" << std::endl;
            //NTT_InverseNegacyclicRNS_NTTTest<uint32_t>();
            //std::cout << "-------XeHE NTT NegacyclicRNS_NTTInverse_Testt32 tests passed-------" << std::endl;
        }

        template<typename T>
        void NTT_NegacyclicRNS_VarTest(int coeff_count_power, int q_base, bool benchmark = false) {

            xehe::dpcpp::Context ctx;
            auto queue = ctx.queue();

            size_t q_base_sz = q_base;

            size_t n = (size_t(1) << coeff_count_power);
            auto xe_modulus = cl::sycl::malloc_shared<T>(q_base_sz, queue);

            auto roots = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(n * q_base_sz, queue);
            auto inv_roots = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(n * q_base_sz, queue);
            auto inv_degree_modulo = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(q_base_sz, queue);


#ifdef BUILD_WITH_SEAL

            auto gen_modulus = seal::util::get_primes(n, 60, q_base_sz);
            auto pool = seal::MemoryManager::GetPool();
            seal::util::Pointer<seal::util::RNSTool> rns_tool;
            std::vector<seal::util::NTTTables> ntt;

            for (int j = 0; j < q_base_sz; ++j) {
                xe_modulus[j] = gen_modulus[j].value();
                seal::util::NTTTables ntt({coeff_count_power, seal::Modulus(gen_modulus[j])});
                for (int i = 0; i < n; i++) {
                    roots[j * n + i].operand = ntt.get_from_root_powers()[i].operand;
                    roots[j * n + i].quotient = ntt.get_from_root_powers()[i].quotient;

                    inv_roots[j * n + i].operand = ntt.get_from_inv_root_powers()[i].operand;
                    inv_roots[j * n + i].quotient = ntt.get_from_inv_root_powers()[i].quotient;
                }
                inv_degree_modulo[j].operand = ntt.inv_degree_modulo().operand;
                inv_degree_modulo[j].quotient = ntt.inv_degree_modulo().quotient;
            }

#endif

            auto in = cl::sycl::malloc_shared<T>(n * q_base_sz, queue);
            auto tmp = cl::sycl::malloc_shared<T>(n * q_base_sz, queue);

            std::random_device rd;
            for (int j = 0; j < q_base_sz; ++j) {
                for (int i = 0; i < n; ++i) {
                    in[j * n + i] = static_cast<T>(rd()) % xe_modulus[j];
                }
            }
            memcpy(tmp, in, n * q_base_sz * sizeof(T));


            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(queue, int(q_base_sz),
                                                                                           coeff_count_power, in,
                                                                                           xe_modulus, roots);

            inverse_ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(queue,
                                                                                                   int(q_base_sz),
                                                                                                   coeff_count_power,
                                                                                                   in, xe_modulus,
                                                                                                   inv_roots,
                                                                                                   inv_degree_modulo);

            for (size_t i = 0; i < n * q_base_sz; i++) {
                REQUIRE(tmp[i] == in[i]);
            }

            if (benchmark) {
                BENCHMARK("NTT RNS Negacyclic Harvey")
                            {
                                return ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(
                                        queue, int(q_base_sz), coeff_count_power, in,
                                        xe_modulus, roots);

                            };

                BENCHMARK("Inverse NTT RNS Negacyclic Harvey")
                            {
                                return inverse_ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(
                                        queue, int(q_base_sz), coeff_count_power, in, xe_modulus, inv_roots,
                                        inv_degree_modulo);


                            };
            }

        }

        TEST_CASE("Negacyclic RNS Var Test", "[gpu][RNS_Var]")
        {
            int coeff_count_power = 14;
            int q_base = 16;

            std::cout << "Poly order " << (1 << coeff_count_power) << " RNS base size " << q_base << " Modulus width "
                      << (sizeof(uint64_t) * 8) << std::endl;

            NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base);
            std::cout << "-------XeHE NegacyclicRNS_VarTest64 tests passed-------" << std::endl;
            //NTT_NegacyclicRNS_VarTest<uint32_t>(coeff_count_power, q_base);
            //std::cout << "-------XeHE NegacyclicRNS_VarTest32 tests passed-------" << std::endl;
        }


        TEST_CASE("Negacyclic RNS NTT Benchmark", "[gpu][perf][RNS_NTT]") {
            NTT_NegacyclicRNS_NTTTest<uint64_t>();

            bool benchmark_flag = true;
            int q_base = 16;

            SECTION("RNS NTT negacyclic harvey log2(power)=14 qbase=16")
            {
                int coeff_count_power = 14;

                NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base, benchmark_flag);
            }

            SECTION("RNS NTT negacyclic harvey log2(power)=15 qbase=16")
            {
                int coeff_count_power = 14;
                NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base, benchmark_flag);
            }

            SECTION("RNS NTT negacyclic harvey log2(power)=16 qbase=16")
            {
                int coeff_count_power = 16;
                NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base, benchmark_flag);
            }

            q_base = 17;

            SECTION("RNS NTT negacyclic harvey log2(power)=14 qbase=17")
            {
                int coeff_count_power = 14;

                NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base, benchmark_flag);
            }

            SECTION("RNS NTT negacyclic harvey log2(power)=15 qbase=17")
            {
                int coeff_count_power = 14;
                NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base, benchmark_flag);
            }

            SECTION("RNS NTT negacyclic harvey log2(power)=16 qbase=17")
            {
                int coeff_count_power = 16;
                NTT_NegacyclicRNS_VarTest<uint64_t>(coeff_count_power, q_base, benchmark_flag);
            }
        }

    } // namespace util
} // namespace xehetest


#endif //BUILD_WITH_IGPU