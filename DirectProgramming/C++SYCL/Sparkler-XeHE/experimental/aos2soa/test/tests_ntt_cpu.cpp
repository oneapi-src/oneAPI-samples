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

#ifdef BUILD_WITH_SEAL
#include "seal/modulus.h"
#include "seal/util/ntt.h"
#include "seal/util/numth.h"
#include "seal/util/polycore.h"
#include "seal/util/rns.h"
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

namespace xehetest
{
    namespace util
    {
        template <typename T>
        void NTTTablesTest_NegacyclicNTTTest(void)
        //TEST(NTTTablesTest, NegacyclicNTTTest)
        {
            T xe_modulus(0xffffffffffc0001UL);
            auto two_times_modulus = xe_modulus * 2;
            int coeff_count_power = 1;
            size_t n = (1 << coeff_count_power);
            auto s_roots = allocate<MultiplyUIntModOperand<T>>(n * 1);
            auto roots = s_roots.get();
#ifdef BUILD_WITH_SEAL
            seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::Global();
            seal::util::Pointer<seal::util::NTTTables> tables;
            seal::Modulus modulus(xe_modulus);
            tables = seal::util::allocate<seal::util::NTTTables>(pool, coeff_count_power, modulus, pool);
            for (size_t i = 0; i < n; i++)
            {
                roots[i].operand = (*tables).get_from_root_powers()[i].operand;
                roots[i].quotient = (*tables).get_from_root_powers()[i].quotient;
            }

            auto s_poly_xe = allocate<T>(2* 1);
            auto poly_xe = s_poly_xe.get();
            auto s_poly_seal = allocate<T>(2* 1);
            auto poly_seal = s_poly_seal.get();

            poly_xe[0] = 0;
            poly_xe[1] = 0;
            poly_seal[0] = 0;
            poly_seal[1] = 0;
            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, roots);
            tables->ntt_handler().transform_to_rev(poly_seal, tables->coeff_count_power(), tables->get_from_root_powers());
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= two_times_modulus)
                {
                    *I -= two_times_modulus;
                }
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            REQUIRE(T(0) == poly_xe[0]);
            REQUIRE(T(0) == poly_xe[1]);
            REQUIRE(poly_seal[0] == poly_xe[0]);
            REQUIRE(poly_seal[1] == poly_xe[1]);

            poly_xe[0] = 1;
            poly_xe[1] = 0;
            poly_seal[0] = 1;
            poly_seal[1] = 0;
            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, roots);
            tables->ntt_handler().transform_to_rev(poly_seal, tables->coeff_count_power(), tables->get_from_root_powers());
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= two_times_modulus)
                {
                    *I -= two_times_modulus;
                }
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            REQUIRE(T(1) == poly_xe[0]);
            REQUIRE(T(1) == poly_xe[1]);
            REQUIRE(poly_seal[0] == poly_xe[0]);
            REQUIRE(poly_seal[1] == poly_xe[1]);

            poly_xe[0] = 1;
            poly_xe[1] = 1;
            poly_seal[0] = 1;
            poly_seal[1] = 1;
            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, roots);
            tables->ntt_handler().transform_to_rev(poly_seal, tables->coeff_count_power(), tables->get_from_root_powers());
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= two_times_modulus)
                {
                    *I -= two_times_modulus;
                }
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            REQUIRE(T(288794978602139553ULL) == poly_xe[0]);
            REQUIRE(T(864126526004445282ULL) == poly_xe[1]);
            REQUIRE(poly_seal[0] == poly_xe[0]);
            REQUIRE(poly_seal[1] == poly_xe[1]);


            random_device rd;
            for (size_t i = 0; i < n; i++)
            {
                poly_xe[i] = static_cast<T>(rd()) % modulus.value();
                poly_seal[i] = poly_xe[i];
            }
            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, roots);
            tables->ntt_handler().transform_to_rev(poly_seal, tables->coeff_count_power(), tables->get_from_root_powers());
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= two_times_modulus)
                {
                    *I -= two_times_modulus;
                }
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            REQUIRE(poly_seal[0] == poly_xe[0]);
            REQUIRE(poly_seal[1] == poly_xe[1]);
#endif
        }

        TEST_CASE("XeHE NTTTablesTest NegacyclicNTTTest", "[NegacyclicNTTTest][cpu][XeHE]")
        {

            NTTTablesTest_NegacyclicNTTTest<uint64_t>();
            std::cout << "-------XeHE NTTTablesTest NegacyclicNTTTest64 tests passed-------" << std::endl;
            //NTTTablesTest_NegacyclicNTTTest<uint32_t>();
            //std::cout << "-------XeHE NTTTablesTest NegacyclicNTTTest32 tests passed-------" << std::endl;
        }

        template <typename T>
        void NTT_InverseNegacyclicNTTTest(void)
        //TEST(NTTTablesTest, InverseNegacyclicNTTTest)
        {
            int coeff_count_power = 15;
            T xe_modulus(0xffffffffffc0001ULL);
            auto two_times_modulus = xe_modulus * 2;
            size_t n = (1 << coeff_count_power);
            auto s_roots = allocate<MultiplyUIntModOperand<T>>(n * 1);
            auto roots = s_roots.get();
            auto s_inv_roots = allocate<MultiplyUIntModOperand<T>>(n * 1);
            auto inv_roots = s_inv_roots.get();
            auto s_xe_inv_degree_modulo = allocate<MultiplyUIntModOperand<T>>(1);
            auto xe_inv_degree_modulo = s_xe_inv_degree_modulo.get();
#ifdef BUILD_WITH_SEAL
            seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::Global();
            seal::util::Pointer<seal::util::NTTTables> tables;
            seal::Modulus modulus(xe_modulus);

            tables = seal::util::allocate<seal::util::NTTTables>(pool, coeff_count_power, modulus, pool);
            for (int i = 0; i < n; i++)
            {
                roots[i].operand = (*tables).get_from_root_powers()[i].operand;
                roots[i].quotient = (*tables).get_from_root_powers()[i].quotient;

                inv_roots[i].operand = (*tables).get_from_inv_root_powers()[i].operand;
                inv_roots[i].quotient = (*tables).get_from_inv_root_powers()[i].quotient;
            }
            auto inv_degree_modulo = (*tables).inv_degree_modulo();

            xe_inv_degree_modulo->operand = inv_degree_modulo.operand;
            xe_inv_degree_modulo->quotient = inv_degree_modulo.quotient;

            auto s_poly_xe = allocate<T>(n * 1);
            auto s_poly_seal = allocate<T>(n * 1);
            auto s_temp = allocate<T>(n * 1);
            auto poly_xe = s_poly_xe.get();
            auto poly_seal = s_poly_seal.get();
            auto temp = s_temp.get();
            memset(poly_xe, 0, n * sizeof(T));
            memset(poly_seal, 0, n * sizeof(T));

            //inverse_ntt_negacyclic_harvey(poly.get(), *tables);
            inverse_ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, inv_roots, xe_inv_degree_modulo);
            tables->ntt_handler().transform_from_rev(poly_seal, tables->coeff_count_power(), tables->get_from_inv_root_powers(), &inv_degree_modulo);
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            for (size_t i = 0; i < n; i++)
            {
                REQUIRE(T(0) == poly_xe[i]);
                REQUIRE(poly_seal[i] == poly_xe[i]);
            }

            random_device rd;
            for (size_t i = 0; i < n; i++)
            {
                poly_xe[i] = static_cast<T>(rd()) % modulus.value();
                poly_seal[i] = poly_xe[i];
                temp[i] = poly_xe[i];
            }

            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, roots);
            tables->ntt_handler().transform_to_rev(poly_seal, tables->coeff_count_power(), tables->get_from_root_powers());
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= two_times_modulus)
                {
                    *I -= two_times_modulus;
                }
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            inverse_ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_xe, xe_modulus, coeff_count_power, inv_roots, xe_inv_degree_modulo);
            tables->ntt_handler().transform_from_rev(poly_seal, tables->coeff_count_power(), tables->get_from_inv_root_powers(), &inv_degree_modulo);
            for (size_t i = 0; i < n; ++i)
            {
                auto I = &poly_seal[i];
                if (*I >= xe_modulus)
                {
                    *I -= xe_modulus;
                }
            }
            //ntt_negacyclic_harvey(poly.get(), *tables);
            //inverse_ntt_negacyclic_harvey(poly.get(), *tables);
            for (size_t i = 0; i < n; i++)
            {
                REQUIRE(temp[i] == poly_xe[i]);
                REQUIRE(poly_seal[i] == poly_xe[i]);
            }
#endif
        }

        TEST_CASE("XeHE NTT InverseNegacyclicNTTTest", "[InverseNegacyclicNTTTest][cpu][XeHE]")
        {

            NTT_InverseNegacyclicNTTTest<uint64_t>();
            std::cout << "-------XeHE NTT InverseNegacyclicNTTTest64 tests passed-------" << std::endl;
            //NTT_InverseNegacyclicNTTTest<uint32_t>();
            //std::cout << "-------XeHE NTT InverseNegacyclicNTTTest32 tests passed-------" << std::endl;
        }


        // CPU RNS NTT correctness test, compared to CPU normal NTT for each prime base
        template <typename T>
        void NTT_NegacyclicRNS_NTTTest(void)
        {
            int poly_num = 3;
            int coeff_count_power = 2;
            size_t n = (1 << coeff_count_power);

            std::vector<T> xe_modulus = { 113, 97, 73 };
            size_t q_base_sz = xe_modulus.size();
            std::vector<MultiplyUIntModOperand<T >> roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_degree_modulo(q_base_sz);

#ifdef BUILD_WITH_SEAL
            auto pool = seal::MemoryManager::GetPool();
            seal::util::Pointer<seal::util::RNSTool> rns_tool;
            seal::util::NTTTables ntt[]{ { coeff_count_power, seal::Modulus(xe_modulus[0]) }, { coeff_count_power, seal::Modulus(xe_modulus[1])} , { coeff_count_power, seal::Modulus(xe_modulus[2]) } };

            for (int j = 0; j < q_base_sz; ++j)
            {
                for (int i = 0; i < n; i++)
                {
                    roots[j*n + i].operand = ntt[j].get_from_root_powers()[i].operand;
                    roots[j*n + i].quotient = ntt[j].get_from_root_powers()[i].quotient;

                    inv_roots[j*n + i].operand = ntt[j].get_from_inv_root_powers()[i].operand;
                    inv_roots[j*n + i].quotient = ntt[j].get_from_inv_root_powers()[i].quotient;
                }
                inv_degree_modulo[j].operand = ntt[j].inv_degree_modulo().operand;
                inv_degree_modulo[j].quotient = ntt[j].inv_degree_modulo().quotient;
            }
#endif

            std::vector<T> in_rns(n * q_base_sz * poly_num);
            std::vector<T> in_normal(n * q_base_sz * poly_num);

            // random test
            random_device rd;
            for (T i = 0; i < n * q_base_sz * poly_num; ++i)
            {
                T temp = static_cast<T>(rd()) % xe_modulus[(i >> coeff_count_power) % q_base_sz];
                in_rns[i] = temp;
                in_normal[i] = temp;
            }

            ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_num, q_base_sz,
                                                                                           coeff_count_power,
                                                                                           in_rns.data(),
                                                                                           xe_modulus.data(),
                                                                                           roots.data());
            for (int i = 0; i < q_base_sz * poly_num; ++i) {
                ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(in_normal.data() + i * n,
                                                                                               xe_modulus[i % q_base_sz],
                                                                                               coeff_count_power,
                                                                                               roots.data() + (i % q_base_sz) * n);
            }

            for (size_t i = 0; i < n * q_base_sz * poly_num; i++)
            {
                REQUIRE(in_rns[i] == in_normal[i]);
            }


        }

        TEST_CASE("XeHE NTT NegacyclicRNS_NTT_Test", "[NegacyclicRNS_NTT_Test][cpu][XeHE]")
        {

            NTT_NegacyclicRNS_NTTTest<uint64_t>();
            std::cout << "-------XeHE NTT NegacyclicRNS_NTTTest64 tests passed-------" << std::endl;
            //RNS<uint32_t>();
            //std::cout << "-------XeHE NTT NegacyclicRNS_NTTTest32 tests passed-------" << std::endl;
        }

        // CPU RNS Inverse NTT correctness test, compared to CPU normal Inverse NTT for each prime base
        template <typename T>
        void NTT_InverseNegacyclicRNS_NTTTest(void)
        {
            int poly_num = 3;
            int coeff_count_power = 2;
            size_t n = (1 << coeff_count_power);

            std::vector<T> xe_modulus = { 113, 97, 73 };
            size_t q_base_sz = xe_modulus.size();
            std::vector<MultiplyUIntModOperand<T >> roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_roots(n * q_base_sz);
            std::vector<MultiplyUIntModOperand<T >> inv_degree_modulo(q_base_sz);

#ifdef BUILD_WITH_SEAL
            auto pool = seal::MemoryManager::GetPool();
                    seal::util::Pointer<seal::util::RNSTool> rns_tool;
                    seal::util::NTTTables ntt[]{ { coeff_count_power, seal::Modulus(xe_modulus[0]) }, { coeff_count_power, seal::Modulus(xe_modulus[1])} , { coeff_count_power, seal::Modulus(xe_modulus[2]) } };

                    for (int j = 0; j < q_base_sz; ++j)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            roots[j*n + i].operand = ntt[j].get_from_root_powers()[i].operand;
                            roots[j*n + i].quotient = ntt[j].get_from_root_powers()[i].quotient;

                            inv_roots[j*n + i].operand = ntt[j].get_from_inv_root_powers()[i].operand;
                            inv_roots[j*n + i].quotient = ntt[j].get_from_inv_root_powers()[i].quotient;
                        }
                        inv_degree_modulo[j].operand = ntt[j].inv_degree_modulo().operand;
                        inv_degree_modulo[j].quotient = ntt[j].inv_degree_modulo().quotient;
                    }
#endif

            std::vector<T> in_rns(n * q_base_sz * poly_num);
            std::vector<T> in_normal(n * q_base_sz * poly_num);
            // random test
            random_device rd;
            for (T i = 0; i < n * q_base_sz * poly_num; ++i)
            {
                T temp = static_cast<T>(rd()) % xe_modulus[(i >> coeff_count_power) % q_base_sz];
                in_rns[i] = temp;
                in_normal[i] = temp;
            }

            inverse_ntt_negacyclic_harvey <T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(poly_num, q_base_sz,
                    coeff_count_power,
                    in_rns.data(),
                    xe_modulus.data(),
                    inv_roots.data(),
                    inv_degree_modulo.data());
            for (int i = 0; i < q_base_sz * poly_num; ++i) {
                inverse_ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>> (in_normal.data() + i * n,
                        xe_modulus[i % q_base_sz],
                        coeff_count_power,
                        inv_roots.data() + (i % q_base_sz) * n,
                        &inv_degree_modulo[i % q_base_sz]);
            }

            for (size_t i = 0; i < n * q_base_sz * poly_num; i++)
            {
                REQUIRE(in_rns[i] == in_normal[i]);
            }


        }

        TEST_CASE("XeHE NTT InverseNegacyclicRNS_NTT_Test", "[NegacyclicRNS_NTTInverse_Test][cpu][XeHE]")
        {
            NTT_InverseNegacyclicRNS_NTTTest<uint64_t>();
            std::cout << "-------XeHE NTT InverseNegacyclicRNS_NTTTest64 tests passed-------" << std::endl;
            //RNS<uint32_t>();
            //std::cout << "-------XeHE NTT NegacyclicRNS_NTTTest32 tests passed-------" << std::endl;
        }

    } // namespace util
} // namespace xehetest
