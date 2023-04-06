/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifdef SEAL_USE_INTEL_XEHE

#ifndef XEHE_SEAL_PLGIN_HPP
#define XEHE_SEAL_PLGIN_HPP

#include <assert.h>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <iso646.h>
#include <random>
#include <vector>
#include <cstddef>
#include <ctime>
#include <string>

#include "seal/xehe_seal_plgin.fwd.h"

#include "seal/batchencoder.h"
#include "seal/ckks.h"
#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"


using namespace std;

namespace xehe
{
    namespace plgin
    {
        template <typename T, typename S>
        SEAL_NODISCARD inline bool are_same_scale(const T &value1, const S &value2) noexcept
        {
            return seal::util::are_close<double>(value1.scale(), value2.scale());
        }

        SEAL_NODISCARD inline bool is_scale_within_bounds(
            double scale, const seal::SEALContext::ContextData &context_data) noexcept
        {
            int scale_bit_count_bound = 0;
            switch (context_data.parms().scheme())
            {
            case seal::scheme_type::bfv:
                scale_bit_count_bound = context_data.parms().plain_modulus().bit_count();
                break;
            case seal::scheme_type::ckks:
                scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
                break;
            default:
                // Unsupported scheme; check will fail
                scale_bit_count_bound = -1;
            };

            return !(scale <= 0 || (static_cast<int>(log2(scale)) >= scale_bit_count_bound));
        }

        template <typename T>
        std::shared_ptr<xehe::ext::Buffer<T>> &get_ciphertext_gpu_mem(seal::Ciphertext &seal_buffer)
        {
            return (seal_buffer.get_gpu_cipher().get_gpu_mem<T>());
        }

        template <typename T>
        void set_ciphertext_gpu_mem(seal::Ciphertext &seal_buffer, std::shared_ptr<xehe::ext::Buffer<T>> b)
        {
            seal_buffer.get_gpu_cipher().get_gpu_mem<T>() = b;
        }

        template <typename T>
        bool is_ciphertext_host_mem_dirty(const seal::Ciphertext &c)
        {
            return (c.is_dirty());
        }

        template <typename T>
        void set_ciphertext_host_mem_dirty(seal::Ciphertext &c, bool flag = false)
        {
            c.set_dirty(flag);
        }

        template<typename T> 
        T calc_quotient(const seal::Modulus &modulus, T operand)
        {
#ifdef SEAL_DEBUG
            if (operand >= T(modulus.value()))
            {
                throw std::invalid_argument("input must be less than modulus");
            }
#endif
            T quotient = 0;
            if constexpr (sizeof(T) == 8)
            {
                T num2[2]{ 0, operand };
                T quo2[2];
                seal::util::divide_uint128_inplace(num2, modulus.value(), quo2);
                quotient = quo2[0];
            }
            else if constexpr (sizeof(T) == 4)
            {
                uint64_t wide_op = (uint64_t(operand) << 32);
                quotient = T(wide_op / modulus.value());
            }

            return (quotient);
        }

        template<typename T>
        void calc_quotient2(T *quo2, const seal::Modulus &modulus, T operand)
        {
            auto mod = modulus.value();
#ifdef SEAL_DEBUG
            if (operand >= mod)
            {
                throw std::invalid_argument("input must be less than modulus");
            }
#endif
            if constexpr (sizeof(T) == 8)
            {
                T num3[3]{ 0, 0, operand };
                T quo3[3];
                seal::util::divide_uint192_inplace(num3, mod, quo3);
                quo2[0] = quo3[0];
                quo2[1] = quo3[1];
            }
            else if constexpr (sizeof(T) == 4)
            {
                *((uint64_t *)quo2) = calc_quotient<uint64_t>(modulus, uint64_t(operand));
            }
        }



        template <typename T>
        class XEHEvaluator
        {
        public:
            XEHEvaluator(void)
            {
                //xehe::ext::activate_memory_cache();
            }

            ~XEHEvaluator(void)
            {
                //xehe::ext::free_memory_cache();
                //xehe::ext::free_free_cache();
                size_t alloced, freed;
                xehe::ext::get_mem_cache_stat(alloced, freed);
            }

            XEHEvaluator(const seal::SEALContext *seal_ctx)
            {
                init(seal_ctx);
            }

            void init(const seal::SEALContext *seal_ctx)
            {
                seal_ctx_ = seal_ctx;
                // Extract encryption parameters.
                auto &ctx = *get_ctx();
                // the last level
                auto &context_data = *(ctx.key_context_data());

                ckks_scheme_ = (context_data.parms().scheme() == seal::scheme_type::ckks);
                if (ckks_scheme_)
                {
                    std::cout << "XeHE Evaluator Plugin is instantiated with CKKS context" << std::endl;
                }
                else
                {
                    std::cout << "XeHE Evaluator Plugin Error: unsupported scheme" << std::endl;
                }
            }

            inline bool is_ckks(void) const
            {
                return (ckks_scheme_);
            }

            /**
            Upload precomputed data into GPU memory buffers in GPU-friendly layout.
            The data presumed to be invariant over the SEALContext lifetime.

            */

            void params_to_gpu(void)
            {
                std::lock_guard<std::mutex> lk(gpu_invdata_lock_);
                if (get_gpu_map().empty())
                {
                    auto context_data_ptr = get_ctx()->key_context_data();
                    auto &parms0 = context_data_ptr->parms();
                    auto &coeff_modulus0 = parms0.coeff_modulus();
                    auto const_ratio_sz = coeff_modulus0[0].const_ratio().size();

                    while (context_data_ptr)
                    {
                        xehe::ext::XeHE_mem_context<T> gpu_context_entry;

                        auto &parms = context_data_ptr->parms();
                        auto &coeff_modulus = parms.coeff_modulus();
                        auto coeff_count = parms.poly_modulus_degree();
                        auto q_base_sz = coeff_modulus.size();
                        auto NTTTables = context_data_ptr->small_ntt_tables();
                        auto rns_tool = context_data_ptr->rns_tool();
                        auto inv_q_last_mod_q = rns_tool->inv_q_last_mod_q();
                        auto next_context_data_ptr = context_data_ptr->next_context_data();
                        size_t next_coeff_modulus_size = 0;
                        if (next_context_data_ptr)
                        {
                            auto &next_parms = (*next_context_data_ptr).parms();
                            next_coeff_modulus_size = next_parms.coeff_modulus().size();
                        }

                        gpu_context_entry.xe_modulus = xehe::ext::XeHE_malloc<T>(q_base_sz/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        gpu_context_entry.xe_inv1 = xehe::ext::XeHE_malloc<T>(q_base_sz/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        gpu_context_entry.xe_inv2 = xehe::ext::XeHE_malloc<T>(2 * q_base_sz/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        // gpu_context_entry.xe_inv_modulus = xehe::ext::XeHE_malloc<T>(q_base_sz * const_ratio_sz);
                        //                        gpu_context_entry.xe_prim_roots =
                        //                            xehe::ext::XeHE_malloc<xehe::ext::MulModOperand<T>>(q_base_sz *
                        //                            coeff_count);
                        gpu_context_entry.xe_prim_roots_op = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        gpu_context_entry.xe_prim_roots_quo = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        //                        gpu_context_entry.xe_inv_prim_roots =
                        //                            xehe::ext::XeHE_malloc<xehe::ext::MulModOperand<T>>(q_base_sz *
                        //                            coeff_count);
                        gpu_context_entry.xe_inv_prim_roots_op = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        gpu_context_entry.xe_inv_prim_roots_quo = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        //                        gpu_context_entry.xe_inv_degree =
                        //                            xehe::ext::XeHE_malloc<xehe::ext::MulModOperand<T>>(q_base_sz);
                        gpu_context_entry.xe_inv_degree_op = xehe::ext::XeHE_malloc<T>(q_base_sz/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        gpu_context_entry.xe_inv_degree_quo = xehe::ext::XeHE_malloc<T>(q_base_sz/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        //                        gpu_context_entry.xe_inv_q_last_mod_q =
                        //                            xehe::ext::XeHE_malloc<xehe::ext::MulModOperand<T>>(q_base_sz -
                        //                            1);
                        if (q_base_sz - 1 > 0)
                        {
                            gpu_context_entry.xe_inv_q_last_mod_q_op = xehe::ext::XeHE_malloc<T>(q_base_sz - 1/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                            gpu_context_entry.xe_inv_q_last_mod_q_quo = xehe::ext::XeHE_malloc<T>(q_base_sz - 1/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                        }
                        //
                        std::vector<T> cpu_modulus(q_base_sz);
                        std::vector<T> cpu_inv1(q_base_sz);
                        std::vector<T> cpu_inv2(2 * q_base_sz);
                        //                        std::vector<xehe::ext::MulModOperand<T>> cpu_roots(q_base_sz *
                        //                        coeff_count);
                        std::vector<T> cpu_roots_op(q_base_sz * coeff_count);
                        std::vector<T> cpu_roots_quo(q_base_sz * coeff_count);
                        //                        std::vector<xehe::ext::MulModOperand<T>> cpu_inv_roots(q_base_sz *
                        //                        coeff_count);
                        std::vector<T> cpu_inv_roots_op(q_base_sz * coeff_count);
                        std::vector<T> cpu_inv_roots_quo(q_base_sz * coeff_count);
                        //                        std::vector<xehe::ext::MulModOperand<T>>
                        //                        inv_degree_modulus(q_base_sz);
                        std::vector<T> cpu_inv_degree_modulus_op(q_base_sz);
                        std::vector<T> cpu_inv_degree_modulus_quo(q_base_sz);
                        //                        std::vector<xehe::ext::MulModOperand<T>>
                        //                        cpu_inv_q_last_mod_q(q_base_sz - 1);
                        std::vector<T> cpu_inv_q_last_mod_q_op(q_base_sz - 1);
                        std::vector<T> cpu_inv_q_last_mod_q_quo(q_base_sz - 1);

                        for (size_t j = 0; j < q_base_sz; j++)
                        {
                            cpu_modulus[j] = T(coeff_modulus[j].value());
                            auto mod = cpu_modulus[j];
                            // inv1
                            cpu_inv1[j] = calc_quotient<T>(mod, T(1));

                            // inv2
                            calc_quotient2<T>(&cpu_inv2[2 * j], mod, T(1));

                            auto ntt = NTTTables[j];
                            if constexpr (sizeof(T) == 8)
                            {
                                for (size_t i = 0; i < coeff_count; i++)
                                {
                                    cpu_roots_op[j * coeff_count + i] = ntt.get_from_root_powers()[i].operand;
                                    cpu_roots_quo[j * coeff_count + i] = ntt.get_from_root_powers()[i].quotient;

                                    cpu_inv_roots_op[j * coeff_count + i] = ntt.get_from_inv_root_powers()[i].operand;
                                    cpu_inv_roots_quo[j * coeff_count + i] = ntt.get_from_inv_root_powers()[i].quotient;
                                }
                                cpu_inv_degree_modulus_op[j] = ntt.inv_degree_modulo().operand;
                                cpu_inv_degree_modulus_quo[j] = ntt.inv_degree_modulo().quotient;

                                if (j < q_base_sz - 1)
                                {
                                    cpu_inv_q_last_mod_q_op[j] = inv_q_last_mod_q[j].operand;
                                    cpu_inv_q_last_mod_q_quo[j] = inv_q_last_mod_q[j].quotient;
                                }
                            }
                            else if constexpr (sizeof(T) == 4)
                            {
                                for (size_t i = 0; i < coeff_count; i++)
                                {
                                    auto root_op = T(ntt.get_from_root_powers()[i].operand);
                                    cpu_roots_op[j * coeff_count + i] = root_op;
                                    cpu_roots_quo[j * coeff_count + i] = calc_quotient<T>(coeff_modulus[j], root_op);

                                    auto inv_root_op = T(ntt.get_from_inv_root_powers()[i].operand);
                                    cpu_inv_roots_op[j * coeff_count + i] = inv_root_op;
                                    cpu_inv_roots_quo[j * coeff_count + i] =
                                        calc_quotient<T>(coeff_modulus[j], inv_root_op);
                                }
                                auto mod_op = ntt.inv_degree_modulo().operand;
                                cpu_inv_degree_modulus_op[j] = T(mod_op);
                                cpu_inv_degree_modulus_quo[j] = calc_quotient<T>(coeff_modulus[j], T(mod_op));

                                if (j < q_base_sz - 1)
                                {
                                    auto inv_q_last_mod_q_op = T(inv_q_last_mod_q[j].operand);
                                    cpu_inv_q_last_mod_q_op[j] = inv_q_last_mod_q_op;
                                    cpu_inv_q_last_mod_q_quo[j] =
                                        calc_quotient<T>(coeff_modulus[j], inv_q_last_mod_q_op);
                                }
                            }
                        }

                        // bulk copy
                        // ptr , size , off
                        //                        gpu_context_entry.xe_prim_roots->set_data(cpu_roots.data(), q_base_sz
                        //                        * coeff_count, 0);
                        gpu_context_entry.xe_prim_roots_op->set_data(cpu_roots_op.data(), q_base_sz * coeff_count, 0, false);
                        gpu_context_entry.xe_prim_roots_quo->set_data(cpu_roots_quo.data(), q_base_sz * coeff_count, 0, false);

                        //                        gpu_context_entry.xe_inv_prim_roots->set_data(cpu_inv_roots.data(),
                        //                        q_base_sz * coeff_count, 0);
                        gpu_context_entry.xe_inv_prim_roots_op->set_data(
                            cpu_inv_roots_op.data(), q_base_sz * coeff_count, 0, false);
                        gpu_context_entry.xe_inv_prim_roots_quo->set_data(
                            cpu_inv_roots_quo.data(), q_base_sz * coeff_count, 0, false);

                        //                        gpu_context_entry.xe_inv_degree->set_data(inv_degree_modulus.data(),
                        //                        q_base_sz, 0);
                        gpu_context_entry.xe_inv_degree_op->set_data(cpu_inv_degree_modulus_op.data(), q_base_sz, 0, false);
                        gpu_context_entry.xe_inv_degree_quo->set_data(cpu_inv_degree_modulus_quo.data(), q_base_sz, 0, false);

                        if ( q_base_sz - 1 > 0)
                        {
                            gpu_context_entry.xe_inv_q_last_mod_q_op->set_data(
                                cpu_inv_q_last_mod_q_op.data(), q_base_sz - 1, 0, false);
                            gpu_context_entry.xe_inv_q_last_mod_q_quo->set_data(
                                cpu_inv_q_last_mod_q_quo.data(), q_base_sz - 1, 0, false);
                        }
                        gpu_context_entry.xe_modulus->set_data(cpu_modulus.data(), q_base_sz, 0, false);

                        gpu_context_entry.xe_inv1->set_data(cpu_inv1.data(), q_base_sz, 0, false);
                        gpu_context_entry.xe_inv2->set_data(cpu_inv2.data(), 2 * q_base_sz, 0, true);
                        // gpu_context_entry.xe_inv_modulus->set_data(
                        //    coeff_modulus[0].const_ratio().data(), q_base_sz * const_ratio_sz);
                        // gpu_context_entry.xe_inv2->set_value(quo3[0], 1, 2 * j);
                        // gpu_context_entry.xe_inv2->set_value(quo3[1], 1, 2 * j + 1);
                        //                        gpu_context_entry.xe_inv_q_last_mod_q->set_data(cpu_inv_q_last_mod_q.data(),
                        //                        q_base_sz - 1, 0);

                        // xehe::ext::wait_for_queue(/*queue = 0*/);

                        gpu_context_entry.rns_base_size = q_base_sz;
                        gpu_context_entry.coeff_count = coeff_count;
                        gpu_context_entry.mod_inv_size = const_ratio_sz;
                        gpu_context_entry.next_rns_base_size = next_coeff_modulus_size;
                        get_gpu_map()[context_data_ptr->parms_id()] = gpu_context_entry;

                        context_data_ptr = context_data_ptr->next_context_data();
                    }

                }
            }

            /**
            Upload precomputed relinarization keys data into GPU memory buffers in GPU-friendly layout.
            The data presumed to be invariant over the SEALContext lifetime.

            */

            void key_to_gpu_internal(xehe::ext::XeHE_mem_keys<T> &gpu_struct, const seal::KSwitchKeys &destination)
            {
                auto &context = *get_ctx();
                size_t coeff_count = context.key_context_data()->parms().poly_modulus_degree();
                // size_t decomp_mod_count = context.first_context_data()->parms().coeff_modulus().size();
                auto &key_context_data = *context.key_context_data();
                auto &key_parms = key_context_data.parms();
                auto &key_modulus = key_parms.coeff_modulus();

                // -> vector<PublicKeys> -> Cybertext
                //[decomp_modulus_size][key_component_count][key_modulus_size][coeff_count]
                auto &key_vector0 = destination.data()[0];
                size_t key_component_count = key_vector0[0].data().size();
                gpu_struct.num_keys = destination.size();
                gpu_struct.rns_base_size = destination.data().data()->size();
                gpu_struct.n_polys = key_component_count;
                gpu_struct.key_rns_base_size = key_modulus.size();
                gpu_struct.coeff_count = coeff_count;
                uint64_t key_poly_sz = gpu_struct.n_polys * gpu_struct.key_rns_base_size * gpu_struct.coeff_count;
                uint64_t alloc_sz = gpu_struct.num_keys * gpu_struct.rns_base_size * key_poly_sz;
                gpu_struct.xe_keys = xehe::ext::XeHE_malloc<T>(alloc_sz/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                /*
                std::cout << "n keys " << gpu_struct.num_keys << " rns sz " << gpu_struct.rns_base_size
                    << " comp_count " << gpu_struct.n_polys
                    << " key rns sz " << gpu_struct.key_rns_base_size
                    << " n " << gpu_struct.coeff_count
                    << " total " << alloc_sz
                    << std::endl;
                */
                for (int i = 0; i < int(gpu_struct.num_keys); ++i)
                {
                    auto &key_vector = destination.data()[size_t(i)];
                    for (size_t j = 0; j < gpu_struct.rns_base_size; j++)
                    {
                        // src ptr, size, offset
                        gpu_struct.xe_keys->template set_data_adapter<uint64_t>(
                            key_vector[j].data().data(), key_poly_sz, j * key_poly_sz);
                    }
                }
            }

       
            std::shared_ptr<xehe::ext::Buffer<T>> &upload_plaintext_gpu_mem_lazy(const seal::Plaintext &p)
            {
                // uploading only if the plain is use in the op
                {
                    std::lock_guard<std::mutex> lk(gpu_plain_upload_lock_);   
                    if (p.is_dirty())
                    {
                        //TODO wait as a parameter
                        ((seal::Plaintext &)p).lazy_upload();
                        xehe::ext::wait_for_queue(/*queue = 0*/);
                    }
                }
                return (((seal::Plaintext &)p).lazy_upload());
            }

            /**
            Negates a ciphertext.

            @param[in] encrypted The ciphertext to negate
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */
            void negate_inplace(seal::Ciphertext &encrypted)
            {
                negate(encrypted, encrypted);
            }

            void negate(const seal::Ciphertext &encrypted, seal::Ciphertext &destination, bool allow_aliasing = false)
            {
                if (encrypted.on_gpu() || destination.on_gpu())
                {
                    // upload static precumputed context data

                    if (!encrypted.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted).gpu();
                    }
                    if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }

                    params_to_gpu();

                    // Extract encryption parameters.
                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly = cypherdata_to_gpu(encrypted);
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_res =
                        cypherdata_to_gpu(destination, poly, allow_aliasing, true);
                    auto enc_sz = gpu_ctx_params(gpu_context_entry, encrypted);

                    xehe::ext::XeHE_negate(int(enc_sz), poly, gpu_context_entry, poly_res);

                    // destination buffer is going to be downloded only on decryption call
                }
            }

            /**
            Sums up 2 ciphertext messages to produce ciphertext of the sum

            @param[in] encrypted1 The input1
            @param[in] encrypted2 The input2
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */
            void add(
                const seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2, seal::Ciphertext &destination,
                bool allow_aliasing = false)
            {
                //if (encrypted1.on_gpu() || encrypted2.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted1.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted1).gpu();
                    }
                    //if (!encrypted2.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted2).gpu();
                    }
                    //if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }
                    // upload static precumputed context data
                    params_to_gpu();

                    // Extract encryption parameters.

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly1;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly2;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_res;

                    size_t enc1_sz, enc2_sz, max_sz;
                    double new_scale;
                    max_sz = cypherdata_to_gpu3(
                        gpu_context_entry,
                        enc1_sz,   // n polys in input 1
                        enc2_sz,   // n polys in input 2
                        new_scale, // new CKKS scale = input1.scale*input2.scale
                        poly1, poly2, poly_res, encrypted1, encrypted2,
                        destination, // output only
                        allow_aliasing);

                    xehe::ext::XeHE_add(
                        int(max_sz), int(enc1_sz), int(enc2_sz), poly1, poly2, gpu_context_entry, poly_res);
                }
            }

            void add_inplace(seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2)
            {
                add(encrypted1, encrypted2, encrypted1, true);
            }

            /**
            Substruct 1 ciphertext message from the other to produce a ciphertext of the difference

            @param[in] encrypted1 The input1
            @param[in] encrypted2 The input2
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */
            void sub(
                const seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2, seal::Ciphertext &destination,
                bool allow_aliasing = false)
            {
                //if (encrypted1.on_gpu() || encrypted2.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted1.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted1).gpu();
                    }
                    //if (!encrypted2.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted2).gpu();
                    }
                    //if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }
                    // upload static precumputed context data
                    params_to_gpu();

                    // Extract encryption parameters.
                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;

                    std::shared_ptr<xehe::ext::Buffer<T>> poly1;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly2;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_res;

                    size_t enc1_sz, enc2_sz, max_sz;
                    double new_scale;
                    max_sz = cypherdata_to_gpu3(
                        gpu_context_entry,
                        enc1_sz,   // n polys in input 1
                        enc2_sz,   // n polys in input 2
                        new_scale, // new CKKS scale = input1.scale*input2.scale
                        poly1, poly2, poly_res, encrypted1, encrypted2,
                        destination, // output only
                        allow_aliasing);

                    xehe::ext::XeHE_sub(
                        int(max_sz), int(enc1_sz), int(enc2_sz), poly1, poly2, gpu_context_entry, poly_res);

                    // result is going to be sync with destination host buffer only on decryption call
                }
            }

            void sub_inplace(seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2)
            {
                sub(encrypted1, encrypted2, encrypted1, true);
            }

            /**
            Add a plaintext message to a ciphertext message to produce a ciphertext of the sum.

            @param[in] encrypted The encrypted input
            @param[in] plain The plain input
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */
            void add_plain_ckks(
                const seal::Ciphertext &encrypted, const seal::Plaintext &plain, seal::Ciphertext &destination,
                bool allow_aliasing = false)
            {
                //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();
                    

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    // uploading only if the plain is use in the op
                    auto pln = upload_plaintext_gpu_mem_lazy(plain);

                    // buffers
                    auto poly = cypherdata_to_gpu(encrypted);
                    auto poly_res = cypherdata_to_gpu(destination, poly, allow_aliasing, true);
                    gpu_ctx_params(gpu_context_entry, encrypted);

                    // only the first poly is affected
                    xehe::ext::XeHE_add(1, 1, 1, poly, pln, gpu_context_entry, poly_res);

                    // result is going to be sync with destination host buffer only on decryption call
                }
            }

            void add_plain_inplace_ckks(seal::Ciphertext &encrypted, const seal::Plaintext &plain)
            {
                add_plain_ckks((const seal::Ciphertext &)encrypted, plain, encrypted, true);
            }

            /**
            Sabstruct a plaintext message from a ciphertext message to produce a ciphertext of the difference.

            @param[in] encrypted The encrypted input
            @param[in] plain The plain input
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */

            void sub_plain_ckks(
                const seal::Ciphertext &encrypted, const seal::Plaintext &plain, seal::Ciphertext &destination,
                bool allow_aliasing = false)
            {
                //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    // uploading only if the plain is use in the op
                    auto pln = upload_plaintext_gpu_mem_lazy(plain);

                    // buffers
                    auto poly = cypherdata_to_gpu(encrypted);
                    auto poly_res = cypherdata_to_gpu(destination, poly, allow_aliasing, true);
                    gpu_ctx_params(gpu_context_entry, encrypted);

                    // only the first poly is affected
                    xehe::ext::XeHE_sub(1, 1, 1, poly, pln, gpu_context_entry, poly_res);

                    // result is going to be sync with destination host buffer only on decryption call
                }
            }

            void sub_plain_inplace_ckks(seal::Ciphertext &encrypted, const seal::Plaintext &plain)
            {
                sub_plain_ckks((const seal::Ciphertext &)encrypted, plain, encrypted, true);
            }

            /**
            Multiply a ciphertext message with a plaintext message to produce a ciphertext of the sum.

            @param[in] encrypted The encrypted input
            @param[in] plain The plain input
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */
            void mul_plain_ckks(
                const seal::Ciphertext &encrypted, const seal::Plaintext &plain, seal::Ciphertext &destination,
                bool allow_aliasing = false)
            {
                //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();

                    // Verify parameters.
                    if (!plain.is_ntt_form())
                    {
                        throw invalid_argument("plain_ntt is not in NTT form");
                    }
                    if (encrypted.parms_id() != plain.parms_id())
                    {
                        throw invalid_argument("encrypted_ntt and plain_ntt parameter mismatch");
                    }

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;

                    // uploading only if the plain is use in the op
                    auto pln = upload_plaintext_gpu_mem_lazy(plain);

                    // buffers
                    auto poly = cypherdata_to_gpu(encrypted);
                    auto poly_res = cypherdata_to_gpu(destination, poly, allow_aliasing, true);
                    auto enc_sz = gpu_ctx_params(gpu_context_entry, encrypted);

                    xehe::ext::XeHE_multiply_plain_ckks(int(enc_sz), poly, pln, gpu_context_entry, poly_res);

                    // result is going to be sync with destination host buffer only on decryption call
                    // Set the scale
                    double new_scale = encrypted.scale() * plain.scale();
                    destination.scale() = new_scale;
                }
            }

            void mul_plain_inplace_ckks(seal::Ciphertext &encrypted, const seal::Plaintext &plain)
            {
                mul_plain_ckks((const seal::Ciphertext &)encrypted, plain, encrypted, true);
            }

            /**
            Multiply up 2 ciphertext messages to produce ciphertext of the product.

            @param[in] encrypted1 The input1
            @param[in] encrypted2 The input2
            @param[out] destination The output ciphertext
            @param[in] allow_aliasing Permission to overlap input and output buffers

            */
            void multiply_ckks(
                const seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2, seal::Ciphertext &destination,
                bool allow_aliasing = false)
            {
                //if (encrypted1.on_gpu() || encrypted2.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted1.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted1).gpu();
                    }
                    //if (!encrypted2.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted2).gpu();
                    }
                    //if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }
                    // upload static precumputed context data
                    params_to_gpu();

                    if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
                    {
                        throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
                    }

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly1;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly2;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_res;

                    size_t enc1_sz, enc2_sz, max_sz;
                    double new_scale;

                    max_sz = cypherdata_to_gpu3(
                        gpu_context_entry,
                        enc1_sz,   // n polys in input 1
                        enc2_sz,   // n polys in input 2
                        new_scale, // new CKKS scale = input1.scale*input2.scale
                        poly1, poly2, poly_res, encrypted1, encrypted2, destination, allow_aliasing,
                        true,   // output only
                        false); // multiplier

                    xehe::ext::XeHE_mul_ckks(max_sz, enc1_sz, enc2_sz, poly1, poly2, gpu_context_entry, poly_res);

                    // Set the scale
                    destination.scale() = new_scale;
                }
            }

            void multiply_inplace_ckks(seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2)
            {
                multiply_ckks(encrypted1, encrypted2, encrypted1, true);
            }


            void multiply_plain_add_ckks(
                const seal::Ciphertext &encrypted_add, const seal::Ciphertext &encrypted_mul, const seal::Plaintext &plain,
                seal::Ciphertext &destination, bool allow_aliasing = false)
            {
                //if (encrypted_mul.on_gpu() || encrypted_add.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted_mul.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted_mul).gpu();
                    }
                    //if (!encrypted_add.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted_add).gpu();
                    }
                    //if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }
                    // upload static precumputed context data
                    params_to_gpu();

                    // Verify parameters.
                    if (!plain.is_ntt_form())
                    {
                        throw invalid_argument("plain_ntt is not in NTT form");
                    }
                    if (encrypted_add.parms_id() != plain.parms_id())
                    {
                        throw invalid_argument("encrypted_mul and plain_ntt parameter mismatch");
                    }

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;

                    // uploading only if the plain is use in the op
                    auto pln = upload_plaintext_gpu_mem_lazy(plain);

                    std::shared_ptr<xehe::ext::Buffer<T>> poly_add;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_mul;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_res;

                    size_t enc_add_sz, enc_mul_sz, max_sz;
                    double new_scale;
                    max_sz = cypherdata_to_gpu3(
                        gpu_context_entry,
                        enc_add_sz,   // n polys in input 1
                        enc_mul_sz,   // n polys in input 2
                        new_scale,
                        poly_add, poly_mul, poly_res, encrypted_add, encrypted_mul,
                        destination, // output only
                        allow_aliasing);
                    new_scale = std::max(encrypted_mul.scale() * plain.scale(), encrypted_add.scale());
                    xehe::ext::XeHE_multiply_plain_ckks_add(int(max_sz), int(enc_add_sz), int(enc_mul_sz), 
                                                            poly_add, poly_mul, pln,
                                                            gpu_context_entry, poly_res);

                    // result is going to be sync with destination host buffer only on decryption call
                    // Set the scale
                    destination.scale() = new_scale;
                }
            }

            void multiply_plain_add_inplace_ckks(seal::Ciphertext &encrypted_add, const seal::Ciphertext &encrypted_mul, const seal::Plaintext &plain)
            {
                multiply_plain_add_ckks((const seal::Ciphertext &)encrypted_add, encrypted_mul, plain, encrypted_add, true);
            }

            void multiply_add_ckks(
                const seal::Ciphertext &encrypted_add, const seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2,
                seal::Ciphertext &destination, bool allow_aliasing = false)
            {
                //if (encrypted1.on_gpu() || encrypted2.on_gpu() || encrypted_add.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted1.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted1).gpu();
                    }
                    //if (!encrypted2.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted2).gpu();
                    }
                    //if (!encrypted_add.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted_add).gpu();
                    }
                    //if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }
                    // upload static precumputed context data
                    params_to_gpu();

                    if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
                    {
                        throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
                    }

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly1;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly2;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_add;
                    std::shared_ptr<xehe::ext::Buffer<T>> poly_res;

                    size_t enc1_sz, enc2_sz, enc_add_sz, max_sz;
                    double new_scale;

                    max_sz = cypherdata_to_gpu_mad(
                        gpu_context_entry,
                        enc1_sz,   // n polys in input 1
                        enc2_sz,   // n polys in input 2
                        enc_add_sz,
                        new_scale, // new CKKS scale = input1.scale*input2.scale
                        poly1, poly2, poly_add, poly_res,
                        encrypted1, encrypted2, encrypted_add, destination,
                        allow_aliasing, true);   // output only

                    if (enc1_sz!=2 || enc2_sz!=2 || max_sz!=3){
                        std::cout << "Unsupported ciphertext size for MAD operation" << std::endl;
                        return;
                    }

                    xehe::ext::XeHE_multiply_ckks_add(int(max_sz), int(enc_add_sz), int(enc1_sz), int(enc2_sz), 
                                                      poly_add, poly1, poly2, gpu_context_entry, poly_res);

                    // Set the scale
                    destination.scale() = new_scale;
                }
            }

            void multiply_add_inplace_ckks(seal::Ciphertext &encrypted_add, const seal::Ciphertext &encrypted1, const seal::Ciphertext &encrypted2)
            {
                multiply_add_ckks((const seal::Ciphertext &)encrypted_add, encrypted1, encrypted2, encrypted_add, true);
            }


            /**
            Relinearize an input ciphertext messages with relianirization keys to produce a new ciphertext
            with a low number of encrypted polinomials.
            The processing is done in-place, that destroys the content of the input GPU buffers.

            @param[in/out] encrypted The input
            @param[in] relin_keys The precomputed reliarization keys
            @param[in] destination_size The number of polynomial in the resulting ciphertext

            */
            void relinearize_internal(
                seal::Ciphertext &encrypted, const seal::RelinKeys &relin_keys, size_t destination_size)
            {
                //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();
                    encrypted.gpu();
                    auto context_data_ptr = (*get_ctx()).get_context_data(encrypted.parms_id());

                    if (!context_data_ptr)
                    {
                        throw invalid_argument("encrypted is not valid for encryption parameters");
                    }
                    if (relin_keys.parms_id() != (*get_ctx()).key_parms_id())
                    {
                        throw invalid_argument("relin_keys is not valid for encryption parameters");
                    }

                    size_t encrypted_size = encrypted.size();

                    // Verify parameters.

                    if (destination_size < 2 || destination_size > encrypted_size)
                    {
                        throw invalid_argument(
                            "destination_size must be at least 2 and less than or equal to current count");
                    }
                    if (relin_keys.size() < seal::util::sub_safe(encrypted_size, size_t(2)))
                    {
                        throw invalid_argument("not enough relinearization keys");
                    }

                    // If encrypted is already at the desired level, return
                    if (destination_size == encrypted_size)
                    {
                        return;
                    }

                    // Calculate number of relinearize_one_step calls needed
                    size_t relins_needed = encrypted_size - destination_size;

                    // Iterator pointing to the last component of encrypted
                    auto encrypted_iter = seal::util::iter(encrypted);
                    encrypted_iter += encrypted_size - 1;

                    size_t key_component_count = 2;
                    auto output = cypherdata_to_gpu(encrypted);
                    size_t RNS_poly_len;
                    // Extract encryption parameters.
                    {
                        xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                        gpu_ctx_params(gpu_context_entry, encrypted);
                        RNS_poly_len = gpu_context_entry.coeff_count * gpu_context_entry.rns_base_size;
                    }
                    // temp
                    // read write
                    // will be reused


                    //static 
                    std::shared_ptr< xehe::ext::Buffer<T>> input;
                    //if ( input == nullptr || input->get_size() < (key_component_count * RNS_poly_len))
                    {
                        input = xehe::ext::XeHE_malloc<T>((key_component_count * RNS_poly_len));
                    }


                    for (size_t i = 0; i < relins_needed; ++i)
                    {
                        // src buffer, len, dst buffer offset, src buffer offset
                        input->deep_copy(output, RNS_poly_len, 0, RNS_poly_len * (encrypted_size - 1 - i));

                        switch_key_inplace_ckks(
                            encrypted, output, input, static_cast<const seal::KSwitchKeys &>(relin_keys),
                            seal::RelinKeys::get_index(encrypted_size - 1 - i));
                    }

                    // Put the output of final relinearization into destination.
                    // Prepare destination only at this point because we are resizing down
                    encrypted.resize(*get_ctx(), context_data_ptr->parms_id(), destination_size);
                }
            }

            /**
            Switch keys of the last level of the an input ciphertext messages with provided keys to produce a new
            ciphertext with a lower number of encrypted polinomials. The processing is done in-place, that destroys the
            content of the input GPU buffers. In case of realization keys are already in GPU memory Galois keys are
            lazily uploaded and cached in GPU memory.

            @param[in] encrypted The input ciphertext
            @param[in/out] output The input/output GPU buffer
            @param[in/out] input The key swith target GPU buffer; temp mempory, reused while in processing
            @param[in] kswitch_keys The public switch keys
            @param[in] kswitch_keys_index The index in the array of switch keys related to the position of the last
            polinomila to be processed
            @param[in] relinear_keys True if this is realinization keys, False if it's Galois keys.

            */
            void switch_key_inplace_ckks(
                seal::Ciphertext &encrypted, std::shared_ptr<xehe::ext::Buffer<T>> output,
                std::shared_ptr<xehe::ext::Buffer<T>> input, const seal::KSwitchKeys &kswitch_keys,
                size_t kswitch_keys_index, bool relinear_keys = true)
            {
                // upload static precumputed context data
                params_to_gpu();

                auto parms_id = encrypted.parms_id();
                auto &context_data = *(*get_ctx()).get_context_data(parms_id);
                auto &parms = context_data.parms();

                auto scheme = parms.scheme();

                if (scheme != seal::scheme_type::ckks)
                {
                    std::cout << "XeHE Error: non supported scheme " << std::endl;
                    return;
                }

                // Verify parameters.
                if (!is_metadata_valid_for(encrypted, *get_ctx()) || !is_buffer_valid(encrypted))
                {
                    throw invalid_argument("encrypted is not valid for encryption parameters");
                }

                if (!(*get_ctx()).using_keyswitching())
                {
                    throw logic_error("keyswitching is not supported by the context");
                }

                // Don't validate all of kswitch_keys but just check the parms_id.
                if (kswitch_keys.parms_id() != (*get_ctx()).key_parms_id())
                {
                    throw invalid_argument("parameter mismatch");
                }

                if (kswitch_keys_index >= kswitch_keys.data().size())
                {
                    throw out_of_range("kswitch_keys_index");
                }

                if (scheme == seal::scheme_type::ckks && !encrypted.is_ntt_form())
                {
                    throw invalid_argument("CKKS encrypted must be in NTT form");
                }

                // Extract encryption parameters.
                size_t decomp_modulus_size = 0;
                size_t coeff_count = 0;
                {
                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    gpu_ctx_params(gpu_context_entry, encrypted);
                    decomp_modulus_size = gpu_context_entry.rns_base_size;
                    coeff_count = gpu_context_entry.coeff_count;
                }
                size_t rns_modulus_size = decomp_modulus_size + 1;

                // Size check
                if (!seal::util::product_fits_in(coeff_count, rns_modulus_size, size_t(2)))
                {
                    throw logic_error("invalid parameters");
                }

                xehe::ext::XeHE_mem_context<T> gpu_key_context_entry;
                extract_key_params(gpu_key_context_entry);

                // TODO: move it to init
                // key structure
                // -> vector<PublicKeys> -> Cybertext
                //[decomp_modulus_size][key_component_count][key_modulus_size][coeff_count]
                auto &key_vector = kswitch_keys.data()[kswitch_keys_index];
                size_t key_component_count = key_vector[0].data().size();
                // Check only the used component in KSwitchKeys.

                xehe::ext::XeHE_mem_keys<T> *gpu_struct;
                // relinearization
                if (relinear_keys)
                {
                    // reduce number of primes in the RNS base
                    // key are saved with the max size of RNS base
                    {
                        std::lock_guard<std::mutex> lk(gpu_relin_lock_);

                        gpu_struct = &get_gpu_relin_keys();
                    // lazy upload
                        if (gpu_struct->xe_keys == nullptr)
                        {
                            key_to_gpu_internal(*gpu_struct, kswitch_keys);

                            xehe::ext::wait_for_queue(/*queue = 0*/);
                        }
                        
                    }
                }
                // Galois keys
                else
                {
                    // cache it for the future use
                    std::shared_ptr<xehe::ext::Buffer<T>> xe_key_vector;
                    {
                        std::lock_guard<std::mutex> lk(gpu_galois_lock_);

                        if (get_gpu_galois_keys().size() <= kswitch_keys_index)
                        {
                           get_gpu_galois_keys().resize(kswitch_keys_index + 1);
                        }

                        gpu_struct = &get_gpu_galois_keys()[kswitch_keys_index];
                    // lazy upload
                        if ((xe_key_vector = gpu_struct->xe_keys) == nullptr)
                        {
                            size_t keys_sz = key_component_count * coeff_count * gpu_key_context_entry.rns_base_size;
                            xe_key_vector = xehe::ext::XeHE_malloc<T>(keys_sz * decomp_modulus_size/*, xehe::ext::Buffer<T>::NONE_CACHED*/);
                            for (size_t j = 0; j < decomp_modulus_size; j++)
                            {
                            // src ptr, size, offset
                                xe_key_vector->template set_data_adapter<uint64_t>(key_vector[j].data().data(), keys_sz, j * keys_sz);
                            }
                            gpu_struct->xe_keys = xe_key_vector;
                            gpu_struct->rns_base_size = decomp_modulus_size;
                            gpu_struct->coeff_count = coeff_count;
                            gpu_struct->key_rns_base_size = gpu_key_context_entry.rns_base_size;
                            gpu_struct->n_polys = key_component_count;
                            gpu_struct->num_keys = 1;

                            xehe::ext::wait_for_queue(/*queue = 0*/);
                        }

                    }
                }

                if (decomp_modulus_size != gpu_struct->rns_base_size)
                {
                    gpu_struct->rns_base_size = decomp_modulus_size;
                }

                xehe::ext::XeHE_relinearize(kswitch_keys_index, *gpu_struct, gpu_key_context_entry, input, output);
            }

            /**
            Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
            stores the result in the destination parameter.

            @param[in] encrypted The input ciphertext
            @param[out] destination The ciphertext with the modulus switched result. A new GPU buffer is going to be
            allocated if encrypted and destination have their GPU buffer aliased
            */
            void mod_switch_scale_to_next(
                const seal::Ciphertext &encrypted, seal::Ciphertext &destination, bool allow_aliasing = false)
            {
                //if (encrypted.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted).gpu();
                    }
                   // if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }

                    // upload static precumputed context data
                    params_to_gpu();

                    // Extract encryption parameters.
                    auto &context_data = *(*get_ctx()).get_context_data(encrypted.parms_id());

                    // Assuming at this point encrypted is already validated.

                    if (context_data.parms().scheme() == seal::scheme_type::bfv && encrypted.is_ntt_form())
                    {
                        throw invalid_argument("BFV encrypted cannot be in NTT form");
                    }
                    if (context_data.parms().scheme() == seal::scheme_type::ckks && !encrypted.is_ntt_form())
                    {
                        throw invalid_argument("CKKS encrypted must be in NTT form");
                    }

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    auto input = cypherdata_to_gpu(encrypted);
                    auto encrypted_size = gpu_ctx_params(gpu_context_entry, encrypted);
                    auto &next_context_data = *context_data.next_context_data();
                    destination.resize((*get_ctx()), next_context_data.parms_id(), encrypted_size);

                    size_t new_dest_sz =
                        encrypted_size * gpu_context_entry.next_rns_base_size * gpu_context_entry.coeff_count;
                    auto output = cypherdata_to_gpu(destination, input, allow_aliasing, true, new_dest_sz);

                    xehe::ext::XeHE_divide_round_q_last_ckks(encrypted_size, gpu_context_entry, input, output);



                    // Set other attributes
                    destination.is_ntt_form() = encrypted.is_ntt_form();
                    // Change the scale when using CKKS
                    destination.scale() =
                        encrypted.scale() / static_cast<double>(context_data.parms().coeff_modulus().back().value());
                }
            }

            /**
            Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
            scales the message down accordingly.
            @param[in] encrypted The ciphertext to be switched to a smaller modulus
            @param[out] destination The ciphertext to be scaled down
            */
            void mod_switch_drop_to_next_ckks(
                const seal::Ciphertext &encrypted, seal::Ciphertext &destination, bool allow_aliasing = false)
            {
               //if (encrypted.on_gpu() || destination.on_gpu())
                {
                    //if (!encrypted.on_gpu())
                    {
                        ((seal::Ciphertext &)encrypted).gpu();
                    }
                    //if (!destination.on_gpu())
                    {
                        destination.gpu();
                    }
                    // upload static precumputed context data
                    params_to_gpu();

                    // Assuming at this point encrypted is already validated.
                    auto context_data_ptr = (*get_ctx()).get_context_data(encrypted.parms_id());
                    if (context_data_ptr->parms().scheme() == seal::scheme_type::ckks && !encrypted.is_ntt_form())
                    {
                        throw invalid_argument("CKKS encrypted must be in NTT form");
                    }

                    // Extract encryption parameters.
                    auto &next_context_data = *context_data_ptr->next_context_data();
                    // auto& next_parms = next_context_data.parms();

                    if (!is_scale_within_bounds(encrypted.scale(), next_context_data))
                    {
                        throw invalid_argument("scale out of bounds");
                    }

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    auto input = cypherdata_to_gpu(encrypted);
                    auto enc_sz = gpu_ctx_params(gpu_context_entry, encrypted);
                    // Resize destination before writing
                    destination.resize(*get_ctx(), next_context_data.parms_id(), enc_sz);
                    size_t new_dest_sz = enc_sz * gpu_context_entry.next_rns_base_size * gpu_context_entry.coeff_count;
                    auto output = cypherdata_to_gpu(destination, input, allow_aliasing, true, new_dest_sz);
                    // Size check
                    if (!seal::util::product_fits_in(
                            enc_sz, gpu_context_entry.coeff_count, gpu_context_entry.next_rns_base_size))
                    {
                        throw logic_error("invalid parameters");
                    }

                    xehe::ext::XeHE_mod_switch_ckks(
                        int(enc_sz),                               // number of output compunents
                        int(gpu_context_entry.next_rns_base_size), // destination rns base size
                        gpu_context_entry,                         // context
                        input, output);


                    destination.is_ntt_form() = true;
                    destination.scale() = encrypted.scale();
                }
            }

            /**
           Squares a ciphertext. This functions computes the square of encrypted and stores the result in the
           destination parameter.

           @param[in/out] encrypted The ciphertext to square
           */
            void square_inplace_ckks(seal::Ciphertext &encrypted)
            {
               //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();
                    encrypted.gpu();

                    if (!encrypted.is_ntt_form())
                    {
                        throw invalid_argument("encrypted must be in NTT form");
                    }

                    // Extract encryption parameters.
                    auto &context_data = *(*get_ctx()).get_context_data(encrypted.parms_id());
                    auto &parms = context_data.parms();

                    auto scheme = parms.scheme();

                    if (scheme != seal::scheme_type::ckks)
                    {
                        std::cout << "XeHE Error: non supported scheme " << std::endl;
                        return;
                    }
                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;

                    size_t enc1_sz, enc2_sz, max_sz;
                    double new_scale;
                    std::shared_ptr<xehe::ext::Buffer<T>> inp_output;
                    max_sz = cypherdata_to_gpu3(
                        gpu_context_entry,
                        enc1_sz,   // n polys in input 1
                        enc2_sz,   // n polys in input 2
                        new_scale, // new CKKS scale = input1.scale*input2.scale
                        inp_output, inp_output, inp_output, encrypted, encrypted,
                        encrypted, // resized with max_sz
                        true,
                        false, // do not discard output
                        false); // not addition but multiplication

                    // Optimization implemented currently only for size 2 ciphertexts
                    if (enc1_sz != 2)
                    {
                        return;
                    }

                    if (!is_scale_within_bounds(new_scale, context_data))
                    {
                        throw invalid_argument("scale out of bounds");
                    }

                    // Determine destination.size()
                    // Default is 3 (c_0, c_1, c_2)
                    size_t dest_size = max_sz; // seal::util::sub_safe(seal::util::add_safe(encrypted_size,
                                               // encrypted_size), size_t(1));

                    // Size check
                    if (!seal::util::product_fits_in(
                            dest_size, gpu_context_entry.coeff_count, gpu_context_entry.rns_base_size))
                    {
                        throw logic_error("invalid parameters");
                    }

                    // on GPU
                    xehe::ext::XeHE_square(
                        int(dest_size),    // number of output compunents
                        gpu_context_entry, // context
                        inp_output         // input/output inplace
                    );

                    // Set the scale
                    encrypted.scale() = new_scale;
                }
            }

            /**
            Applies a Galois automorphism to a ciphertext and updates the encrypted parameter. To evaluate
            the Galois automorphism, an appropriate set of Galois keys must also be provided.
            Galois keys cached in GPU memor for a future reuse.

            The desired Galois automorphism is given as a Galois element, and must be an odd integer in the interval
            [1, M-1], where M = 2*N, and N = poly_modulus_degree. Used with batching, a Galois element 3^i % M
            corresponds to a cyclic row rotation i steps to the left, and a Galois element 3^(N/2-i) % M corresponds to
            a cyclic row rotation i steps to the right. The Galois element M-1 corresponds to a column rotation (row
            swap) in BFV, and complex conjugation in CKKS. In the polynomial view (not batching), a Galois automorphism
            by a Galois element p changes Enc(plain(x)) to Enc(plain(x^p)).

            @param[in/out] encrypted The ciphertext to apply the Galois automorphism to
            @param[in] galois_elt The Galois element
            @param[in] galois_keys The Galois keys

            */
            void apply_galois_inplace(
                seal::Ciphertext &encrypted, uint32_t galois_elt, const seal::GaloisKeys &galois_keys)
            {
                //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();
                    encrypted.gpu();
                    // Verify parameters.
                    if (!is_metadata_valid_for(encrypted, (*get_ctx())) || !is_buffer_valid(encrypted))
                    {
                        throw invalid_argument("encrypted is not valid for encryption parameters");
                    }

                    // Don't validate all of galois_keys but just check the parms_id.
                    if (galois_keys.parms_id() != (*get_ctx()).key_parms_id())
                    {
                        throw invalid_argument("galois_keys is not valid for encryption parameters");
                    }

                    // Use key_context_data where permutation tables exist since previous runs.
                    auto galois_tool = (*get_ctx()).key_context_data()->galois_tool();

                    xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                    auto output = cypherdata_to_gpu(encrypted);
                    auto encrypted_size = gpu_ctx_params(gpu_context_entry, encrypted);

                    // Size check
                    if (!seal::util::product_fits_in(gpu_context_entry.coeff_count, gpu_context_entry.rns_base_size))
                    {
                        throw logic_error("invalid parameters");
                    }

                    // Check if Galois key is generated or not.
                    if (!galois_keys.has_key(galois_elt))
                    {
                        throw invalid_argument("Galois key not present");
                    }

                    uint64_t m =
                        seal::util::mul_safe(static_cast<uint64_t>(gpu_context_entry.coeff_count), uint64_t(2));

                    // Verify parameters
                    if (!(galois_elt & 1) || seal::util::unsigned_geq(galois_elt, m))
                    {
                        throw invalid_argument("Galois element is not valid");
                    }
                    if (encrypted_size > 2)
                    {
                        throw invalid_argument("encrypted size must be 2");
                    }

                    auto poly_size = gpu_context_entry.coeff_count * gpu_context_entry.rns_base_size;
                    // temp space to process down the road

                    //static 
                    std::shared_ptr< xehe::ext::Buffer<T>> temp;
                    //if ( temp == nullptr || temp->get_size() < (2 * poly_size))
                    {
                        temp = xehe::ext::XeHE_malloc<T>((2 * poly_size));
                    }

                    // Perform permutation.

                    uint32_t index = uint32_t(galois_tool->GetIndexFromElt(galois_elt));
                    xehe::ext::XeHE_permute(
                        uint32_t(encrypted_size), uint32_t(gpu_context_entry.rns_base_size),
                        uint32_t(gpu_context_entry.coeff_count), galois_elt, index, output, temp);

                    // TODO: MOVE TO GPU
                    // move permuted into encrypted.data(0)
                    output->deep_copy(temp, poly_size);
                    // Wipe encrypted.data(1)
                    output->set_value(0, poly_size, poly_size);

                    // input only temp[0] component; temp[1] will be used inside routine as a temp space

                    temp->deep_copy(temp, poly_size, 0, poly_size);
                    // Calculate (temp * galois_key[0], temp * galois_key[1]) + (ct[0], 0)
                    switch_key_inplace_ckks(
                        encrypted, output, temp, static_cast<const seal::KSwitchKeys &>(galois_keys),
                        seal::GaloisKeys::get_index(galois_elt),
                        false); // galois keys
                }
            }

            void ntt(seal::Ciphertext &encrypted){

                //if (!encrypted.on_gpu())
                {
                    encrypted.gpu();
                }

                params_to_gpu();

                // Extract encryption parameters.
                xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                std::shared_ptr<xehe::ext::Buffer<T>> poly = cypherdata_to_gpu(encrypted);
                gpu_ctx_params(gpu_context_entry, encrypted);
                xehe::ext::XeHE_NTT<T>(1, gpu_context_entry,
                                    gpu_context_entry.xe_modulus,
                                    gpu_context_entry.xe_prim_roots_op,
                                    gpu_context_entry.xe_prim_roots_quo,
                                    poly);
            }

            void inverse_ntt(seal::Ciphertext &encrypted){

                //if (!encrypted.on_gpu())
                {
                    encrypted.gpu();
                }

                params_to_gpu();

                // Extract encryption parameters.
                xehe::ext::XeHE_mem_context<T> gpu_context_entry;
                std::shared_ptr<xehe::ext::Buffer<T>> poly = cypherdata_to_gpu(encrypted);
                gpu_ctx_params(gpu_context_entry, encrypted);
                xehe::ext::XeHE_invNTT<T>(1, gpu_context_entry,
                                    gpu_context_entry.xe_modulus,
                                    gpu_context_entry.xe_inv_prim_roots_op,
                                    gpu_context_entry.xe_inv_prim_roots_quo,
                                    poly,
                                    gpu_context_entry.xe_inv_degree_op,
                                    gpu_context_entry.xe_inv_degree_quo);
            }

        protected:
            /*

            A single point of the host to gpu ciphertext upload

            on GPU/Dirty bits logic:

            the only CT (ciphertext) objects we care about are those reaching GPU OPS.
            Meaning: only CKKS, and GPU_mem == gpu(), no BFV, no Keys' CT - we do not care about them at all (see copy
            constructor logic)

            1. OPs
                before any op begins, all operands and destination have their GPU bit set.
                no BFV, nor Key CT may appear at this point

                if CT does not have yet GPU memory, allocate it.
                if it's new or its' old but Dirty upload Host buffer to GPU memory.

                if ((new = !ct.GPU_mem)) ct.GPU_mem = alloc;
                if (new || (ct.GPU_mem && ct.Dirty)) upload(ct.Host_mem, ct.GPU_mem); ct.Dirty = false

            2. Copy constructor, = OP

                src is on GPU than tgt will be on GPU as well
                if (src.on_gpu ()) tgt.gpu = true

                 if src has GPU memory data copy it to tgt
                 but not if src CT is Dirty

                if (src.GPU_mem && !src.Dirty)
                    tgt.GPU_mem = alloc; copy(src.GPU_mem. tgt.GPU_mem)

                copy host mem if not on GPU or Dirty
                if Dirty, tgt will be treated as a new
                and GPU memory will be allocated and uploaded only if CT reaches any GPU op (see 1.)
                else
                    copy(src.Host_mem. tgt.Host_mem)

            3. Encryptor
                ONLY Encryptor sets Dirty flag!!
                if (on_gpu()) Dirty = true;

            */

            std::shared_ptr<xehe::ext::Buffer<T>> cypherdata_to_gpu(
                const seal::Ciphertext &encrypted,
                const std::shared_ptr<xehe::ext::Buffer<T>> alias = nullptr, // possible allias ptr
                bool allow_aliasing = false,                                 // is it OK to alias?
                bool output_only = false, // serve as output onlt, do not need to be sync with host
                size_t new_data_size = 0) // new size for the GPU buffer
            {
                std::shared_ptr<xehe::ext::Buffer<T>> encrypted_gpu_mem = nullptr;
                //if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    //params_to_gpu();

                    // get current ptr
                    auto encrypted_gpu_mem_old = get_ciphertext_gpu_mem<T>((seal::Ciphertext &)encrypted);
                    // get host data size
                    auto data_size = encrypted.dyn_array().size();

                    // atomic upload
                    std::lock_guard<std::mutex> lk(gpu_cypher_upload_lock_);


                    // is new data
                    bool new_data = (encrypted_gpu_mem_old == nullptr && (data_size > 0 || new_data_size > 0));
                    // does the buffer need a resize
                    bool resize =
                        (encrypted_gpu_mem_old != nullptr && new_data_size > 0 &&
                         new_data_size != encrypted_gpu_mem_old->get_size());
                    // make sure output available and does not alias
                    bool not_alias =
                        (encrypted_gpu_mem_old != nullptr && alias != nullptr &&
                         alias->is_alias(encrypted_gpu_mem_old) && !allow_aliasing);
                    // need allocation and/or upload
                    if ((new_data || not_alias || resize))
                    {
                        // aloocation size calculation
                        auto alloc_sz = (resize) ? new_data_size : ((data_size > 0) ? data_size : new_data_size);

                        encrypted_gpu_mem = xehe::ext::XeHE_malloc<T>(alloc_sz);

                        // if need a sync
                        // not destructed and need to move new data
                        if (!output_only || not_alias)
                        {
                            encrypted_gpu_mem->template set_data_adapter<uint64_t>(encrypted.data());
                            // uncoditionally
                            xehe::plgin::set_ciphertext_host_mem_dirty<T>((seal::Ciphertext &)encrypted, false);
                        }
                        else if (resize)
                        {
                            // if resize copy from old to new
                            encrypted_gpu_mem->deep_copy(encrypted_gpu_mem_old);
                        }
                        xehe::plgin::set_ciphertext_gpu_mem<T>((seal::Ciphertext &)encrypted, encrypted_gpu_mem);
                    }
                    else
                    {
                        // keep the old
                        encrypted_gpu_mem = encrypted_gpu_mem_old;
                    }

                    // still dirty?
                    bool dirty = xehe::plgin::is_ciphertext_host_mem_dirty<T>(encrypted);
                    if (dirty)
                    {
                        encrypted_gpu_mem->template set_data_adapter<uint64_t>(encrypted.data());
                        xehe::plgin::set_ciphertext_host_mem_dirty<T>((seal::Ciphertext &)encrypted, false);
                    }
                }
                return (encrypted_gpu_mem);
            }

            size_t gpu_ctx_params(xehe::ext::XeHE_mem_context<T> &context_mod, const seal::Ciphertext &encrypted)
            {
                size_t enc_sz = 0;
                if (encrypted.on_gpu())
                {
                    // upload static precumputed context data
                    params_to_gpu();
                    enc_sz = encrypted.size();
                    auto context_data_ptr = ((*get_ctx()).get_context_data(encrypted.parms_id()));
                    if (context_data_ptr)
                    {
                        context_mod = get_gpu_map()[(*context_data_ptr).parms_id()];
                    }
                }
                return (enc_sz);
            }

            // extract parameters from SEAL, 2 encrypted inputs, 1 destination
            // allocate and copy modulus, inv_modulus
            // resize destination
            size_t cypherdata_to_gpu3(
                xehe::ext::XeHE_mem_context<T> &context_mod,
                size_t &enc1_sz,   // n polys in input 1
                size_t &enc2_sz,   // n polys in input 2
                double &new_scale, // new CKKS scale = input1.scale*input2.scale
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem1,
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem2,
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem3, const seal::Ciphertext &encrypted1,
                const seal::Ciphertext &encrypted2,
                seal::Ciphertext &destination, // resized with max_sz
                bool allow_aliasing = false, bool output_only = true, bool add_op = true /* otherwise == multiply*/)
            {
                // upload static precumputed context data
                //params_to_gpu();

                encrypted_gpu_mem1 = cypherdata_to_gpu(encrypted1);
                encrypted_gpu_mem2 = cypherdata_to_gpu(encrypted2, encrypted_gpu_mem1, allow_aliasing);

                enc1_sz = gpu_ctx_params(context_mod, encrypted1);

                {
                    xehe::ext::XeHE_mem_context<T> temp_context_mod;
                    enc2_sz = gpu_ctx_params(temp_context_mod, encrypted2);
                }
                encrypted_gpu_mem3 = cypherdata_to_gpu(destination, encrypted_gpu_mem1, allow_aliasing, output_only);

                // ckks
                auto &ctx = *get_ctx();
                auto context_data_ptr = (ctx.get_context_data(encrypted1.parms_id()));
                size_t max_sz = 0; // (add_op==true) ? max(enc1_sz, enc2_sz) : enc1_sz + enc2_sz -1
                if (context_data_ptr)
                {
                    auto &context_data = *context_data_ptr;
                    new_scale = !(add_op) ? encrypted1.scale() * encrypted2.scale() : std::max(encrypted1.scale(), encrypted2.scale());

                    max_sz = (add_op) ? std::max(enc1_sz, enc2_sz) : (enc1_sz + enc2_sz - 1);

                    auto scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
                    if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >= scale_bit_count_bound))
                    {
                        std::cout << "Error: scale out of bounds" << std::endl;
                    }

                    // Size check
                    if (!seal::util::product_fits_in(max_sz, context_mod.coeff_count, context_mod.rns_base_size))
                    {
                        throw logic_error("invalid parameters");
                    }

                    if (max_sz > destination.size())
                    {
                        bool discard_alloc = output_only;
                        destination.resize(ctx, context_data.parms_id(), max_sz, discard_alloc);
                        encrypted_gpu_mem3 = cypherdata_to_gpu(destination, encrypted_gpu_mem1, false, output_only);
                    }
                }
                return max_sz;
            }

            size_t cypherdata_to_gpu_mad(
                xehe::ext::XeHE_mem_context<T> &context_mod,
                size_t &enc1_sz,   // n polys in input 1
                size_t &enc2_sz,   // n polys in input 2
                size_t &enc_add_sz,   // n polys in input 3
                double &new_scale, // new CKKS scale = input1.scale*input2.scale
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem1,
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem2,
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem_add,
                std::shared_ptr<xehe::ext::Buffer<T>> &encrypted_gpu_mem_res,
                const seal::Ciphertext &encrypted1,
                const seal::Ciphertext &encrypted2,
                const seal::Ciphertext &encrypted_add,
                seal::Ciphertext &destination, // resized with max_sz
                bool allow_aliasing = false, bool output_only = true)
            {
                // upload static precumputed context data
                //params_to_gpu();

                encrypted_gpu_mem_add = cypherdata_to_gpu(encrypted_add);

                encrypted_gpu_mem1 = cypherdata_to_gpu(encrypted1, encrypted_gpu_mem_add, allow_aliasing);
                encrypted_gpu_mem2 = cypherdata_to_gpu(encrypted2, encrypted_gpu_mem_add, allow_aliasing);

                enc1_sz = gpu_ctx_params(context_mod, encrypted1);

                {
                    xehe::ext::XeHE_mem_context<T> temp_context_mod;
                    enc2_sz = gpu_ctx_params(temp_context_mod, encrypted2);
                }
                {
                    xehe::ext::XeHE_mem_context<T> temp_context_mod;
                    enc_add_sz = gpu_ctx_params(temp_context_mod, encrypted_add);
                }

                encrypted_gpu_mem_res = cypherdata_to_gpu(destination, encrypted_gpu_mem_add, allow_aliasing, output_only);

                // ckks
                auto &ctx = *get_ctx();
                auto context_data_ptr = (ctx.get_context_data(encrypted_add.parms_id()));
                size_t max_sz = 0; // (add_op==true) ? max(enc1_sz, enc2_sz) : enc1_sz + enc2_sz -1
                if (context_data_ptr)
                {
                    auto &context_data = *context_data_ptr;

                    new_scale = std::max(encrypted1.scale() * encrypted2.scale(), encrypted_add.scale());

                    max_sz = std::max(enc_add_sz, enc1_sz + enc2_sz - 1);

                    auto scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
                    if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >= scale_bit_count_bound))
                    {
                        std::cout << "Error: scale out of bounds" << std::endl;
                    }

                    // Size check
                    if (!seal::util::product_fits_in(max_sz, context_mod.coeff_count, context_mod.rns_base_size))
                    {
                        throw logic_error("invalid parameters");
                    }

                    if (max_sz > destination.size())
                    {
                        bool discard_alloc = output_only;
                        destination.resize(ctx, context_data.parms_id(), max_sz, discard_alloc);
                        encrypted_gpu_mem_res = cypherdata_to_gpu(destination, encrypted_gpu_mem_add, false, output_only);
                    }
                }
                return max_sz;
            }

            void extract_key_params(xehe::ext::XeHE_mem_context<T> &context_mod)
            {
                // upload static precumputed context data
                params_to_gpu();

                auto &ctx = *get_ctx();
                // the last level
                auto &context_data = *(ctx.key_context_data());
                //auto &parms = context_data.parms();
                context_mod = get_gpu_map()[context_data.parms_id()];
            }

            const seal::SEALContext *get_ctx(void) const
            {
                return seal_ctx_;
            }

            std::unordered_map<seal::parms_id_type, xehe::ext::XeHE_mem_context<T>> &get_gpu_map(void)
            {
                return (context_mod_data_map_);
            }

            xehe::ext::XeHE_mem_keys<T> &get_gpu_relin_keys(void)
            {
                return (context_relin_keys_data_);
            }

            std::vector<xehe::ext::XeHE_mem_keys<T>> &get_gpu_galois_keys(void)
            {
                return (context_galois_keys_data_);
            }

        private:
            const seal::SEALContext *seal_ctx_ = nullptr;
            // array of precomputed modulus related data
            std::unordered_map<seal::parms_id_type, xehe::ext::XeHE_mem_context<T>> context_mod_data_map_;
            // precomputed reliarization keys
            xehe::ext::XeHE_mem_keys<T> context_relin_keys_data_;
            // array of cached Galois key buffer
            std::vector<xehe::ext::XeHE_mem_keys<T>> context_galois_keys_data_;

            bool ckks_scheme_ = true;

            std::mutex gpu_invdata_lock_;
            std::mutex gpu_plain_upload_lock_;
            std::mutex gpu_cypher_upload_lock_;
            std::mutex gpu_relin_lock_;
            std::mutex gpu_galois_lock_;
        };

    } // namespace plgin

} // namespace xehe

#endif // XEHE_SEAL_PLGIN_HPP

#endif //#ifdef SEAL_USE_INTEL_XEHE
