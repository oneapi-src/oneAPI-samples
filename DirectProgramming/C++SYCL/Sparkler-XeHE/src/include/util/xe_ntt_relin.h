/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef _XE_NTT_RELIN_H_
#define _XE_NTT_RELIN_H_

#include "util/rns_dwt_relin_gpu.hpp"
#include "util/rns_invdwt_relin_gpu.hpp"
#include <stdexcept>

namespace xehe
{
    namespace util
    {
        // ntt interface for relin -- CPU
        template <typename T>
        void ntt_negacyclic_harvey(int key_modulus_size, int poly_num, int q_base_size, int log_n, T* values, const T * modulus,
                                   const T* roots_op, const T* roots_quo,
                                   const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                   bool lazy = false, bool wait = false)
        {

            RnsDwtRelin<T>(key_modulus_size, poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)();

        }

        // intt interface for relin -- CPU
        template <typename T>
        void inverse_ntt_negacyclic_harvey(int key_modulus_size, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                           bool lazy = false, bool wait = false)
        {
            InvRnsDwtRelin<T>(key_modulus_size, poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)();
        }


#ifdef BUILD_WITH_IGPU
        // relin interface for ntt
        template <typename T>
        void ntt_negacyclic_harvey(cl::sycl::queue& q,int key_modulus_size, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                   const T* roots_op, const T* roots_quo,
                                   const T* scalar_op = nullptr, const T* scalar_quo = nullptr, bool lazy = false)
        {
            RnsDwtGpuRelin<T>(key_modulus_size, poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)(q);
        }

        // relin inverse NTT interface on GPU
        template <typename T>
        void inverse_ntt_negacyclic_harvey(cl::sycl::queue& q, int key_modulus_size, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr, bool lazy = false)
        {
            InvRnsDwtGpuRelin<T>(key_modulus_size, poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)(q);
        }

#endif

    } // namespace util
} // namespace xehe

#endif //#ifndef _XE_NTT_RELIN_H_