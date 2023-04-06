/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef _XE_NTT_H_
#define _XE_NTT_H_

#include "util/rns_dwt_gpu.hpp"
#include "util/rns_invdwt_gpu.hpp"
#include <stdexcept>

namespace xehe
{
    namespace util
    {
        // CPU NTT
        template <typename T>
        void ntt_negacyclic_harvey(int poly_num, int q_base_size, int log_n, T* values, const T * modulus,
                                   const T* roots_op, const T* roots_quo,
                                   const T* scalar_op = nullptr, const T* scalar_quo = nullptr, bool lazy = false)
        {

            RnsDwt<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)();

        }

        // CPU inverse NTT
        template <typename T>
        void inverse_ntt_negacyclic_harvey(int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr, bool lazy = false)
        {
            InvRnsDwt<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)();
        }


#ifdef BUILD_WITH_IGPU


        // old interface without poly_num
        template <typename T>
        void ntt_negacyclic_harvey(cl::sycl::queue& q, int q_base_size, int log_n, T* values, const T* modulus,
                                   const T* roots_op, const T* roots_quo,
                                   const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                   bool lazy = false, bool wait = false)
        {
            RnsDwtGpu<T>(1, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)(q);
        }

        // new interface with poly_num
        template <typename T>
        void ntt_negacyclic_harvey(cl::sycl::queue& q, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                   const T* roots_op, const T* roots_quo,
                                   const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                   bool lazy = false, bool wait = false)
        {
            RnsDwtGpu<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)(q);
        }

        // old interface without poly_num
        template <typename T>
        void inverse_ntt_negacyclic_harvey(cl::sycl::queue& q, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                           bool lazy = false, bool wait = false)
        {
            InvRnsDwtGpu<T>(1, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)(q);
        }

        // new interface with poly_num
        template <typename T>
        void inverse_ntt_negacyclic_harvey(cl::sycl::queue& q, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                           bool lazy = false, bool wait = false)
        {
            InvRnsDwtGpu<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)(q);
        }

        // rescaling
        // template <typename T>
        // void inverse_ntt_negacyclic_harvey(cl::sycl::queue& q, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
        //                                    const T* roots_op, const T* roots_quo,
        //                                    const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
        //                                    bool lazy = false, bool wait = false)
        // {
        //     InvRnsDwtGpu<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)(q);
        // }

#endif

    } // namespace util
} // namespace xehe

#endif //#ifndef _XE_NTT_H_