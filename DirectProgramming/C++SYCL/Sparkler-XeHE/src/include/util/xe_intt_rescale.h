/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef _XE_INTT_RESCALE_H_
#define _XE_INTT_RESCALE_H_

#include "util/rns_invdwt_rescale_gpu.hpp"
#include <stdexcept>

namespace xehe
{
    namespace util
    {
        // rescaling CPU version
        template <typename T>
        void inverse_ntt_negacyclic_harvey_rescale(int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                           bool lazy = false, bool wait = false)
        {
            InvRnsDwtRescale<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)();
        }

#ifdef BUILD_WITH_IGPU


        // rescaling GPU version
        template <typename T>
        void inverse_ntt_negacyclic_harvey_rescale(cl::sycl::queue& q, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                                           const T* roots_op, const T* roots_quo,
                                           const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                                           bool lazy = false, bool wait = false)
        {
            InvRnsDwtGpuRescale<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy, wait)(q);
        }

#endif

    } // namespace util
} // namespace xehe

#endif //#ifndef _XE_INTT_RESCALE_H_