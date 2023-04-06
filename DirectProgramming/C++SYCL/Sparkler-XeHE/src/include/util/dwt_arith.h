// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef _DWT_ARITH_H_
#define _DWT_ARITH_H_



#include "native/xe_uintarith_core.hpp"
#include <stdexcept>

namespace xehe
{
    namespace util
    {

        template <typename T>
        inline T dwt_add(const T& a, const T& b)
        {
            return a + b;
        }

        template <typename T>
        inline T dwt_sub(const T& a, const T& b, const T& two_times_modulus)
        {

            return a + two_times_modulus - b;
        }

        template <typename T>
        inline T dwt_mul_root(const T& a, const T& r_op, const T& r_quo, const T& modulus)
        {
            return native::mul_uint_mod_lazy<T>(a, r_op, modulus, r_quo);
        }

        template <typename T>
        inline T dwt_mul_scalar(const T& a, const T& s_op, const T& s_quo, const T& modulus)
        {
            return native::mul_uint_mod_lazy<T>(a, s_op, modulus, s_quo);
        }

        template <typename T>
        inline void dwt_mul_root_scalar(
            const T& r_op, const T& s_op, const T& s_quo, const T& modulus, T& result_op, T& result_quo)
        {
            native::set_operand<T>(native::mul_uint_mod_lazy<T>(r_op, s_op, modulus, s_quo), modulus, result_op, result_quo);
        }

        template <typename T>
        inline T dwt_guard(const std::uint64_t& a, const T& two_times_modulus)
        {
            return a - (two_times_modulus &
                static_cast<std::uint64_t>(-static_cast<T>(a >= two_times_modulus)));
        }


    } // namespace util
} // namespace xehe


#endif //#ifndef _DWT_ARITH_H_