/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#pragma once

#include "util/uintarithsmallmod.h"
#include "util/xe_uintcore.h"
#include "util/dwt_arith.h"
#include <stdexcept>

namespace xehe
{
    namespace util
    {

        /**
        Provides an interface that performs the fast discrete weighted transform (DWT) and its inverse that are used to
        accelerate polynomial multiplications, batch multiple messages into a single plaintext polynomial. This class
        template is specialized with integer modular arithmetic for DWT over integer quotient rings, and is used in
        polynomial multiplications and BatchEncoder. It is also specialized with double-precision complex arithmetic for
        DWT over the complex field, which is used in CKKSEncoder.

        @par The discrete weighted transform (DWT) is a variantion on the discrete Fourier transform (DFT) over
        arbitrary rings involving weighing the input before transforming it by multiplying element-wise by a weight
        vector, then weighing the output by another vector. The DWT can be used to perform negacyclic convolution on
        vectors just like how the DFT can be used to perform cyclic convolution. The DFT of size n requires a primitive
        n-th root of unity, while the DWT for negacyclic convolution requires a primitive 2n-th root of unity, \psi.
        In the forward DWT, the input is multiplied element-wise with an incrementing power of \psi, the forward DFT
        transform uses the 2n-th primitve root of unity \psi^2, and the output is not weighed. In the backward DWT, the
        input is not weighed, the backward DFT transform uses the 2n-th primitve root of unity \psi^{-2}, and the output
        is multiplied element-wise with an incrementing power of \psi^{-1}.

        @par A fast Fourier transform is an algorithm that computes the DFT or its inverse. The Cooley-Tukey FFT reduces
        the complexity of the DFT from O(n^2) to O(n\log{n}). The DFT can be interpretted as evaluating an (n-1)-degree
        polynomial at incrementing powers of a primitive n-th root of unity, which can be accelerated by FFT algorithms.
        The DWT evaluates incrementing odd powers of a primitive 2n-th root of unity, and can also be accelerated by
        FFT-like algorithms implemented in this class.

        @par Algorithms implemented in this class are based on algorithms 1 and 2 in the paper by Patrick Longa and
        Michael Naehrig (https://eprint.iacr.org/2016/504.pdf) with three modifications. First, we generalize in this
        class the algorithms to DWT over arbitrary rings. Second, the powers of \psi^{-1} used by the IDWT are stored
        in a scrambled order (in contrast to bit-reversed order in paper) to create coalesced memory accesses. Third,
        the multiplication with 1/n in the IDWT is merged to the last iteration, saving n/2 multiplications. Last, we
        unroll the loops to create coalesced memory accesses to input and output vectors. In earlier versions of SEAL,
        the mutiplication with 1/n is done by merging a multiplication of 1/2 in all interations, which is slower than
        the current method on CPUs but more efficient on some hardware architectures.

        @par The order in which the powers of \psi^{-1} used by the IDWT are stored is unnatural but efficient:
        the i-th slot stores the (reverse_bits(i - 1, log_n) + 1)-th power of \psi^{-1}.
        */

        /**
        Performs in place a fast multiplication with the DWT matrix.
        Accesses to powers of root is coalesced.
        Accesses to values is not coalesced without loop unrolling.

        @param[values] inputs in normal order, outputs in bit-reversed order
        @param[log_n] log 2 of the DWT size
        @param[roots] powers of a root in bit-reversed order
        @param[scalar] an optional scalar that is multiplied to all output values
        */
        template <typename ValueType, typename RootType, typename ScalarType>
        void transform_to_rev(
            ValueType* values, int log_n, ValueType modulus, const RootType* roots, const ScalarType* scalar = nullptr)
        {
            // constant transform size
            size_t n = size_t(1) << log_n;
            auto two_times_modulus = (modulus << 1);
            // registers to hold temporary values
            RootType r;
            ValueType u;
            ValueType v;
            // pointers for faster indexing
            ValueType* x = nullptr;
            ValueType* y = nullptr;
            // variables for indexing
            std::size_t gap = n >> 1;
            std::size_t m = 1;

            for (; m < (n >> 1); m <<= 1)
            {
                std::size_t offset = 0;
                if (gap < 4)
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        x = values + offset;
                        y = x + gap;
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = dwt_guard(*x, two_times_modulus);
                            v = dwt_mul_root(*y, r, modulus);
                            *x++ = dwt_add(u, v);
                            *y++ = dwt_sub(u, v, two_times_modulus);
                        }
                        offset += gap << 1;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        x = values + offset;
                        y = x + gap;
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = dwt_guard(*x, two_times_modulus);
                            v = dwt_mul_root(*y, r, modulus);
                            *x++ = dwt_add(u, v);
                            *y++ = dwt_sub(u, v, two_times_modulus);

                            u = dwt_guard(*x, two_times_modulus);
                            v = dwt_mul_root(*y, r, modulus);
                            *x++ = dwt_add(u, v);
                            *y++ = dwt_sub(u, v, two_times_modulus);

                            u = dwt_guard(*x, two_times_modulus);
                            v = dwt_mul_root(*y, r, modulus);
                            *x++ = dwt_add(u, v);
                            *y++ = dwt_sub(u, v, two_times_modulus);

                            u = dwt_guard(*x, two_times_modulus);
                            v = dwt_mul_root(*y, r, modulus);
                            *x++ = dwt_add(u, v);
                            *y++ = dwt_sub(u, v, two_times_modulus);
                        }
                        offset += gap << 1;
                    }
                }
                gap >>= 1;
            }

            if (scalar != nullptr)
            {
                RootType scaled_r;
                for (std::size_t i = 0; i < m; i++)
                {
                    r = *++roots;
                    scaled_r = dwt_mul_root_scalar(r, *scalar, modulus);
                    u = dwt_mul_scalar(dwt_guard(values[0], two_times_modulus), *scalar, modulus);
                    v = dwt_mul_root(values[1], scaled_r, modulus);
                    values[0] = dwt_add(u, v);
                    values[1] = dwt_sub(u, v, two_times_modulus);
                    values += 2;
                }
            }
            else
            {
                for (std::size_t i = 0; i < m; i++)
                {
                    r = *++roots;
                    u = dwt_guard(values[0], two_times_modulus);
                    v = dwt_mul_root(values[1], r, modulus);
                    values[0] = dwt_add(u, v);
                    values[1] = dwt_sub(u, v, two_times_modulus);
                    values += 2;
                }
            }
        }

        /**
        Performs in place a fast multiplication with the DWT matrix.
        Accesses to powers of root is coalesced.
        Accesses to values is not coalesced without loop unrolling.

        @param[values] inputs in bit-reversed order, outputs in normal order
        @param[roots] powers of a root in scrambled order
        @param[scalar] an optional scalar that is multiplied to all output values
        */

        template <typename ValueType, typename RootType, typename ScalarType>
        void transform_from_rev(
            ValueType* values, int log_n, ValueType modulus, const RootType* roots, const ScalarType* scalar = nullptr)
        {
            // constant transform size
            size_t n = size_t(1) << log_n;
            auto two_times_modulus = (modulus << 1);
            // registers to hold temporary values
            RootType r;
            ValueType u;
            ValueType v;
            // pointers for faster indexing
            ValueType* x = nullptr;
            ValueType* y = nullptr;
            // variables for indexing
            std::size_t gap = 1;
            std::size_t m = n >> 1;

            for (; m > 1; m >>= 1)
            {
                std::size_t offset = 0;
                if (gap < 4)
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        x = values + offset;
                        y = x + gap;
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);
                        }
                        offset += gap << 1;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        x = values + offset;
                        y = x + gap;
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);

                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);

                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);

                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);
                        }
                        offset += gap << 1;
                    }
                }
                gap <<= 1;
            }

            if (scalar != nullptr)
            {
                r = *++roots;
                RootType scaled_r = dwt_mul_root_scalar(r, *scalar, modulus);
                x = values;
                y = x + gap;
                if (gap < 4)
                {
                    for (std::size_t j = 0; j < gap; j++)
                    {
                        u = dwt_guard(*x, two_times_modulus);
                        v = *y;
                        *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), *scalar, modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r, modulus);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < gap; j += 4)
                    {
                        u = dwt_guard(*x, two_times_modulus);
                        v = *y;
                        *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), *scalar, modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r, modulus);

                        u = dwt_guard(*x, two_times_modulus);
                        v = *y;
                        *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), *scalar, modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r, modulus);

                        u = dwt_guard(*x, two_times_modulus);
                        v = *y;
                        *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), *scalar, modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r, modulus);

                        u = dwt_guard(*x, two_times_modulus);
                        v = *y;
                        *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), *scalar, modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r, modulus);
                    }
                }
            }
            else
            {
                r = *++roots;
                x = values;
                y = x + gap;
                if (gap < 4)
                {
                    for (std::size_t j = 0; j < gap; j++)
                    {
                        u = *x;
                        v = *y;
                        *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < gap; j += 4)
                    {
                        u = *x;
                        v = *y;
                        *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);

                        u = *x;
                        v = *y;
                        *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);

                        u = *x;
                        v = *y;
                        *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);

                        u = *x;
                        v = *y;
                        *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r, modulus);
                    }
                }
            }
        }



    } // namespace util
} // namespace xehe
