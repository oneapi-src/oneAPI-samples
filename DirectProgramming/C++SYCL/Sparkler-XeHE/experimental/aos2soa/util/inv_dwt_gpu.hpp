/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#pragma once
#ifdef BUILD_WITH_IGPU

#include "dpcpp_utils.h"
#include <CL/sycl.hpp>


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



        template <typename ValueType, typename RootType>
        class InvDwtGapLE4 {
        public:
            InvDwtGapLE4(ValueType* values, int log_n, std::size_t gap, int rounds, ValueType modulus, const RootType* roots)
            {
                //round_ = round;
                values_ = values;
                log_n_ = log_n;
                rounds_ = rounds;
                gap_ = gap;
                modulus_ = modulus;
                two_times_modulus_ = (modulus << 1);
                roots_ = roots;
            }

            void operator()(cl::sycl::id<1> ind) const {

                auto i = ind / gap_;
                auto r = roots_[rounds_ + i];
                auto j = ind % gap_;

                std::size_t offset = i * (gap_ << 1) + j;
                auto x = values_ + offset;
                auto y = x + gap_;
                auto u = *x;
                auto v = *y;
                *x = dwt_guard(dwt_add(u, v), two_times_modulus_);
                *y = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);
                
            }

        protected:
            //int round_;
            ValueType* values_;
            int log_n_;
            int rounds_;
            std::size_t gap_;
            ValueType modulus_;
            ValueType two_times_modulus_;
            const RootType* roots_;

        };

        template <typename ValueType, typename RootType, int Unroll>
        class InvDwtLargeGap {
        public:
            InvDwtLargeGap(ValueType* values, int log_n, std::size_t gap, int rounds, ValueType modulus, const RootType* roots)
            {
                //round_ = round;
                values_ = values;
                log_n_ = log_n;
                rounds_ = rounds;
                gap_ = gap;
                modulus_ = modulus;
                two_times_modulus_ = (modulus << 1);
                roots_ = roots;
            }

            void operator()(cl::sycl::id<1> ind) const {

                auto i = ind / (gap_ / Unroll);
                auto j = ind % (gap_ / Unroll);

                auto r = roots_[rounds_ + i];
                std::size_t offset = i * (gap_ << 1) + j * Unroll;

                auto x = values_ + offset;
                auto y = x + gap_;
                for (std::size_t k = 0; k < Unroll; ++k)
                {
                    auto u = *x;
                    auto v = *y;
                    *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                    *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);
                }
            }

        protected:
            //int round_;
            ValueType* values_;
            int log_n_;
            int rounds_;
            std::size_t gap_;
            ValueType modulus_;
            ValueType two_times_modulus_;
            const RootType* roots_;

        };

        template <typename ValueType, typename RootType, typename ScalarType>
        class InvDwtLastRoundScalar {
        public:
            InvDwtLastRoundScalar(ValueType* values, int gap, int rounds, ValueType modulus, const RootType* roots, const ScalarType* scalar)
            {
                values_ = values;
                modulus_ = modulus;
                two_times_modulus_ = (modulus << 1);
                gap_ = gap;
                rounds_ = rounds;
                roots_ = roots;
                scalar_ = scalar;

            }
            void operator()(cl::sycl::id<1> ind) const {
                auto r = roots_[rounds_];
                RootType scaled_r = dwt_mul_root_scalar(r, *scalar_, modulus_);

                auto x = values_ + ind;
                auto y = x + gap_;

                auto u = dwt_guard(*x, two_times_modulus_);
                auto v = *y;
                auto uu = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus_), *scalar_, modulus_);
                auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), scaled_r, modulus_);
                uu -= (uu >= modulus_) ? modulus_ : 0;
                vv -= (vv >= modulus_) ? modulus_ : 0;

                *x = uu;
                *y = vv;
            }
        protected:
            ValueType* values_;
            ValueType modulus_;
            ValueType two_times_modulus_;
            int gap_;
            int rounds_;
            const RootType* roots_;
            const ScalarType* scalar_;


        };

        template <typename ValueType, typename RootType>
        class InvDwtLastRound{
        public:
            InvDwtLastRound(ValueType* values, int gap, int rounds, ValueType modulus, const RootType* roots)
            {
                values_ = values;
                modulus_ = modulus;
                two_times_modulus_ = (modulus << 1);
                gap_ = gap;
                rounds_ = rounds;
                roots_ = roots;

            }
            void operator()(cl::sycl::id<1> ind) const {
                auto r = roots_[rounds_];
                auto x = values_ + ind;
                auto y = x + gap_;
                auto u = *x;
                auto v = *y;
                auto uu = dwt_guard(dwt_add(u, v), two_times_modulus_);
                auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);
                uu -= (uu >= modulus_) ? modulus_ : 0;
                vv -= (vv >= modulus_) ? modulus_ : 0;

                *x = uu;
                *y = vv;
            }
        protected:
            ValueType* values_;
            ValueType modulus_;
            ValueType two_times_modulus_;
            int gap_;
            int rounds_;
            const RootType* roots_;


        };


        /**
        Performs in place a fast multiplication with the DWT matrix.
        Accesses to powers of root is coalesced.
        Accesses to values is not coalesced without loop unrolling.

        @param[values] inputs in bit-reversed order, outputs in normal order
        @param[roots] powers of a root in scrambled order
        @param[scalar] an optional scalar that is multiplied to all output values
        */
        template <typename ValueType, typename RootType, typename ScalarType>
        class InvDwt_gpu {
        public:
            InvDwt_gpu(cl::sycl::queue q, ValueType* values, int log_n, ValueType modulus, const RootType* roots, const ScalarType* scalar = nullptr)
            {
                values_ = values;
                log_n_ = log_n;
                modulus_ = modulus;
                two_times_modulus_ = (modulus << 1);
                roots_ = roots;
                scalar_ = scalar;
                queue_ = q;

            }

            void operator() (void) {
                // constant transform size
                size_t n = size_t(1) << log_n_;
                // variables for indexing
                std::size_t gap = 1;
                std::size_t m = n >> 1;
                std::size_t total_r = 1;
                for (; m > 1; total_r += m, m >>= 1, gap <<= 1)
                {
                    if (gap < 4)
                    {
#if 1
                        queue_.submit([&](cl::sycl::handler& h) {
                            h.parallel_for((m * gap), InvDwtGapLE4<ValueType, RootType>(values_, log_n_, gap, total_r, modulus_, roots_));
                            }).wait();
#else
                        for (std::size_t ind = 0; ind < m * gap; ind++)
                        {
                            auto i = ind / gap;
                            auto r = roots_[total_r + i];
                            auto j = ind % gap;

                            std::size_t offset = i * (gap << 1) + j;
                            auto x = values_ + offset;
                            auto y = x + gap;
                            auto u = *x;
                            auto v = *y;
                            *x = dwt_guard(dwt_add(u, v), two_times_modulus_);
                            *y = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);


                        }
#endif
                    }
                    else
                    {
                        static const int unroll = 4;

#if 1
                        queue_.submit([&](cl::sycl::handler& h) {
                            h.parallel_for((m * gap / unroll), InvDwtLargeGap<ValueType, RootType, unroll>(values_, log_n_, gap, total_r, modulus_, roots_));
                            }).wait();
#else

                        for (std::size_t ind = 0; ind < m*gap/unroll; ind++)
                        {
                            auto i = ind / (gap / unroll);
                            auto j = ind % (gap / unroll);

                            auto r = roots_[total_r + i];
                            std::size_t offset = i * (gap << 1) + j*unroll;

                            auto x = values_ + offset;
                            auto y = x + gap;
                            //for (std::size_t j = 0; j < gap; j += 4)
                            {
                                auto u = *x;
                                auto v = *y;
                                *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                                *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);

                                u = *x;
                                v = *y;
                                *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                                *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);

                                u = *x;
                                v = *y;
                                *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                                *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);

                                u = *x;
                                v = *y;
                                *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                                *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);
                            }

                        }
#endif
                    }
                }

                if (scalar_ != nullptr)
                {
#if 1
                    queue_.submit([&](cl::sycl::handler& h) {
                        h.parallel_for((gap), InvDwtLastRoundScalar<ValueType, RootType, ScalarType>(values_, gap, total_r, modulus_, roots_, scalar_));
                        }).wait();

#else
                    if (gap < 4)
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            auto u = dwt_guard(*x, two_times_modulus_);
                            auto v = *y;
                            *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus_), *scalar_, modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), scaled_r, modulus_);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            auto u = dwt_guard(*x, two_times_modulus_);
                            auto v = *y;
                            *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus_), *scalar_, modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), scaled_r, modulus_);

                            u = dwt_guard(*x, two_times_modulus_);
                            v = *y;
                            *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus_), *scalar_, modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), scaled_r, modulus_);

                            u = dwt_guard(*x, two_times_modulus_);
                            v = *y;
                            *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus_), *scalar_, modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), scaled_r, modulus_);

                            u = dwt_guard(*x, two_times_modulus_);
                            v = *y;
                            *x++ = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus_), *scalar_, modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), scaled_r, modulus_);
                        }
                    }
#endif
                }
                else
                {



#if 1
                    queue_.submit([&](cl::sycl::handler& h) {
                        h.parallel_for((gap), InvDwtLastRound<ValueType, RootType>(values_, gap, total_r, modulus_, roots_));
                        }).wait();
#else
                    auto r = roots_[total_r];
                    auto x = values_;
                    auto y = x + gap;
                    if (gap < 4)
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            auto u = *x;
                            auto v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            auto u = *x;
                            auto v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);

                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);

                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);

                            u = *x;
                            v = *y;
                            *x++ = dwt_guard(dwt_add(u, v), two_times_modulus_);
                            *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus_), r, modulus_);
                        }
                    }
#endif
                }
            }

        protected:

            cl::sycl::queue queue_;
            ValueType* values_;
            int log_n_;
            ValueType modulus_;
            ValueType two_times_modulus_;
            const RootType* roots_;
            const ScalarType* scalar_;

        };

    } // namespace util
} // namespace xehe


#endif //#ifdef BUILD_WITH_IGPU
