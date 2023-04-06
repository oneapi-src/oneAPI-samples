/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/


#ifndef _RNS_INVDWT_RESCALE_GPU_HPP_
#define _RNS_INVDWT_RESCALE_GPU_HPP_

#include "util/dwt_arith.h"
#include <stdexcept>
#include "util/ntt_params.h"
#include "util/rns_dwt_radix_macros.h"

template <typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write, sycl::access::target::local>;

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
        // Normal RNS Inverse NTT kernel with generic unroll param
        template <typename T, int log_Unroll>
        class RnsInvDwtGapRescale {
        public:
            RnsInvDwtGapRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds,
                         T* values, const T* modulus,
                         const T* roots_op,
                         const T* roots_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                rounds_ = rounds;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap);
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }

            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind) const {

                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);

                auto i = (ind >> (log_gap_ - log_Unroll));
                auto j = ind - (i << (log_gap_ - log_Unroll));

                auto r_op = roots_op_[rounds_ + i];
                auto r_quo = roots_quo_[rounds_ + i];
                std::size_t offset = (i << (log_gap_ + 1)) + (j << log_Unroll);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                auto x = values_ + offset + poly_offset;
                auto y = x + gap_;
                for (std::size_t k = 0; k < (1 << log_Unroll); ++k)
                {
                    auto u = *x;
                    auto v = *y;
                    *x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                    *y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);
                }

            }

            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t poly_idx = ind[0];
                uint64_t i = ind[1];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int rounds_;
            std::size_t log_gap_;
            std::size_t gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;

        };

        template <typename T>
        class RnsInvDwtGapRadix2FusedRescale {
        public:
            RnsInvDwtGapRadix2FusedRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds,
                         T* values, const T* modulus,
                         const T* roots_op,
                         const T* roots_quo,
                         const T* scalar_op, const T* scalar_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                rounds_ = rounds;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap);
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
            }

            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind) const {

                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];

                auto i = (ind >> (log_gap_));
                auto j = ind - (i << (log_gap_));

                auto r_op = roots_op_[rounds_ + i];
                auto r_quo = roots_quo_[rounds_ + i];
                std::size_t offset = (i << (log_gap_ + 1)) + (j);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                auto x = values_ + offset + poly_offset;
                auto y = x + gap_;

                auto u = *x;
                auto v = *y;
                auto uu = dwt_guard(dwt_add(u, v), two_times_modulus);
                auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);

                uu = dwt_mul_scalar(dwt_guard(uu, two_times_modulus), scalar_op, scalar_quo, modulus);
                vv = dwt_mul_scalar(dwt_guard(vv, two_times_modulus), scalar_op, scalar_quo, modulus);

                uu -= (uu >= modulus) ? modulus : 0;
                vv -= (vv >= modulus) ? modulus : 0;

                *x = uu;
                *y = vv;
            }

            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t poly_idx = ind[0];
                uint64_t i = ind[1];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int rounds_;
            std::size_t log_gap_;
            std::size_t gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
        };

        template <typename T>
        class RnsInvDwtGapRadix4Rescale {
        public:
            RnsInvDwtGapRadix4Rescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind) const {
                auto modulus = modulus_[position_];
                auto m_ = (1<<log_m_);
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                T *x[4];
                T dx[4];
                T r_op[2], r_quo[2];
                T u, v;
                uint64_t ind1, ind2, i1, i2;
                ind1 = ((ind >> (log_gap_-1)) << log_gap_) + (ind & ((gap_>>1) - 1));
                ind2 = ind1 + (gap_>>1);

                compute_ind_2(log_gap_-1)
                auto j = ind1 & ((gap_>>1) - 1);
                auto offset = (i1 << ((log_gap_-1) + 1)) + j;

                ld_roots_2(roots_op_, r_op, (rounds_))
                ld_roots_2(roots_quo_, r_quo, (rounds_))

                init_global_ptr_4((gap_>>1))
                ld_global_mem_to_reg_4(0)

                // round 1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)

                // round 2
                compute_ind_2(log_gap_)
                ld_roots_2(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_2(roots_quo_, r_quo, (rounds_+ m_ ))

                butterfly_inv_ntt_reg2gmem(0, 2, 0)
                butterfly_inv_ntt_reg2gmem(1, 3, 1)
            }

            
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
        };

        template <typename T>
        class RnsInvDwtGapRadix4FusedRescale {
        public:
            RnsInvDwtGapRadix4FusedRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind) const {
                auto modulus = modulus_[position_];
                auto m_ = (1<<log_m_);
                auto two_times_modulus = (modulus << 1);
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                T *x[4];
                T dx[4];
                T r_op[2], r_quo[2];
                T u, v;
                uint64_t ind1, ind2, i1, i2;
                ind1 = ((ind >> (log_gap_-1)) << log_gap_) + (ind & ((gap_>>1) - 1));
                ind2 = ind1 + (gap_>>1);

                compute_ind_2(log_gap_-1)
                auto j = ind1 & ((gap_>>1) - 1);
                auto offset = (i1 << ((log_gap_-1) + 1)) + j;

                ld_roots_2(roots_op_, r_op, (rounds_))
                ld_roots_2(roots_quo_, r_quo, (rounds_))

                init_global_ptr_4((gap_>>1))
                ld_global_mem_to_reg_4(0)

                // round 1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)

                // round 2
                compute_ind_2(log_gap_)
                ld_roots_2(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_2(roots_quo_, r_quo, (rounds_+ m_ ))

                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)

                st_reg2gmem_scale_4(0)
            }

            
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
        };


        template <typename T>
        class RnsInvDwtGapRadix8Rescale {
        public:
            RnsInvDwtGapRadix8Rescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind) const {
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                auto m_ = (1 << log_m_);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                T *x[8];
                T dx[8];
                T r_op[4], r_quo[4];
                T u, v;
                uint64_t ind1, ind2, ind3, ind4, i1, i2, i3, i4;

                ind1 = ((ind >> (log_gap_-2)) << log_gap_) + (ind & ((gap_>>2) - 1));
                ind2 = ind1 + (gap_>>2);
                ind3 = ind2 + (gap_>>2);
                ind4 = ind3 + (gap_>>2);

                compute_ind_4(log_gap_-2)

                auto j = ind1 & ((gap_>>2) - 1);
                auto offset = (i1 << ((log_gap_-2) + 1)) + j;

                ld_roots_4(roots_op_, r_op, rounds_)
                ld_roots_4(roots_quo_, r_quo, rounds_)
                init_global_ptr_8((gap_>>2))
                ld_global_mem_to_reg_8(0)

                // round1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)


                // inner round 2
                compute_ind_4(log_gap_-1)
                ld_roots_4(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_4(roots_quo_, r_quo, (rounds_+ m_ ))
                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_)
                ld_roots_4(roots_op_, r_op, (rounds_+ m_+ (m_>>1) ))
                ld_roots_4(roots_quo_, r_quo, (rounds_+ m_ + (m_>>1) ))

                butterfly_inv_ntt_reg2gmem(0, 4, 0)
                butterfly_inv_ntt_reg2gmem(1, 5, 1)
                butterfly_inv_ntt_reg2gmem(2, 6, 2)
                butterfly_inv_ntt_reg2gmem(3, 7, 3)

            }

            
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
        };

        template <typename T>
        class RnsInvDwtGapRadix8FusedRescale {
        public:
            RnsInvDwtGapRadix8FusedRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind) const {
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                auto m_ = (1 << log_m_);
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                T *x[8];
                T dx[8];
                T r_op[4], r_quo[4];
                T u, v;
                uint64_t ind1, ind2, ind3, ind4, i1, i2, i3, i4;

                ind1 = ((ind >> (log_gap_-2)) << log_gap_) + (ind & ((gap_>>2) - 1));
                ind2 = ind1 + (gap_>>2);
                ind3 = ind2 + (gap_>>2);
                ind4 = ind3 + (gap_>>2);

                compute_ind_4(log_gap_-2)

                auto j = ind1 & ((gap_>>2) - 1);
                auto offset = (i1 << ((log_gap_-2) + 1)) + j;

                ld_roots_4(roots_op_, r_op, rounds_)
                ld_roots_4(roots_quo_, r_quo, rounds_)
                init_global_ptr_8((gap_>>2))
                ld_global_mem_to_reg_8(0)

                // round1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)


                // inner round 2
                compute_ind_4(log_gap_-1)
                ld_roots_4(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_4(roots_quo_, r_quo, (rounds_+ m_ ))
                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_)
                ld_roots_4(roots_op_, r_op, (rounds_+ m_+ (m_>>1) ))
                ld_roots_4(roots_quo_, r_quo, (rounds_+ m_ + (m_>>1) ))

                butterfly_inv_ntt_reg2reg(0, 4, 0)
                butterfly_inv_ntt_reg2reg(1, 5, 1)
                butterfly_inv_ntt_reg2reg(2, 6, 2)
                butterfly_inv_ntt_reg2reg(3, 7, 3)

                st_reg2gmem_scale_8(0)

            }

            
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
        };

        // Last Round RNS Inverse NTT kernel
        template <typename T>
        class RnsInvDwtLastRoundRescale {
        public:
            RnsInvDwtLastRoundRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_gap, int rounds, T* values, const T* modulus,
                               const T* roots_op, const T* roots_quo, bool lazy = false)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                modulus_ = modulus;
                log_n_ = log_n;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap);
                rounds_ = rounds;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                lazy_ = lazy;
            }

            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t i) const {

                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                auto r_op = roots_op_[rounds_ + global_offset];
                auto r_quo = roots_quo_[rounds_ + global_offset];
                auto x = values_ + global_offset + i + poly_offset;
                auto y = x + gap_;
                auto u = *x;
                auto v = *y;
                auto uu = dwt_guard(dwt_add(u, v), two_times_modulus);
                auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);

                if (!lazy_)
                {
                    uu -= (uu >= modulus) ? modulus : 0;
                    vv -= (vv >= modulus) ? modulus : 0;
                }

                *x = uu;
                *y = vv;


            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t poly_idx = ind[0];
                uint64_t i = ind[1];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_gap_;
            std::size_t gap_;
            T* values_;
            const T* modulus_;
            int rounds_;
            const T* roots_op_;
            const T* roots_quo_;
            bool lazy_;

        };

        // Last Round RNS Inverse NTT kernel with scalar multiplied on the output
        template <typename T>
        class RnsInvDwtLastRoundScalarRescale {
        public:
            RnsInvDwtLastRoundScalarRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_gap, int rounds, T* values, const T* modulus,
                                     const T* roots_op, const T* roots_quo,
                                     const T* scalar_op, const T* scalar_quo, bool lazy = false)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                modulus_ = modulus;
                log_n_ = log_n;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap);
                rounds_ = rounds;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
                lazy_ = lazy;
            }

            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t i) const {
                auto modulus = modulus_[position_];
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                auto r_op = roots_op_[rounds_ + global_offset];
                T scaled_r_op;
                T scaled_r_quo;
                dwt_mul_root_scalar(r_op, scalar_op, scalar_quo, modulus, scaled_r_op, scaled_r_quo);

                auto x = values_ + global_offset + i + poly_offset;
                auto y = x + gap_;

                auto u = dwt_guard(*x, two_times_modulus);
                auto v = *y;
                auto uu = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), scalar_op, scalar_quo, modulus);
                auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r_op, scaled_r_quo, modulus);

                if (!lazy_)
                {
                    uu -= (uu >= modulus) ? modulus : 0;
                    vv -= (vv >= modulus) ? modulus : 0;
                }

                *x = uu;
                *y = vv;

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t poly_idx = ind[0];
                uint64_t i = ind[1];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_gap_;
            std::size_t gap_;
            T* values_;
            const T* modulus_;
            int rounds_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
            bool lazy_;
        };

        // Last Round RNS Inverse NTT kernel with scalar multiplied on the output
        // seperate kernel without basic butterfly ops
        template <typename T>
        class RnsInvDwtLastRoundScalarSeperateRescale {
        public:
            RnsInvDwtLastRoundScalarSeperateRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_gap, int rounds, T* values, const T* modulus,
                                     const T* roots_op, const T* roots_quo,
                                     const T* scalar_op, const T* scalar_quo, bool lazy = false)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                modulus_ = modulus;
                log_n_ = log_n;
                log_gap_ = log_gap;
                gap_ = (1 << log_gap);
                rounds_ = rounds;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
                lazy_ = lazy;
            }

            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t i) const {
                auto modulus = modulus_[position_];
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);


                auto x = values_ + global_offset + 2 * i + poly_offset;
                auto y = x + 1;

                auto uu = dwt_mul_scalar(dwt_guard(*x, two_times_modulus), scalar_op, scalar_quo, modulus);
                auto vv = dwt_mul_scalar(dwt_guard(*y, two_times_modulus), scalar_op, scalar_quo, modulus);

                if (!lazy_)
                {
                    uu -= (uu >= modulus) ? modulus : 0;
                    vv -= (vv >= modulus) ? modulus : 0;
                }

                *x = uu;
                *y = vv;

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t poly_idx = ind[0];
                uint64_t i = ind[1];
                kernel(poly_idx, i);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_gap_;
            std::size_t gap_;
            T* values_;
            const T* modulus_;
            int rounds_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
            bool lazy_;
        };

        // CPU RNS Inverse NTT Class
        template <typename T>
        class InvRnsDwtRescale{
        public:
            InvRnsDwtRescale(int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo,
                      const T* scalar_op = nullptr, const T* scalar_quo = nullptr, bool lazy = false)
            {
                poly_num_ = poly_num;
                q_base_size_ = q_base_size;
                log_n_ = log_n;
                values_ = values;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
                lazy_ = lazy;
            }

            void operator() [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](void) {
                // constant transform size
                size_t n = size_t(1) << log_n_;
                // variables for indexing
                std::size_t gap = 1;
                std::size_t log_gap = 0;
                std::size_t m = n >> 1;
                std::size_t log_m = log_n_ - 1;
                std::size_t total_r = 1;
                for (; m > 1; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                {
                        for (std::size_t i = 0; i < poly_num_; ++i) {
                                for (uint64_t j = 0; j < (1 << (log_m + log_gap)); ++j){
                                    RnsInvDwtGapRescale<T, 0>(0, q_base_size_, log_n_, log_m, log_gap, total_r,
                                                                         values_, modulus_,
                                                                         roots_op_, roots_quo_).kernel(i, j);
                                }
                        }
                }

                {
                    for (std::size_t i = 0; i < poly_num_; ++i) {
                            for (uint64_t j = 0; j < (1 << log_gap); ++j){
                                RnsInvDwtLastRoundScalarRescale<T>(0, q_base_size_, log_n_, log_gap,
                                                                                          total_r, values_,
                                                                                          modulus_, roots_op_, roots_quo_,
                                                                                          scalar_op_, scalar_quo_,
                                                                                          lazy_).kernel(i, j);
                            }
                    }
                }

            }

        protected:
            int poly_num_;
            int q_base_size_;
            int log_n_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
            bool lazy_;

        };

#ifdef BUILD_WITH_IGPU

#include "dpcpp_utils.h"
#include "lib_utils.h"

        template <typename T>
        class RnsInvDwtGapLocalRadix8FusedRescale {
        public:
            RnsInvDwtGapLocalRadix8FusedRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind, uint64_t ind_local, cl::sycl::nd_item<2> &item) const {
                //prologue: load data from the global memory
                //          and store to local memory
                auto n = (1<<log_n_);
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                T *x[8], *x_local[8];
                T dx[8];
                T r_op[4], r_quo[4];
                T u, v;
                uint64_t ind1, ind2, ind3, ind4, i1, i2, i3, i4;
                uint64_t ind1_local, ind2_local, ind3_local, ind4_local;
                uint64_t i1_local, i2_local, i3_local, i4_local;
                uint64_t j, j_local, offset, offset_local;
                ind1 = ((ind >> (log_gap_local-2)) << log_gap_local) + (ind & ((gap_local>>2) - 1));
                ind2 = ind1 + (gap_local>>2);
                ind3 = ind2 + (gap_local>>2);
                ind4 = ind3 + (gap_local>>2);

                ind1_local = ((ind_local >> (log_gap_local-2)) << log_gap_local) + (ind_local & ((gap_local>>2) - 1));
                ind2_local = ind1_local + (gap_local>>2);
                ind3_local = ind2_local + (gap_local>>2);
                ind4_local = ind3_local + (gap_local>>2);

                compute_ind_4((log_gap_local-2))
                j = ind1 & ((gap_local>>2) - 1);

                offset = (i1 << ((log_gap_local-2) + 1)) + j;

                compute_ind_local_4(log_gap_local-2)
                j_local = ind1_local & ((gap_local>>2) - 1);
                offset_local = (i1_local << ((log_gap_local-2) + 1)) + j_local;

                ld_roots_4(roots_op_, r_op, rounds_local)
                ld_roots_4(roots_quo_, r_quo, rounds_local)

                init_global_ptr_8((gap_local>>2))
                init_local_ptr_8((gap_local>>2))

                ld_global_mem_to_reg_8(0)

                // round1 
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)

                // inner round 2
                compute_ind_4(log_gap_local-1)
                ld_roots_4(roots_op_, r_op, (rounds_local+ m_local ))
                ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_local)
                ld_roots_4(roots_op_, r_op, (rounds_local+ m_local+ (m_local>>1) ))
                ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local+ (m_local>>1) ))
                butterfly_inv_ntt_reg2lmem(0, 4, 0)
                butterfly_inv_ntt_reg2lmem(1, 5, 1)
                butterfly_inv_ntt_reg2lmem(2, 6, 2)
                butterfly_inv_ntt_reg2lmem(3, 7, 3)

                item.barrier();
                rounds_local +=( m_local + (m_local>>1) + (m_local>>2)), m_local >>= 3, log_m_local -= 3, gap_local <<= 3, log_gap_local += 3;
                //loop body: play inside of the local memory
                for (; log_gap_local <= log_n_ - 1;rounds_local +=( m_local + (m_local>>1) + (m_local>>2)), m_local >>= 3, log_m_local -= 3, gap_local <<= 3, log_gap_local += 3){
                    ind1 = ((ind >> (log_gap_local-2)) << log_gap_local) + (ind & ((gap_local>>2) - 1));
                    ind2 = ind1 + (gap_local>>2);
                    ind3 = ind2 + (gap_local>>2);
                    ind4 = ind3 + (gap_local>>2);

                    ind1_local = ((ind_local >> (log_gap_local-2)) << log_gap_local) + (ind_local & ((gap_local>>2) - 1));
                    ind2_local = ind1_local + (gap_local>>2);
                    ind3_local = ind2_local + (gap_local>>2);
                    ind4_local = ind3_local + (gap_local>>2);

                    compute_ind_4(log_gap_local-2)
                    compute_ind_local_4(log_gap_local-2)
                    j_local = ind1_local & ((gap_local>>2) - 1);
                    offset_local = (i1_local << ((log_gap_local-2) + 1)) + j_local;

                    ld_roots_4(roots_op_, r_op, rounds_local)
                    ld_roots_4(roots_quo_, r_quo, rounds_local)
                    init_local_ptr_8((gap_local>>2))
                    ld_local_mem_to_reg_8(0)
                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 5, 2)
                    butterfly_inv_ntt_reg2reg(6, 7, 3)

                    // inner round 2
                    compute_ind_4(log_gap_local-1)
                    ld_roots_4(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                    butterfly_inv_ntt_reg2reg(0, 2, 0)
                    butterfly_inv_ntt_reg2reg(1, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 6, 2)
                    butterfly_inv_ntt_reg2reg(5, 7, 3)

                    // inner round 3
                    compute_ind_4(log_gap_local)
                    ld_roots_4(roots_op_, r_op, (rounds_local+ m_local + (m_local>>1) ))
                    ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local + (m_local>>1) ))

                    butterfly_inv_ntt_reg2lmem(0, 4, 0)
                    butterfly_inv_ntt_reg2lmem(1, 5, 1)
                    butterfly_inv_ntt_reg2lmem(2, 6, 2)
                    butterfly_inv_ntt_reg2lmem(3, 7, 3)
                    item.barrier();//sync the work-group
                }
                //recover gap and log_gap before entering the epilogue
                gap_local = (n >> 1); log_gap_local = log_n_ - 1;
                //epilogue: write data back to global memory
                ind1 = ((ind >> (log_gap_local-2)) << log_gap_local) + (ind & ((gap_local>>2) - 1));
                ind2 = ind1 + (gap_local>>2);
                ind3 = ind2 + (gap_local>>2);
                ind4 = ind3 + (gap_local>>2);

                ind1_local = ((ind_local >> (log_gap_local-2)) << log_gap_local) + (ind_local & ((gap_local>>2) - 1));
                ind2_local = ind1_local + (gap_local>>2);
                ind3_local = ind2_local + (gap_local>>2);
                ind4_local = ind3_local + (gap_local>>2);

                compute_ind_4(log_gap_local-2)
                compute_ind_local_4(log_gap_local-2)

                j = ind1 & ((gap_local>>2) - 1);
                j_local = ind1_local & ((gap_local>>2) - 1);
                offset = (i1 << ((log_gap_local-2) + 1)) + j;
                offset_local = (i1_local << ((log_gap_local-2) + 1)) + j_local;

                init_local_ptr_8((gap_local>>2))
                ld_local2reg_scale_8(0)
                init_global_ptr_8((gap_local>>2))
                st_reg_to_global_mem_8(0)

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            int position_;
            local_accessor<T, 1> ptr_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            std::size_t m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
        };

        template <typename T>
        class RnsInvDwtGapLocalRadix8Rescale {
        public:
            RnsInvDwtGapLocalRadix8Rescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind, uint64_t ind_local, cl::sycl::nd_item<2> &item) const {
                //prologue: load data from the global memory
                //          and store to local memory
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                T *x[8], *x_local[8];
                T dx[8];
                T r_op[4], r_quo[4];
                T u, v;
                uint64_t ind1, ind2, ind3, ind4, i1, i2, i3, i4;
                uint64_t ind1_local, ind2_local, ind3_local, ind4_local;
                uint64_t i1_local, i2_local, i3_local, i4_local;
                uint64_t j, j_local, offset, offset_local;
                ind1 = ((ind >> (log_gap_local-2)) << log_gap_local) + (ind & ((gap_local>>2) - 1));
                ind2 = ind1 + (gap_local>>2);
                ind3 = ind2 + (gap_local>>2);
                ind4 = ind3 + (gap_local>>2);

                ind1_local = ((ind_local >> (log_gap_local-2)) << log_gap_local) + (ind_local & ((gap_local>>2) - 1));
                ind2_local = ind1_local + (gap_local>>2);
                ind3_local = ind2_local + (gap_local>>2);
                ind4_local = ind3_local + (gap_local>>2);

                compute_ind_4((log_gap_local-2))
                j = ind1 & ((gap_local>>2) - 1);

                offset = (i1 << ((log_gap_local-2) + 1)) + j;

                compute_ind_local_4(log_gap_local-2)
                j_local = ind1_local & ((gap_local>>2) - 1);
                offset_local = (i1_local << ((log_gap_local-2) + 1)) + j_local;

                ld_roots_4(roots_op_, r_op, rounds_local)
                ld_roots_4(roots_quo_, r_quo, rounds_local)

                init_global_ptr_8((gap_local>>2))
                init_local_ptr_8((gap_local>>2))

                ld_global_mem_to_reg_8(0)

                // round1 
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)

                // inner round 2
                compute_ind_4(log_gap_local-1)
                ld_roots_4(roots_op_, r_op, (rounds_local+ m_local ))
                ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_local)
                ld_roots_4(roots_op_, r_op, (rounds_local+ m_local+ (m_local>>1) ))
                ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local+ (m_local>>1) ))
                butterfly_inv_ntt_reg2lmem(0, 4, 0)
                butterfly_inv_ntt_reg2lmem(1, 5, 1)
                butterfly_inv_ntt_reg2lmem(2, 6, 2)
                butterfly_inv_ntt_reg2lmem(3, 7, 3)

                item.barrier();
                rounds_local +=( m_local + (m_local>>1) + (m_local>>2)), m_local >>= 3, log_m_local -= 3, gap_local <<= 3, log_gap_local += 3;
                //loop body: play inside of SLM
                //assumed the SLM sz = 4096 elements.
                for (; log_gap_local <= 11;rounds_local +=( m_local + (m_local>>1) + (m_local>>2)), m_local >>= 3, log_m_local -= 3, gap_local <<= 3, log_gap_local += 3){
                    ind1 = ((ind >> (log_gap_local-2)) << log_gap_local) + (ind & ((gap_local>>2) - 1));
                    ind2 = ind1 + (gap_local>>2);
                    ind3 = ind2 + (gap_local>>2);
                    ind4 = ind3 + (gap_local>>2);

                    ind1_local = ((ind_local >> (log_gap_local-2)) << log_gap_local) + (ind_local & ((gap_local>>2) - 1));
                    ind2_local = ind1_local + (gap_local>>2);
                    ind3_local = ind2_local + (gap_local>>2);
                    ind4_local = ind3_local + (gap_local>>2);

                    compute_ind_4(log_gap_local-2)
                    compute_ind_local_4(log_gap_local-2)
                    j_local = ind1_local & ((gap_local>>2) - 1);
                    offset_local = (i1_local << ((log_gap_local-2) + 1)) + j_local;

                    ld_roots_4(roots_op_, r_op, rounds_local)
                    ld_roots_4(roots_quo_, r_quo, rounds_local)
                    init_local_ptr_8((gap_local>>2))
                    ld_local_mem_to_reg_8(0)
                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 5, 2)
                    butterfly_inv_ntt_reg2reg(6, 7, 3)

                    // inner round 2
                    compute_ind_4(log_gap_local-1)
                    ld_roots_4(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                    butterfly_inv_ntt_reg2reg(0, 2, 0)
                    butterfly_inv_ntt_reg2reg(1, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 6, 2)
                    butterfly_inv_ntt_reg2reg(5, 7, 3)

                    // inner round 3
                    compute_ind_4(log_gap_local)
                    ld_roots_4(roots_op_, r_op, (rounds_local+ m_local + (m_local>>1) ))
                    ld_roots_4(roots_quo_, r_quo, (rounds_local+ m_local + (m_local>>1) ))

                    butterfly_inv_ntt_reg2lmem(0, 4, 0)
                    butterfly_inv_ntt_reg2lmem(1, 5, 1)
                    butterfly_inv_ntt_reg2lmem(2, 6, 2)
                    butterfly_inv_ntt_reg2lmem(3, 7, 3)
                    item.barrier();//sync the work-group
                }
                //recover gap and log_gap before entering the epilogue
                gap_local = 2048; log_gap_local = 11;
                //epilogue: write data back to global memory
                ind1 = ((ind >> (log_gap_local-2)) << log_gap_local) + (ind & ((gap_local>>2) - 1));
                ind2 = ind1 + (gap_local>>2);
                ind3 = ind2 + (gap_local>>2);
                ind4 = ind3 + (gap_local>>2);

                ind1_local = ((ind_local >> (log_gap_local-2)) << log_gap_local) + (ind_local & ((gap_local>>2) - 1));
                ind2_local = ind1_local + (gap_local>>2);
                ind3_local = ind2_local + (gap_local>>2);
                ind4_local = ind3_local + (gap_local>>2);

                compute_ind_4(log_gap_local-2)
                compute_ind_local_4(log_gap_local-2)

                j = ind1 & ((gap_local>>2) - 1);
                j_local = ind1_local & ((gap_local>>2) - 1);
                offset = (i1 << ((log_gap_local-2) + 1)) + j;
                offset_local = (i1_local << ((log_gap_local-2) + 1)) + j_local;

                init_local_ptr_8((gap_local>>2))
                ld_local2reg_8(0)
                init_global_ptr_8((gap_local>>2))
                st_reg_to_global_mem_8(0)

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            int position_;
            local_accessor<T, 1> ptr_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            std::size_t m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
        };

        template <typename T, int unroll, int log_unroll>
        class RnsInvDwtGapLocalRadix4FusedRescale {
        public:
            RnsInvDwtGapLocalRadix4FusedRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind, uint64_t ind_local, cl::sycl::nd_item<2> &item) const {
                //prologue: load data from the global memory
                //          and store to local memory
                auto n = (1<<log_n_);
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                T *x[4], *x_local[4];
                T dx[4];
                T r_op[2], r_quo[2];
                T u, v;
                uint64_t ind1, ind2, i1, i2;
                uint64_t ind1_local, ind2_local;
                uint64_t i1_local, i2_local;
                uint64_t j, j_local, offset, offset_local;
                for (auto unroll_cnt = 0; unroll_cnt < unroll; unroll_cnt++){
                    auto ind_curr = (ind << log_unroll) + unroll_cnt;
                    auto ind_local_curr = (ind_local << log_unroll) + unroll_cnt;
                    ind1 = ((ind_curr >> (log_gap_local-1)) << log_gap_local) + (ind_curr & ((gap_local>>1) - 1));
                    ind2 = ind1 + (gap_local>>1);

                    ind1_local = ((ind_local_curr >> (log_gap_local-1)) << log_gap_local) + (ind_local_curr & ((gap_local>>1) - 1));
                    ind2_local = ind1_local + (gap_local>>1);

                    compute_ind_2((log_gap_local-1))
                    j = ind1 & ((gap_local>>1) - 1);

                    offset = (i1 << ((log_gap_local-1) + 1)) + j;

                    compute_ind_local_2(log_gap_local-1)
                    j_local = ind1_local & ((gap_local>>1) - 1);
                    offset_local = (i1_local << ((log_gap_local-1) + 1)) + j_local;

                    ld_roots_2(roots_op_, r_op, rounds_local)
                    ld_roots_2(roots_quo_, r_quo, rounds_local)

                    init_global_ptr_4((gap_local>>1))
                    init_local_ptr_4((gap_local>>1))

                    ld_global_mem_to_reg_4(0)

                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)

                    // round2
                    compute_ind_2(log_gap_local)
                    ld_roots_2(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_2(roots_quo_, r_quo, (rounds_local+ m_local ))

                    butterfly_inv_ntt_reg2lmem(0, 2, 0)
                    butterfly_inv_ntt_reg2lmem(1, 3, 1)
                }
                item.barrier();
                rounds_local +=( m_local + (m_local>>1) ), m_local >>= 2, log_m_local -= 2, gap_local <<= 2, log_gap_local += 2;
                //loop body: play inside of the local memory
                for (; log_gap_local <= log_n_ - 1;rounds_local +=( m_local + (m_local>>1)), m_local >>= 2, log_m_local -= 2, gap_local <<= 2, log_gap_local += 2){
                    for (auto unroll_cnt = 0; unroll_cnt < unroll; unroll_cnt++){
                        auto ind_curr = (ind << log_unroll) + unroll_cnt;
                        auto ind_local_curr = (ind_local << log_unroll) + unroll_cnt;
                        ind1 = ((ind_curr >> (log_gap_local-1)) << log_gap_local) + (ind_curr & ((gap_local>>1) - 1));
                        ind2 = ind1 + (gap_local>>1);

                        ind1_local = ((ind_local_curr >> (log_gap_local-1)) << log_gap_local) + (ind_local_curr & ((gap_local>>1) - 1));
                        ind2_local = ind1_local + (gap_local>>1);

                        compute_ind_2(log_gap_local-1)
                        compute_ind_local_2(log_gap_local-1)
                        j_local = ind1_local & ((gap_local>>1) - 1);
                        offset_local = (i1_local << ((log_gap_local-1) + 1)) + j_local;

                        ld_roots_2(roots_op_, r_op, rounds_local)
                        ld_roots_2(roots_quo_, r_quo, rounds_local)
                        init_local_ptr_4((gap_local>>1))
                        ld_local_mem_to_reg_4(0)
                        // round1 
                        butterfly_inv_ntt_reg2reg(0, 1, 0)
                        butterfly_inv_ntt_reg2reg(2, 3, 1)

                        // round2
                        compute_ind_2(log_gap_local)
                        ld_roots_2(roots_op_, r_op, (rounds_local+ m_local ))
                        ld_roots_2(roots_quo_, r_quo, (rounds_local+ m_local ))

                        butterfly_inv_ntt_reg2lmem(0, 2, 0)
                        butterfly_inv_ntt_reg2lmem(1, 3, 1)
                    }

                    item.barrier();//sync the work-group
                }
                //recover gap and log_gap before entering the epilogue
                gap_local = (n >> 1); log_gap_local = log_n_ - 1;
                //epilogue: write data back to global memory
                for (auto unroll_cnt = 0; unroll_cnt < unroll; unroll_cnt++){
                    auto ind_curr = (ind << log_unroll) + unroll_cnt;
                    auto ind_local_curr = (ind_local << log_unroll) + unroll_cnt;
                    ind1 = ((ind_curr >> (log_gap_local-1)) << log_gap_local) + (ind_curr & ((gap_local>>1) - 1));
                    ind2 = ind1 + (gap_local>>1);

                    ind1_local = ((ind_local_curr >> (log_gap_local-1)) << log_gap_local) + (ind_local_curr & ((gap_local>>1) - 1));
                    ind2_local = ind1_local + (gap_local>>1);
                    compute_ind_2(log_gap_local-1)
                    compute_ind_local_2(log_gap_local-1)

                    j = ind1 & ((gap_local>>1) - 1);
                    j_local = ind1_local & ((gap_local>>1) - 1);
                    offset = (i1 << ((log_gap_local-1) + 1)) + j;
                    offset_local = (i1_local << ((log_gap_local-1) + 1)) + j_local;

                    init_local_ptr_4((gap_local>>1))
                    ld_local2reg_scale_4(0)
                    init_global_ptr_4((gap_local>>1))
                    st_reg_to_global_mem_4(0)
                }

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            int position_;
            local_accessor<T, 1> ptr_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            std::size_t m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
        };

        template <typename T, int unroll, int log_unroll>
        class RnsInvDwtGapLocalRadix4Rescale {
        public:
            RnsInvDwtGapLocalRadix4Rescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind, uint64_t ind_local, cl::sycl::nd_item<2> &item) const {
                //prologue: load data from the global memory
                //          and store to local memory
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);

                T *x[4], *x_local[4];
                T dx[4];
                T r_op[2], r_quo[2];
                T u, v;
                uint64_t ind1, ind2, i1, i2;
                uint64_t ind1_local, ind2_local;
                uint64_t i1_local, i2_local;
                uint64_t j, j_local, offset, offset_local;
                for (auto unroll_cnt = 0; unroll_cnt < unroll; unroll_cnt++){
                    auto ind_curr = (ind << log_unroll) + unroll_cnt;
                    auto ind_local_curr = (ind_local << log_unroll) + unroll_cnt;
                    ind1 = ((ind_curr >> (log_gap_local-1)) << log_gap_local) + (ind_curr & ((gap_local>>1) - 1));
                    ind2 = ind1 + (gap_local>>1);

                    ind1_local = ((ind_local_curr >> (log_gap_local-1)) << log_gap_local) + (ind_local_curr & ((gap_local>>1) - 1));
                    ind2_local = ind1_local + (gap_local>>1);

                    compute_ind_2((log_gap_local-1))
                    j = ind1 & ((gap_local>>1) - 1);

                    offset = (i1 << ((log_gap_local-1) + 1)) + j;

                    compute_ind_local_2(log_gap_local-1)
                    j_local = ind1_local & ((gap_local>>1) - 1);
                    offset_local = (i1_local << ((log_gap_local-1) + 1)) + j_local;

                    ld_roots_2(roots_op_, r_op, rounds_local)
                    ld_roots_2(roots_quo_, r_quo, rounds_local)

                    init_global_ptr_4((gap_local>>1))
                    init_local_ptr_4((gap_local>>1))

                    ld_global_mem_to_reg_4(0)

                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)

                    // round2
                    compute_ind_2(log_gap_local)
                    ld_roots_2(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_2(roots_quo_, r_quo, (rounds_local+ m_local ))

                    butterfly_inv_ntt_reg2lmem(0, 2, 0)
                    butterfly_inv_ntt_reg2lmem(1, 3, 1)
                }
                item.barrier();
                rounds_local +=( m_local + (m_local>>1) ), m_local >>= 2, log_m_local -= 2, gap_local <<= 2, log_gap_local += 2;
                //loop body: play inside of the local memory
                for (; log_gap_local <= 11;rounds_local +=( m_local + (m_local>>1)), m_local >>= 2, log_m_local -= 2, gap_local <<= 2, log_gap_local += 2){
                    for (auto unroll_cnt = 0; unroll_cnt < unroll; unroll_cnt++){
                        auto ind_curr = (ind << log_unroll) + unroll_cnt;
                        auto ind_local_curr = (ind_local << log_unroll) + unroll_cnt;
                        ind1 = ((ind_curr >> (log_gap_local-1)) << log_gap_local) + (ind_curr & ((gap_local>>1) - 1));
                        ind2 = ind1 + (gap_local>>1);

                        ind1_local = ((ind_local_curr >> (log_gap_local-1)) << log_gap_local) + (ind_local_curr & ((gap_local>>1) - 1));
                        ind2_local = ind1_local + (gap_local>>1);

                        compute_ind_2(log_gap_local-1)
                        compute_ind_local_2(log_gap_local-1)
                        j_local = ind1_local & ((gap_local>>1) - 1);
                        offset_local = (i1_local << ((log_gap_local-1) + 1)) + j_local;

                        ld_roots_2(roots_op_, r_op, rounds_local)
                        ld_roots_2(roots_quo_, r_quo, rounds_local)
                        init_local_ptr_4((gap_local>>1))
                        ld_local_mem_to_reg_4(0)
                        // round1 
                        butterfly_inv_ntt_reg2reg(0, 1, 0)
                        butterfly_inv_ntt_reg2reg(2, 3, 1)

                        // round2
                        compute_ind_2(log_gap_local)
                        ld_roots_2(roots_op_, r_op, (rounds_local+ m_local ))
                        ld_roots_2(roots_quo_, r_quo, (rounds_local+ m_local ))

                        butterfly_inv_ntt_reg2lmem(0, 2, 0)
                        butterfly_inv_ntt_reg2lmem(1, 3, 1)
                    }

                    item.barrier();//sync the work-group
                }
                //recover gap and log_gap before entering the epilogue
                gap_local = 2048; log_gap_local = 11;
                //epilogue: write data back to global memory
                for (auto unroll_cnt = 0; unroll_cnt < unroll; unroll_cnt++){
                    auto ind_curr = (ind << log_unroll) + unroll_cnt;
                    auto ind_local_curr = (ind_local << log_unroll) + unroll_cnt;
                    ind1 = ((ind_curr >> (log_gap_local-1)) << log_gap_local) + (ind_curr & ((gap_local>>1) - 1));
                    ind2 = ind1 + (gap_local>>1);

                    ind1_local = ((ind_local_curr >> (log_gap_local-1)) << log_gap_local) + (ind_local_curr & ((gap_local>>1) - 1));
                    ind2_local = ind1_local + (gap_local>>1);
                    compute_ind_2(log_gap_local-1)
                    compute_ind_local_2(log_gap_local-1)

                    j = ind1 & ((gap_local>>1) - 1);
                    j_local = ind1_local & ((gap_local>>1) - 1);
                    offset = (i1 << ((log_gap_local-1) + 1)) + j;
                    offset_local = (i1_local << ((log_gap_local-1) + 1)) + j_local;

                    init_local_ptr_4((gap_local>>1))
                    ld_local2reg_4(0)
                    init_global_ptr_4((gap_local>>1))
                    st_reg_to_global_mem_4(0)
                }

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            int position_;
            local_accessor<T, 1> ptr_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            std::size_t m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
        };

        // We don't do unrolling here.
        template <typename T>
        class RnsInvDwtGapSIMDRescale {
        public:
            RnsInvDwtGapSIMDRescale(int position, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo)
            {
                position_ = position;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t idx, cl::sycl::ext::oneapi::sub_group &sg) const {
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                T data[LOCAL_REG_NUMBERS];
                T r_op[LOCAL_REG_SLOTS];
                T r_quo[LOCAL_REG_SLOTS];
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                auto i = (idx >> LOG_SUB_GROUP_SIZE);
                auto idx_r_op_offset = ( ((i << LOG_TER_GAP_SIZE) + (idx & (SUB_GROUP_SIZE - 1))) >> log_gap_local );
                auto j = (idx & (SUB_GROUP_SIZE - 1));
                auto r_op_quo_base = rounds_local + global_offset;
                auto r_op_quo_base_offset = r_op_quo_base + idx_r_op_offset;
                std::size_t offset = ((i << LOG_TER_GAP_SIZE) << 1) + (j<<1);
                std::size_t poly_offset = ((q_base_size_ *poly_idx) << log_n_);
                auto x = values_ + global_offset + offset + poly_offset;
                auto y = x + gap_local;
                auto log_r_op_share = LOG_TER_GAP_SIZE - log_gap_local;
                for (int local_idx = 0, slot_idx = 0; local_idx < LOCAL_REG_NUMBERS; local_idx += 2, slot_idx++){
                    auto r_op_offset_two = (LOG_LOCAL_REG_SLOTS <= log_r_op_share) ? (slot_idx << (log_r_op_share - LOG_LOCAL_REG_SLOTS)) : (slot_idx >> (LOG_LOCAL_REG_SLOTS - log_r_op_share));
                    r_op[slot_idx] = roots_op_[r_op_quo_base_offset + r_op_offset_two];
                    r_quo[slot_idx] = roots_quo_[r_op_quo_base_offset + r_op_offset_two];
                    auto u = *x;
                    auto v = *y;
                    data[local_idx] = dwt_guard(dwt_add(u, v), two_times_modulus);
                    data[local_idx+1] = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op[slot_idx], r_quo[slot_idx], modulus);
                    x += (SUB_GROUP_SIZE<<1); y += (SUB_GROUP_SIZE<<1);
                }
                rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++;

                auto lane_id = (idx & (SUB_GROUP_SIZE - 1));//compute lane_id for swapping
                // this requires lane exchanging.
                for (; gap_local <= SUB_GROUP_SIZE; rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++){
                    auto shift_idx = (lane_id >> (log_gap_local - 1));
                    auto tmp1 = (( shift_idx + 1 ) & 1);
                    auto tgt_idx = lane_id + ((( tmp1 <<1)-1) << (log_gap_local - 1));
                    for (int slot_idx = 0, local_idx = 0; slot_idx < LOCAL_REG_SLOTS; slot_idx++, local_idx += 2){
                        //swapping data
                        data[tmp1 + (slot_idx << 1) ] = sg.shuffle(data[tmp1 + (slot_idx << 1)], tgt_idx);
                    }
                    //comput q indices
                    auto log_r_op_share = LOG_TER_GAP_SIZE - log_gap_local;
                    idx_r_op_offset = ( ((i << LOG_TER_GAP_SIZE) + (idx & (SUB_GROUP_SIZE - 1))) >> log_gap_local );
                    r_op_quo_base = rounds_local + global_offset;
                    r_op_quo_base_offset = r_op_quo_base + idx_r_op_offset;
                    //computing
                    for (int local_idx = 0, slot_idx = 0; local_idx < LOCAL_REG_NUMBERS; local_idx += 2, slot_idx++){
                        auto r_op_offset_two = (LOG_LOCAL_REG_SLOTS <= log_r_op_share) ? (slot_idx << (log_r_op_share - LOG_LOCAL_REG_SLOTS)) : (slot_idx >> (LOG_LOCAL_REG_SLOTS - log_r_op_share));
                        r_op[slot_idx] = roots_op_[r_op_quo_base_offset + r_op_offset_two];
                        r_quo[slot_idx] = roots_quo_[r_op_quo_base_offset + r_op_offset_two];
                        auto u = data[local_idx];
                        auto v = data[local_idx+1];
                        data[local_idx] = dwt_guard(dwt_add(u, v), two_times_modulus);
                        data[local_idx+1] = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op[slot_idx], r_quo[slot_idx], modulus);
                    }
                    sg.barrier();
                }
                for (int log_slot_swap_gap = 0; gap_local <= TER_GAP_SIZE; rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++, log_slot_swap_gap++){
                    //swapping data in register slots
                    auto slot_swap_gap = (1 << log_slot_swap_gap);
                    for (int count_iter = 0; count_iter < (LOCAL_REG_SLOTS_HALF >> log_slot_swap_gap); count_iter++){
                        // to swap idx and idx + slot_swap_gap
                        for (int inner_counter_iter = 0; inner_counter_iter < slot_swap_gap; inner_counter_iter++){
                            auto curr_slot = (count_iter << (log_slot_swap_gap + 1)) + inner_counter_iter;
                            auto tgt_slot = curr_slot + slot_swap_gap;
                            auto curr_slot_idx = (( ((curr_slot << LOG_SUB_GROUP_SIZE) >> (log_gap_local-1)) + 1 ) & 1) + (curr_slot<<1);
                            auto tgt_slot_idx = (( ((tgt_slot << LOG_SUB_GROUP_SIZE) >> (log_gap_local-1)) + 1 ) & 1) + (tgt_slot<<1);
                            //swapping
                            auto tmp = data[curr_slot_idx];
                            data[curr_slot_idx] = data[tgt_slot_idx];
                            data[tgt_slot_idx] = tmp;
                        }
                    }
                    auto log_r_op_share = LOG_TER_GAP_SIZE - log_gap_local;
                    idx_r_op_offset = ( ((i << LOG_TER_GAP_SIZE) + (idx & (SUB_GROUP_SIZE - 1))) >> log_gap_local );
                    r_op_quo_base = rounds_local + global_offset;
                    r_op_quo_base_offset = r_op_quo_base + idx_r_op_offset;
                    
                    //computing
                    for (int local_idx = 0, slot_idx = 0; local_idx < LOCAL_REG_NUMBERS; local_idx += 2, slot_idx++){
                        auto r_op_offset_two = (LOG_LOCAL_REG_SLOTS <= log_r_op_share) ? (slot_idx << (log_r_op_share - LOG_LOCAL_REG_SLOTS)) : (slot_idx >> (LOG_LOCAL_REG_SLOTS - log_r_op_share));
                        r_op[slot_idx] = roots_op_[r_op_quo_base_offset + r_op_offset_two];
                        r_quo[slot_idx] = roots_quo_[r_op_quo_base_offset + r_op_offset_two];
                        auto u = data[local_idx];
                        auto v = data[local_idx+1];
                        data[local_idx] = dwt_guard(dwt_add(u, v), two_times_modulus);
                        data[local_idx+1] = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op[slot_idx], r_quo[slot_idx], modulus);
                    }

                }
                //recover indices from the previous for loop
                gap_local >>= 1;
                log_gap_local--;
                //compute global indices for last round write
                auto t_idx = (idx & (SUB_GROUP_SIZE - 1));
                i = (t_idx >> log_gap_local);
                j = t_idx - (i << log_gap_local);
                offset = (i << (log_gap_local + 1)) + (j) + ((idx >> LOG_SUB_GROUP_SIZE) << (LOG_TER_GAP_SIZE + 1));
                x = values_ + global_offset + offset + poly_offset;
                y = x + gap_local;
                //loop over all the slots and write back to global memory
                for (int local_idx = 0, slot_idx = 0; local_idx < LOCAL_REG_NUMBERS; local_idx += 2, slot_idx++){   
                    *x = data[local_idx];
                    *y = data[local_idx+1];
                    x += SUB_GROUP_SIZE; y += SUB_GROUP_SIZE;
                }
            }

            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                cl::sycl::ext::oneapi::sub_group sg = item.get_sub_group();
                kernel(poly_idx, i, sg);
            }

        protected:
            int position_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            int m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
        };

        template <typename T, int log_Unroll>
        class RnsInvDwtGapLocalRescale {
        public:
            RnsInvDwtGapLocalRescale(int position, int TER_LOCAL_MEM_GAP_SIZE, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                
                position_ = position;
                TER_LOCAL_MEM_GAP_SIZE_ = TER_LOCAL_MEM_GAP_SIZE;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind, uint64_t ind_local, cl::sycl::nd_item<2> &item) const {
                //prologue: load data from the global memory
                //          and store to local memory
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                auto i = (ind >> (log_gap_local - log_Unroll));
                auto j = ind - (i << (log_gap_local - log_Unroll));
                auto i_local = (ind_local >> (log_gap_local - log_Unroll));
                auto j_local = ind_local - (i_local << (log_gap_local - log_Unroll));
                auto r_op = roots_op_[rounds_local + global_offset + i];
                auto r_quo = roots_quo_[rounds_local + global_offset + i];

                std::size_t offset = (i << (log_gap_local + 1)) + (j << log_Unroll);
                std::size_t offset_local = (i_local << (log_gap_local + 1)) + (j_local << log_Unroll);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                auto x = values_ + global_offset + offset + poly_offset;
                auto y = x + gap_local;
                auto local_ptr_x = ptr_.get_pointer() + offset_local;
                auto local_ptr_y = local_ptr_x + gap_local;
                
                for (int k = 0; k < (1<< log_Unroll); k++)
                {
                    auto u = *x++;
                    auto v = *y++;
                    *local_ptr_x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                    *local_ptr_y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);
                }
                item.barrier();
                rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++;
                //loop body: play inside of the local memory
                for (; gap_local <= TER_LOCAL_MEM_GAP_SIZE_; rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++){
                    i = (ind >> (log_gap_local - log_Unroll));
                    i_local = (ind_local >> (log_gap_local - log_Unroll));
                    j = ind_local - (i_local << (log_gap_local - log_Unroll));
                    r_op = roots_op_[rounds_local + global_offset + i];
                    r_quo = roots_quo_[rounds_local + global_offset + i];
                    offset = (i_local << (log_gap_local + 1)) + (j << log_Unroll);
                    local_ptr_x = ptr_.get_pointer() + offset;
                    local_ptr_y = local_ptr_x + gap_local;
                    for (int k = 0; k < (1<< log_Unroll); k++)
                    {
                        auto u = *local_ptr_x;
                        auto v = *local_ptr_y;
                        *local_ptr_x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *local_ptr_y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);
                    }
                    item.barrier();//sync the work-group
                }
                //recover gap and log_gap before entering the epilogue
                gap_local >>= 1; log_gap_local -= 1;
                //epilogue: write data back to global memory
                i = (ind >> (log_gap_local - log_Unroll));
                j = ind - (i << (log_gap_local - log_Unroll));
                i_local = (ind_local >> (log_gap_local - log_Unroll));
                j_local = ind_local - (i_local << (log_gap_local - log_Unroll));
                offset = (i << (log_gap_local + 1)) + (j << log_Unroll);
                offset_local = (i_local << (log_gap_local + 1)) + (j_local << log_Unroll);
                x = values_ + global_offset + offset + poly_offset;
                y = x + gap_local;
                local_ptr_x = ptr_.get_pointer() + offset_local;
                local_ptr_y = local_ptr_x + gap_local;
                
                for (int k = 0; k < (1<< log_Unroll); k++)
                {
                    *x++ = *local_ptr_x++;
                    *y++ = *local_ptr_y++;
                }

            }
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            int position_;
            local_accessor<T, 1> ptr_;
            int TER_LOCAL_MEM_GAP_SIZE_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            std::size_t m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
        };

        template <typename T, int log_Unroll>
        class RnsInvDwtGapLocalFusedRescale {
        public:
            RnsInvDwtGapLocalFusedRescale(int position, int TER_LOCAL_MEM_GAP_SIZE, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo,
                      const T* scalar_op, const T* scalar_quo, local_accessor<T, 1>& ptr, bool lazy = false) : ptr_{ptr}
            {
                
                position_ = position;
                TER_LOCAL_MEM_GAP_SIZE_ = TER_LOCAL_MEM_GAP_SIZE;
                q_base_size_ = q_base_size;
                values_ = values;
                log_n_ = log_n;
                log_m_ = log_m;
                m_ = (1 << log_m_);
                log_gap_ = log_gap;
                gap_ = (1 << log_gap_);
                rounds_ = rounds;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
                lazy_ = lazy;
            }
            // polynomial index, RNS prime base index, coefficient index
            void kernel(uint64_t poly_idx, uint64_t ind, uint64_t ind_local, cl::sycl::nd_item<2> &item) const {
                //prologue: load data from the global memory
                //          and store to local memory
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[position_];
                auto two_times_modulus = (modulus << 1);
                std::size_t global_offset = 0;
                auto i = (ind >> (log_gap_local - log_Unroll));
                auto j = ind - (i << (log_gap_local - log_Unroll));
                auto i_local = (ind_local >> (log_gap_local - log_Unroll));
                auto j_local = ind_local - (i_local << (log_gap_local - log_Unroll));
                auto r_op = roots_op_[rounds_local + global_offset + i];
                auto r_quo = roots_quo_[rounds_local + global_offset + i];

                std::size_t offset = (i << (log_gap_local + 1)) + (j << log_Unroll);
                std::size_t offset_local = (i_local << (log_gap_local + 1)) + (j_local << log_Unroll);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                auto x = values_ + global_offset + offset + poly_offset;
                auto y = x + gap_local;
                auto local_ptr_x = ptr_.get_pointer() + offset_local;
                auto local_ptr_y = local_ptr_x + gap_local;

                for (int k = 0; k < (1<< log_Unroll); k++)
                {
                    auto u = *x++;
                    auto v = *y++;
                    *local_ptr_x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                    *local_ptr_y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);
                }
                item.barrier();
                rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++;
                //loop body: play inside of the local memory
                for (; gap_local <= TER_LOCAL_MEM_GAP_SIZE_; rounds_local += m_local, m_local >>= 1, gap_local <<= 1, log_m_local--, log_gap_local++){
                    i = (ind >> (log_gap_local - log_Unroll));
                    i_local = (ind_local >> (log_gap_local - log_Unroll));
                    j = ind_local - (i_local << (log_gap_local - log_Unroll));
                    r_op = roots_op_[rounds_local + global_offset + i];
                    r_quo = roots_quo_[rounds_local + global_offset + i];
                    offset = (i_local << (log_gap_local + 1)) + (j << log_Unroll);
                    local_ptr_x = ptr_.get_pointer() + offset;
                    local_ptr_y = local_ptr_x + gap_local;
                    for (int k = 0; k < (1<< log_Unroll); k++)
                    {
                        auto u = *local_ptr_x;
                        auto v = *local_ptr_y;
                        *local_ptr_x++ = dwt_guard(dwt_add(u, v), two_times_modulus);
                        *local_ptr_y++ = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op, r_quo, modulus);
                    }
                    item.barrier();//sync the work-group
                }

                i = (ind >> (log_gap_local - log_Unroll));
                i_local = (ind_local >> (log_gap_local - log_Unroll));
                j = ind_local - (i_local << (log_gap_local - log_Unroll));
                r_op = roots_op_[rounds_local + global_offset + i];
                r_quo = roots_quo_[rounds_local + global_offset + i];

                auto scalar_op = scalar_op_[position_];
                auto scalar_quo = scalar_quo_[position_];
                T scaled_r_op;
                T scaled_r_quo;
                dwt_mul_root_scalar(r_op, scalar_op, scalar_quo, modulus, scaled_r_op, scaled_r_quo);

                offset = (i_local << (log_gap_local + 1)) + (j << log_Unroll);
                local_ptr_x = ptr_.get_pointer() + offset;
                local_ptr_y = local_ptr_x + gap_local;
                if (!lazy_){
                    for (int k = 0; k < (1<< log_Unroll); k++)
                    {
                        auto u = dwt_guard(*local_ptr_x, two_times_modulus);
                        auto v = *local_ptr_y;
                        auto uu = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), scalar_op, scalar_quo, modulus);
                        auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r_op, scaled_r_quo, modulus);
                        uu -= (uu >= modulus) ? modulus : 0;
                        vv -= (vv >= modulus) ? modulus : 0;
                        *local_ptr_x++ = uu;
                        *local_ptr_y++ = vv;//dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r_op, scaled_r_quo, modulus);
                    }
                }else{
                    for (int k = 0; k < (1<< log_Unroll); k++)
                    {
                        auto u = dwt_guard(*local_ptr_x, two_times_modulus);
                        auto v = *local_ptr_y;
                        auto uu = dwt_mul_scalar(dwt_guard(dwt_add(u, v), two_times_modulus), scalar_op, scalar_quo, modulus);
                        auto vv = dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r_op, scaled_r_quo, modulus);
                        *local_ptr_x++ = uu;
                        *local_ptr_y++ = vv;//dwt_mul_root(dwt_sub(u, v, two_times_modulus), scaled_r_op, scaled_r_quo, modulus);
                    }
                }
                item.barrier();//sync the work-group

                //epilogue: write data back to global memory
                i = (ind >> (log_gap_local - log_Unroll));
                j = ind - (i << (log_gap_local - log_Unroll));
                i_local = (ind_local >> (log_gap_local - log_Unroll));
                j_local = ind_local - (i_local << (log_gap_local - log_Unroll));
                offset = (i << (log_gap_local + 1)) + (j << log_Unroll);
                offset_local = (i_local << (log_gap_local + 1)) + (j_local << log_Unroll);
                x = values_ + global_offset + offset + poly_offset;
                y = x + gap_local;
                local_ptr_x = ptr_.get_pointer() + offset_local;
                local_ptr_y = local_ptr_x + gap_local;

                for (int k = 0; k < (1<< log_Unroll); k++)
                {
                    *x++ = *local_ptr_x++;
                    *y++ = *local_ptr_y++;
                }

            }
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            int position_;
            local_accessor<T, 1> ptr_;
            int TER_LOCAL_MEM_GAP_SIZE_;
            int q_base_size_;
            std::size_t log_n_;
            std::size_t log_m_;
            std::size_t m_;
            int rounds_;
            std::size_t gap_;
            std::size_t log_gap_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
            bool lazy_;
        };

        // GPU RNS Inverse NTT Class
        template <typename T>
        class InvRnsDwtGpuRescale : public InvRnsDwtRescale<T> {
        public:
            InvRnsDwtGpuRescale(int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                         const T* roots_op, const T* roots_quo,
                         const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                         bool lazy = false, bool wait = false)
                : InvRnsDwtRescale<T>(poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)
            {
                
                poly_num_ = poly_num;
                q_base_size_ = q_base_size;
                log_n_ = log_n;
                values_ = values;
                modulus_ = modulus;
                roots_op_ = roots_op;
                roots_quo_ = roots_quo;
                scalar_op_ = scalar_op;
                scalar_quo_ = scalar_quo;
                lazy_ = lazy;
                wait_ = wait;
            }

            void operator() [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] (cl::sycl::queue& queue) {
                // constant transform size
                size_t n = size_t(1) << log_n_;
                // variables for indexing
                std::size_t gap = 1;
                std::size_t log_gap = 0;
                std::size_t m = n >> 1;
                std::size_t log_m = log_n_ - 1;
                std::size_t total_r = 1;
                const int log_local_unroll = (5 < LOG_TER_GAP_SIZE) ? 5 : LOG_TER_GAP_SIZE;
                if ( (log_m + log_gap <= log_local_unroll) || (log_m + log_gap <= LOG_LOCAL_REG_SLOTS) ){
                    // naive implementation
                    if ( (log_n_ % 3) == 0 ){// radix-8
                        gap = 4; log_gap = 2;
                        for (; log_gap <= log_n_ - 1;total_r +=( m + (m>>1) + (m>>2)), m >>= 3, log_m -= 3, gap <<= 3, log_gap += 3)
                        {
                            auto grid_range = sycl::range<2>(poly_num_ ,(n>>3) );
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                            {
                                h.parallel_for({ grid_range }, RnsInvDwtGapRadix8Rescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                        roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapRadix8Rescale", e);
                        }
                        auto grid_range = sycl::range<2>(poly_num_, (n >> 1));
                        auto e = queue.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarSeperateRescale<T>(0, q_base_size_, log_n_, log_n_ -1, total_r, values_, modulus_,
                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, lazy_));
                        });//.wait();
                        EventCollector::add_event("RnsInvDwtLastRoundScalarSeperateRescale", e);
                    }else if ( (log_n_ & 1) == 0 ){// radix-4
                        gap = 2; log_gap = 1;
                        for (; log_gap <= log_n_ - 1;total_r +=( m + (m>>1) ), m >>= 2, log_m -= 2, gap <<= 2, log_gap += 2)
                        {
                            auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                            {
                                h.parallel_for({ grid_range }, RnsInvDwtGapRadix4Rescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                        roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapRadix4Rescale", e);
                        }
                        auto grid_range = sycl::range<2>(poly_num_, (n >> 1));
                        auto e = queue.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarSeperateRescale<T>(0, q_base_size_, log_n_, log_n_ -1, total_r, values_, modulus_,
                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, lazy_));
                        });//.wait();
                        EventCollector::add_event("RnsInvDwtLastRoundScalarSeperateRescale", e);
                    }else{// radix-2
                        for (; m > 0; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                        {
                            static const int log_unroll = 0;
                            auto grid_range = sycl::range<2>(poly_num_, (1 << (log_m + log_gap - log_unroll)));
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                            {
                                h.parallel_for({ grid_range }, RnsInvDwtGapRescale<T, log_unroll>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                            roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapRescale", e);
                        }
                        auto grid_range = sycl::range<2>(poly_num_, (n >> 1));
                        auto e = queue.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarSeperateRescale<T>(0, q_base_size_, log_n_, log_n_ -1, total_r, values_, modulus_,
                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, lazy_));
                        });//.wait();
                        EventCollector::add_event("RnsInvDwtLastRoundScalarSeperateRescale", e);
                    }
                }else{// normal input: optimized implementation
                    const int MAX_WORK_GROUP_SIZE = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
                    const int LOG_MAX_WORK_GROUP_SIZE = size_t(std::ceil(std::ilogb(MAX_WORK_GROUP_SIZE)));// x86_log2(MAX_WORK_GROUP_SIZE);//log of MAX_WORK_GROUP_SIZE
                    const int LOCAL_MEM_SIZE = queue.get_device().get_info<sycl::info::device::local_mem_size>();
                    const int LOG_LOCAL_MEM_SIZE = size_t(std::ceil(std::ilogb(LOCAL_MEM_SIZE)));// x86_log2(LOCAL_MEM_SIZE);//log of LOCAL_MEM_SIZE
                    if (sizeof(T) == 4){
                        const int MAX_ELE_NUMBER_LOCAL_MEM = (LOCAL_MEM_SIZE >> 2);//LOCAL_MEM_SIZE / sizeof(T)
                        const int LOG_MAX_ELE_NUMBER_LOCAL_MEM = (LOG_LOCAL_MEM_SIZE - 2);//log of MAX_ELE_NUMBER_LOCAL_MEM
                        if ( (MAX_WORK_GROUP_SIZE < 512) || (n >= 65536) || (n <= 2048 && n != 512) ) {// cannot be handled by radix-8 local-mem kernel, degrade to previous implementation
                            int WORK_GROUP_SIZE_LOCAL = (MAX_WORK_GROUP_SIZE < (1 << (log_m + log_gap - log_local_unroll)) ) ?  MAX_WORK_GROUP_SIZE : (1 << (log_m + log_gap - log_local_unroll));
                            int LOG_WORK_GROUP_SIZE_LOCAL = (LOG_MAX_WORK_GROUP_SIZE < (log_m + log_gap - log_local_unroll) ) ?  LOG_MAX_WORK_GROUP_SIZE : (log_m + log_gap - log_local_unroll);
                            if ( (LOG_WORK_GROUP_SIZE_LOCAL + log_local_unroll) >= LOG_MAX_ELE_NUMBER_LOCAL_MEM ) {
                                WORK_GROUP_SIZE_LOCAL = ( WORK_GROUP_SIZE_LOCAL >> (LOG_WORK_GROUP_SIZE_LOCAL + log_local_unroll + 1 - LOG_MAX_ELE_NUMBER_LOCAL_MEM) );
                            }
                            int WORK_GROUP_SIZE_SIMD = (MAX_WORK_GROUP_SIZE < (1 << (log_m + log_gap - LOG_LOCAL_REG_SLOTS)) ) ?  MAX_WORK_GROUP_SIZE : (1 << (log_m + log_gap - LOG_LOCAL_REG_SLOTS));
                            int TER_LOCAL_MEM_GAP_SIZE = (MAX_ELE_NUMBER_LOCAL_MEM < (WORK_GROUP_SIZE_LOCAL << log_local_unroll)) ? MAX_ELE_NUMBER_LOCAL_MEM : (WORK_GROUP_SIZE_LOCAL << log_local_unroll);
                            TER_LOCAL_MEM_GAP_SIZE = ( TER_LOCAL_MEM_GAP_SIZE <= (n >> 2) ) ? TER_LOCAL_MEM_GAP_SIZE : (n >> 2);
                            auto nd_grid_range = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                    static_cast<uint32_t>( (1 << (log_m + log_gap - LOG_LOCAL_REG_SLOTS)))},\
                                                                    {static_cast<uint32_t>(1),\
                                                                    static_cast<uint32_t>(WORK_GROUP_SIZE_SIMD)}\
                                                                );
                            auto e1 = queue.submit([&](cl::sycl::handler& h)
                            {
                                h.parallel_for(nd_grid_range, RnsInvDwtGapSIMDRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                                roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapSIMDRescaleRescale", e1);
                            for (; gap <= TER_GAP_SIZE; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++) ;
                            bool simd_not_overlap_with_local = TER_GAP_SIZE != TER_LOCAL_MEM_GAP_SIZE;
                            bool last_round_in_local_mem = ((TER_LOCAL_MEM_GAP_SIZE << 2) <= MAX_ELE_NUMBER_LOCAL_MEM);
                            bool terminate_at_local_mem = (TER_LOCAL_MEM_GAP_SIZE == (n >> 2));
                            bool unroll_cover_last_round_in_local_kernel = ( (WORK_GROUP_SIZE_LOCAL << log_local_unroll) >= (TER_LOCAL_MEM_GAP_SIZE << 1) );
                            if (terminate_at_local_mem && simd_not_overlap_with_local && last_round_in_local_mem && unroll_cover_last_round_in_local_kernel) {
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (1 << (log_m + log_gap - log_local_unroll)))},\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>(WORK_GROUP_SIZE_LOCAL)}\
                                                                    );
                                auto e2 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalFusedRescale<T, log_local_unroll>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
#if (5 < LOG_TER_GAP_SIZE)
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalFusedRescale<T, 5>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalFusedRescale<T, LOG_TER_GAP_SIZE>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#endif

#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalFusedRescale", e2);
                            }else{
                                if (simd_not_overlap_with_local){
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (1 << (log_m + log_gap - log_local_unroll)))},\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>(WORK_GROUP_SIZE_LOCAL)}\
                                                                    );
                                auto e2 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
    #ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRescale<T, log_local_unroll>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, shared_buffer));
    #else
    #if (5 < LOG_TER_GAP_SIZE) 
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRescale<T, 5>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
    #else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRescale<T, LOG_TER_GAP_SIZE>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
    #endif

    #endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRescale", e2);
                                }
                                for (; gap <= TER_LOCAL_MEM_GAP_SIZE; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++) ;
                                for (; m > 1; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                                {
                                    static const int log_unroll = 0;
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << (log_m + log_gap - log_unroll)));
                                    auto e3 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRescale<T, log_unroll>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRescale", e3);
                                }

                                if (scalar_op_ != nullptr && scalar_quo_ != nullptr)
                                {
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << log_gap));
                                    auto e3 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarRescale<T>(0, q_base_size_, log_n_, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_ , lazy_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtLastRoundScalarRescale", e3);
                                }
                                else
                                {
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << log_gap));
                                    auto e3 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtLastRoundRescale<T>(0, q_base_size_, log_n_, log_gap, total_r, values_, modulus_,
                                                                                            roots_op_, roots_quo_, lazy_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtLastRoundRescale", e3);
                                }
                            }
                        }else{//optimized high-radix kernels
                            gap = 4; log_gap = 2;
                            if (n == 4096 || n == 512){
                                int WORK_GROUP_SIZE_LOCAL = n < 4096 ? 64 : 512;
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( n >> 3 ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( WORK_GROUP_SIZE_LOCAL )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8FusedRescale", e);
                            }else{
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (n >> 3) ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8Rescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8Rescale", e);
                                total_r += m + (m>>1) + (m>>2); m >>= 3; log_m -= 3;gap <<= 1; log_gap += 1;
                                for (; log_gap <= 11;total_r +=m, m >>= 1, log_m -= 1, gap <<= 1, log_gap += 1) ;
                                if (n == 8192){
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>1) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix2FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRescale", e1);
                                }else if (n == 16384){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4FusedRescale", e1); 
                                }else if (n == 32768){
                                    gap <<= 2; log_gap += 2;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>3) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix8FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix8FusedRescale", e1); 
                                }

                            }


                        }

                    }else if (sizeof(T) == 8){
                        const int MAX_ELE_NUMBER_LOCAL_MEM = (LOCAL_MEM_SIZE >> 3);//LOCAL_MEM_SIZE / sizeof(T)
                        const int LOG_MAX_ELE_NUMBER_LOCAL_MEM = (LOG_LOCAL_MEM_SIZE - 3);//log of MAX_ELE_NUMBER_LOCAL_MEM
#if _SIMD_WIDTH_==8
                        if ( (MAX_WORK_GROUP_SIZE < 512) || (n >= 65536) || (n <= 2048 && n != 512) ) {// cannot be handled by radix-8 local-mem kernel, degrade to previous implementation
#else
                        if ( (MAX_WORK_GROUP_SIZE < 512) || (n >= 65536) || (n <= 2048) ) {// cannot be handled by radix-8 local-mem kernel, degrade to previous implementation
#endif
                            int WORK_GROUP_SIZE_LOCAL = (MAX_WORK_GROUP_SIZE < (1 << (log_m + log_gap - log_local_unroll)) ) ?  MAX_WORK_GROUP_SIZE : (1 << (log_m + log_gap - log_local_unroll));
                            int LOG_WORK_GROUP_SIZE_LOCAL = (LOG_MAX_WORK_GROUP_SIZE < (log_m + log_gap - log_local_unroll) ) ?  LOG_MAX_WORK_GROUP_SIZE : (log_m + log_gap - log_local_unroll);
                            if ( (LOG_WORK_GROUP_SIZE_LOCAL + log_local_unroll) >= LOG_MAX_ELE_NUMBER_LOCAL_MEM ) {
                                WORK_GROUP_SIZE_LOCAL = ( WORK_GROUP_SIZE_LOCAL >> (LOG_WORK_GROUP_SIZE_LOCAL + log_local_unroll + 1 - LOG_MAX_ELE_NUMBER_LOCAL_MEM) );
                            }
                            int WORK_GROUP_SIZE_SIMD = (MAX_WORK_GROUP_SIZE < (1 << (log_m + log_gap - LOG_LOCAL_REG_SLOTS)) ) ?  MAX_WORK_GROUP_SIZE : (1 << (log_m + log_gap - LOG_LOCAL_REG_SLOTS));
                            int TER_LOCAL_MEM_GAP_SIZE = (MAX_ELE_NUMBER_LOCAL_MEM < (WORK_GROUP_SIZE_LOCAL << log_local_unroll)) ? MAX_ELE_NUMBER_LOCAL_MEM : (WORK_GROUP_SIZE_LOCAL << log_local_unroll);
                            TER_LOCAL_MEM_GAP_SIZE = ( TER_LOCAL_MEM_GAP_SIZE <= (n >> 2) ) ? TER_LOCAL_MEM_GAP_SIZE : (n >> 2);
                            auto nd_grid_range = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                    static_cast<uint32_t>( (1 << (log_m + log_gap - LOG_LOCAL_REG_SLOTS)))},\
                                                                    {static_cast<uint32_t>(1),\
                                                                    static_cast<uint32_t>(WORK_GROUP_SIZE_SIMD)}\
                                                                );
                            auto e1 = queue.submit([&](cl::sycl::handler& h)
                            {
                                h.parallel_for(nd_grid_range, RnsInvDwtGapSIMDRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                                roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapSIMDRescale", e1);
                            for (; gap <= TER_GAP_SIZE; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++) ;
                            bool simd_not_overlap_with_local = TER_GAP_SIZE != TER_LOCAL_MEM_GAP_SIZE;
                            bool last_round_in_local_mem = ((TER_LOCAL_MEM_GAP_SIZE << 2) <= MAX_ELE_NUMBER_LOCAL_MEM);
                            bool terminate_at_local_mem = (TER_LOCAL_MEM_GAP_SIZE == (n >> 2));
                            bool unroll_cover_last_round_in_local_kernel = ( (WORK_GROUP_SIZE_LOCAL << log_local_unroll) >= (TER_LOCAL_MEM_GAP_SIZE << 1) );
                            if (terminate_at_local_mem && simd_not_overlap_with_local && last_round_in_local_mem && unroll_cover_last_round_in_local_kernel) {
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (1 << (log_m + log_gap - log_local_unroll)))},\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>(WORK_GROUP_SIZE_LOCAL)}\
                                                                    );
                                auto e2 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalFusedRescale<T, log_local_unroll>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
#if (5 < LOG_TER_GAP_SIZE)
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalFusedRescale<T, 5>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalFusedRescale<T, LOG_TER_GAP_SIZE>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#endif

#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalFusedRescale", e2);

                            }else{
                                if (simd_not_overlap_with_local){
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (1 << (log_m + log_gap - log_local_unroll)))},\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>(WORK_GROUP_SIZE_LOCAL)}\
                                                                    );
                                auto e2 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRescale<T, log_local_unroll>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, shared_buffer));
#else
#if (5 < LOG_TER_GAP_SIZE) 
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRescale<T, 5>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRescale<T, LOG_TER_GAP_SIZE>(0, TER_LOCAL_MEM_GAP_SIZE, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
#endif

#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRescale", e2);
                                }
                                for (; gap <= TER_LOCAL_MEM_GAP_SIZE; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++) ;
                                for (; m > 1; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                                {
                                    static const int log_unroll = 0;
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << (log_m + log_gap - log_unroll)));
                                    auto e3 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRescale<T, log_unroll>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRescale", e3);
                                }

                                if (scalar_op_ != nullptr && scalar_quo_ != nullptr)
                                {
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << log_gap));
                                    auto e3 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarRescale<T>(0, q_base_size_, log_n_, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_ , lazy_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtLastRoundScalarRescale", e3);
                                }
                                else
                                {
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << log_gap));
                                    auto e3 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtLastRoundRescale<T>(0, q_base_size_, log_n_, log_gap, total_r, values_, modulus_,
                                                                                            roots_op_, roots_quo_, lazy_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtLastRoundRescale", e3);
                                }
                            }
                        }else{//optimized high-radix kernels
#if _SIMD_WIDTH_==8
                            gap = 4; log_gap = 2;
                            if (n == 4096 || n == 512){
                                int WORK_GROUP_SIZE_LOCAL = n < 4096 ? 64 : 512;
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( n >> 3 ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( WORK_GROUP_SIZE_LOCAL )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8FusedRescale", e);
                            }else{
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (n >> 3) ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8Rescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8Rescale", e);
                                total_r += m + (m>>1) + (m>>2); m >>= 3; log_m -= 3;gap <<= 1; log_gap += 1;
                                for (; log_gap <= 11;total_r +=m, m >>= 1, log_m -= 1, gap <<= 1, log_gap += 1) ;
                                if (n == 8192){
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>1) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix2FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRescale", e1);
                                }else if (n == 16384){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4FusedRescale", e1); 
                                }else if (n == 32768){
                                    gap <<= 2; log_gap += 2;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>3) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix8FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix8FusedRescale", e1); 
                                }

                            }

#else//SIMD16

                            gap = 2; log_gap = 1;
                            if (n == 4096){
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( n >> 3 ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix4FusedRescale<T, 2, 1>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix4FusedRescale", e);
                                
                            }else{
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (n >> 3) ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix4Rescale<T, 2, 1>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix4Rescale", e);
                                total_r += m + (m>>1); m >>= 2; log_m -= 2;gap <<= 1; log_gap += 1;
                                for (; log_gap <= 11;total_r +=m, m >>= 1, log_m -= 1, gap <<= 1, log_gap += 1) ;
                                if (n == 8192){
                                    auto grid_range = sycl::range<2>(poly_num_, (n>>1) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix2FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRescale", e1);
                                }else if (n == 16384){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_, (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4FusedRescale", e1); 
                                }else if (n == 32768){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_, (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4Rescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4Rescale", e1);
                                    total_r += m + (m>>1); m >>= 2; log_m -= 2;gap <<= 1; log_gap += 1;

                                    auto grid_range1 = sycl::range<2>(poly_num_, (n>>1) );
                                    auto e2 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range1 }, RnsInvDwtGapRadix2FusedRescale<T>(0, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRescale", e2);
                                }
                            }

#endif
                        }
                    }
                }
                if (wait_){
                    queue.wait();
                }

            }

        protected:
            int poly_num_;
            int q_base_size_;
            int log_n_;
            T* values_;
            const T* modulus_;
            const T* roots_op_;
            const T* roots_quo_;
            const T* scalar_op_;
            const T* scalar_quo_;
            bool lazy_;
            bool wait_;
        };

#endif

    } // namespace util
} // namespace xehe

#endif //#ifndef _RNS_INVDWT_RESCALE_GPU_HPP_


