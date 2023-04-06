/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef _RNS_INVDWT_RELIN_GPU_HPP_
#define _RNS_INVDWT_RELIN_GPU_HPP_

#include "rns_invdwt_gpu.hpp"

template <typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write, sycl::access::target::local>;

namespace xehe
{
    namespace util
    {
        // inverse NTT kernel for relin
        template <typename T, int log_Unroll>
        class RnsInvDwtGapRelin {
        public:
            RnsInvDwtGapRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds,
                         T* values, const T* modulus,
                         const T* roots_op,
                         const T* roots_quo)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);

                auto i = (ind >> (log_gap_ - log_Unroll));
                auto j = ind - (i << (log_gap_ - log_Unroll));

                auto r_op = roots_op_[rounds_ + (kk<<log_n_) + i];
                auto r_quo = roots_quo_[rounds_+ (kk<<log_n_) + i];
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
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int key_modulus_size_;
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
        class RnsInvDwtGapRadix4Relin {
        public:
            RnsInvDwtGapRadix4Relin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto modulus = modulus_[kk];
                auto m_ = (1<<log_m_);
                auto two_times_modulus = (modulus << 1);
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

                ld_roots_relin_2(roots_op_, r_op, (rounds_))
                ld_roots_relin_2(roots_quo_, r_quo, (rounds_))

                init_global_ptr_inv_relin_4((gap_>>1))
                ld_global_mem_to_reg_4(0)

                // round 1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)

                // round 2
                compute_ind_2(log_gap_)
                ld_roots_relin_2(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_relin_2(roots_quo_, r_quo, (rounds_+ m_ ))

                butterfly_inv_ntt_reg2gmem(0, 2, 0)
                butterfly_inv_ntt_reg2gmem(1, 3, 1)
            }

            
            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::id<2> ind) const {
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int key_modulus_size_;
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
        class RnsInvDwtGapRadix8Relin {
        public:
            RnsInvDwtGapRadix8Relin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
                auto m_ = (1 << log_m_);
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

                ld_roots_relin_4(roots_op_, r_op, rounds_)
                ld_roots_relin_4(roots_quo_, r_quo, rounds_)
                init_global_ptr_inv_relin_8((gap_>>2))
                ld_global_mem_to_reg_8(0)

                // round1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)


                // inner round 2
                compute_ind_4(log_gap_-1)
                ld_roots_relin_4(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_+ m_ ))
                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_)
                ld_roots_relin_4(roots_op_, r_op, (rounds_+ m_+ (m_>>1) ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_+ m_ + (m_>>1) ))

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
            int key_modulus_size_;
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
        class RnsInvDwtLastRoundScalarSeperateRelin {
        public:
            RnsInvDwtLastRoundScalarSeperateRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_gap, int rounds, T* values, const T* modulus,
                                     const T* roots_op, const T* roots_quo,
                                     const T* scalar_op, const T* scalar_quo, bool lazy = false)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto modulus = modulus_[kk];
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
                auto two_times_modulus = (modulus << 1);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);


                auto x = values_ + 2 * i + poly_offset;
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
            int key_modulus_size_;
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

        template <typename T>
        class RnsInvDwtGapRadix2FusedRelin {
        public:
            RnsInvDwtGapRadix2FusedRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds,
                         T* values, const T* modulus,
                         const T* roots_op,
                         const T* roots_quo,
                         const T* scalar_op, const T* scalar_quo)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];

                auto i = (ind >> (log_gap_));
                auto j = ind - (i << (log_gap_));

                auto r_op = roots_op_[rounds_ + (kk<<log_n_) + i];
                auto r_quo = roots_quo_[rounds_ + (kk<<log_n_) + i];
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
            int key_modulus_size_;
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
        class RnsInvDwtGapRadix4FusedRelin {
        public:
            RnsInvDwtGapRadix4FusedRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto modulus = modulus_[kk];
                auto m_ = (1<<log_m_);
                auto two_times_modulus = (modulus << 1);
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
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

                ld_roots_relin_2(roots_op_, r_op, (rounds_))
                ld_roots_relin_2(roots_quo_, r_quo, (rounds_))

                init_global_ptr_inv_relin_4((gap_>>1))
                ld_global_mem_to_reg_4(0)

                // round 1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)

                // round 2
                compute_ind_2(log_gap_)
                ld_roots_relin_2(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_relin_2(roots_quo_, r_quo, (rounds_+ m_ ))

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
            int key_modulus_size_;
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
        class RnsInvDwtGapRadix8FusedRelin {
        public:
            RnsInvDwtGapRadix8FusedRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo)
            {
                key_modulus_size_ = key_modulus_size;
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

                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
                auto m_ = (1 << log_m_);
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
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

                ld_roots_relin_4(roots_op_, r_op, rounds_)
                ld_roots_relin_4(roots_quo_, r_quo, rounds_)
                init_global_ptr_inv_relin_8((gap_>>2))
                ld_global_mem_to_reg_8(0)

                // round1
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)


                // inner round 2
                compute_ind_4(log_gap_-1)
                ld_roots_relin_4(roots_op_, r_op, (rounds_+ m_ ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_+ m_ ))
                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_)
                ld_roots_relin_4(roots_op_, r_op, (rounds_+ m_+ (m_>>1) ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_+ m_ + (m_>>1) ))

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
            int key_modulus_size_;
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

        // last round scalar kernel for relin
        template <typename T>
        class RnsInvDwtLastRoundScalarRelin {
        public:
            RnsInvDwtLastRoundScalarRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_gap, int rounds, T* values, const T* modulus,
                                     const T* roots_op, const T* roots_quo,
                                     const T* scalar_op, const T* scalar_quo, bool lazy = false)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto modulus = modulus_[kk];
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
                auto two_times_modulus = (modulus << 1);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                auto r_op = roots_op_[rounds_ + (kk<<log_n_)];
                T scaled_r_op;
                T scaled_r_quo;
                dwt_mul_root_scalar(r_op, scalar_op, scalar_quo, modulus, scaled_r_op, scaled_r_quo);

                auto x = values_ + i + poly_offset;
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
                uint64_t i = ind[1];
                uint64_t poly_idx = ind[0];
                kernel(poly_idx, i);
            }

        protected:
            int key_modulus_size_;
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

        // CPU RNS Inverse NTT Class for relin
        template <typename T>
        class InvRnsDwtRelin{
        public:
            InvRnsDwtRelin(int key_modulus_size, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo,
                      const T* scalar_op = nullptr, const T* scalar_quo = nullptr, bool lazy = false)
            {
                key_modulus_size_ = key_modulus_size;
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
                    if (gap < 4)
                    {
                        for (std::size_t i = 0; i < poly_num_; ++i) {
                            for (uint64_t j = 0; j < (1 << (log_m + log_gap)); ++j){
                                RnsInvDwtGapRelin<T, 0>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r,
                                                         values_, modulus_,
                                                         roots_op_, roots_quo_).kernel(i, j);
                            }
                        }
                    }
                    else
                    {
                        static const int log_unroll = 2;

                        for (std::size_t i = 0; i < poly_num_; ++i) {
                            for (uint64_t j = 0; j < (1 << (log_m + log_gap - log_unroll)); ++j){
                                RnsInvDwtGapRelin<T, log_unroll>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r,
                                                                  values_, modulus_,
                                                                  roots_op_, roots_quo_).kernel(i, j);
                            }
                        }

                    }
                }
                

                for (std::size_t i = 0; i < poly_num_; ++i) {
                    for (uint64_t j = 0; j < (1 << log_gap); ++j){
                        RnsInvDwtLastRoundScalarRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_gap,
                                                    total_r, values_,
                                                    modulus_, roots_op_, roots_quo_,
                                                    scalar_op_, scalar_quo_,
                                                    lazy_).kernel(i, j);
                    }

                }


            }

        protected:
            int key_modulus_size_;
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

        template <typename T>
        class RnsInvDwtGapLocalRadix8FusedRelin {
        public:
            RnsInvDwtGapLocalRadix8FusedRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto n = (1<<log_n_);
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
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

                ld_roots_relin_4(roots_op_, r_op, rounds_local)
                ld_roots_relin_4(roots_quo_, r_quo, rounds_local)

                init_global_ptr_inv_relin_8((gap_local>>2))
                init_local_ptr_8((gap_local>>2))

                ld_global_mem_to_reg_8(0)

                // round1 
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)

                // inner round 2
                compute_ind_4(log_gap_local-1)
                ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_local)
                ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local+ (m_local>>1) ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local+ (m_local>>1) ))
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

                    ld_roots_relin_4(roots_op_, r_op, rounds_local)
                    ld_roots_relin_4(roots_quo_, r_quo, rounds_local)
                    init_local_ptr_8((gap_local>>2))
                    ld_local_mem_to_reg_8(0)
                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 5, 2)
                    butterfly_inv_ntt_reg2reg(6, 7, 3)

                    // inner round 2
                    compute_ind_4(log_gap_local-1)
                    ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                    butterfly_inv_ntt_reg2reg(0, 2, 0)
                    butterfly_inv_ntt_reg2reg(1, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 6, 2)
                    butterfly_inv_ntt_reg2reg(5, 7, 3)

                    // inner round 3
                    compute_ind_4(log_gap_local)
                    ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local + (m_local>>1) ))
                    ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local + (m_local>>1) ))

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
                init_global_ptr_inv_relin_8((gap_local>>2))
                st_reg_to_global_mem_8(0)

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            local_accessor<T, 1> ptr_;
            int key_modulus_size_;
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
        class RnsInvDwtGapLocalRadix8Relin {
        public:
            RnsInvDwtGapLocalRadix8Relin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
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

                ld_roots_relin_4(roots_op_, r_op, rounds_local)
                ld_roots_relin_4(roots_quo_, r_quo, rounds_local)

                init_global_ptr_inv_relin_8((gap_local>>2))
                init_local_ptr_8((gap_local>>2))

                ld_global_mem_to_reg_8(0)

                // round1 
                butterfly_inv_ntt_reg2reg(0, 1, 0)
                butterfly_inv_ntt_reg2reg(2, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 5, 2)
                butterfly_inv_ntt_reg2reg(6, 7, 3)

                // inner round 2
                compute_ind_4(log_gap_local-1)
                ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                butterfly_inv_ntt_reg2reg(0, 2, 0)
                butterfly_inv_ntt_reg2reg(1, 3, 1)
                butterfly_inv_ntt_reg2reg(4, 6, 2)
                butterfly_inv_ntt_reg2reg(5, 7, 3)

                // inner round 3
                compute_ind_4(log_gap_local)
                ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local+ (m_local>>1) ))
                ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local+ (m_local>>1) ))
                butterfly_inv_ntt_reg2lmem(0, 4, 0)
                butterfly_inv_ntt_reg2lmem(1, 5, 1)
                butterfly_inv_ntt_reg2lmem(2, 6, 2)
                butterfly_inv_ntt_reg2lmem(3, 7, 3)

                item.barrier();
                rounds_local +=( m_local + (m_local>>1) + (m_local>>2)), m_local >>= 3, log_m_local -= 3, gap_local <<= 3, log_gap_local += 3;
                //loop body: play inside of the local memory
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

                    ld_roots_relin_4(roots_op_, r_op, rounds_local)
                    ld_roots_relin_4(roots_quo_, r_quo, rounds_local)
                    init_local_ptr_8((gap_local>>2))
                    ld_local_mem_to_reg_8(0)
                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 5, 2)
                    butterfly_inv_ntt_reg2reg(6, 7, 3)

                    // inner round 2
                    compute_ind_4(log_gap_local-1)
                    ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local ))

                    butterfly_inv_ntt_reg2reg(0, 2, 0)
                    butterfly_inv_ntt_reg2reg(1, 3, 1)
                    butterfly_inv_ntt_reg2reg(4, 6, 2)
                    butterfly_inv_ntt_reg2reg(5, 7, 3)

                    // inner round 3
                    compute_ind_4(log_gap_local)
                    ld_roots_relin_4(roots_op_, r_op, (rounds_local+ m_local + (m_local>>1) ))
                    ld_roots_relin_4(roots_quo_, r_quo, (rounds_local+ m_local + (m_local>>1) ))

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
                init_global_ptr_inv_relin_8((gap_local>>2))
                st_reg_to_global_mem_8(0)

            }


            void operator()[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]](cl::sycl::nd_item<2> item) const {
                uint64_t poly_idx = item.get_global_id()[0];
                uint64_t i = item.get_global_id()[1];
                uint64_t ind_local = item.get_local_id()[1];
                kernel(poly_idx, i, ind_local, item);
            }

        protected:
            local_accessor<T, 1> ptr_;
            int key_modulus_size_;
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
        class RnsInvDwtGapLocalRadix4FusedRelin {
        public:
            RnsInvDwtGapLocalRadix4FusedRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, const T* scalar_op, const T* scalar_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto n = (1<<log_n_);
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
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

                    ld_roots_relin_2(roots_op_, r_op, rounds_local)
                    ld_roots_relin_2(roots_quo_, r_quo, rounds_local)

                    init_global_ptr_inv_relin_4((gap_local>>1))
                    init_local_ptr_4((gap_local>>1))

                    ld_global_mem_to_reg_4(0)

                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)

                    // round2
                    compute_ind_2(log_gap_local)
                    ld_roots_relin_2(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_relin_2(roots_quo_, r_quo, (rounds_local+ m_local ))

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

                        ld_roots_relin_2(roots_op_, r_op, rounds_local)
                        ld_roots_relin_2(roots_quo_, r_quo, rounds_local)
                        init_local_ptr_4((gap_local>>1))
                        ld_local_mem_to_reg_4(0)
                        // round1 
                        butterfly_inv_ntt_reg2reg(0, 1, 0)
                        butterfly_inv_ntt_reg2reg(2, 3, 1)

                        // round2
                        compute_ind_2(log_gap_local)
                        ld_roots_relin_2(roots_op_, r_op, (rounds_local+ m_local ))
                        ld_roots_relin_2(roots_quo_, r_quo, (rounds_local+ m_local ))

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
                    init_global_ptr_inv_relin_4((gap_local>>1))
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
            local_accessor<T, 1> ptr_;
            int key_modulus_size_;
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
        class RnsInvDwtGapLocalRadix4Relin {
        public:
            RnsInvDwtGapLocalRadix4Relin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto kk_mul_n = (kk << log_n_);
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
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

                    ld_roots_relin_2(roots_op_, r_op, rounds_local)
                    ld_roots_relin_2(roots_quo_, r_quo, rounds_local)

                    init_global_ptr_inv_relin_4((gap_local>>1))
                    init_local_ptr_4((gap_local>>1))

                    ld_global_mem_to_reg_4(0)

                    // round1 
                    butterfly_inv_ntt_reg2reg(0, 1, 0)
                    butterfly_inv_ntt_reg2reg(2, 3, 1)

                    // round2
                    compute_ind_2(log_gap_local)
                    ld_roots_relin_2(roots_op_, r_op, (rounds_local+ m_local ))
                    ld_roots_relin_2(roots_quo_, r_quo, (rounds_local+ m_local ))

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

                        ld_roots_relin_2(roots_op_, r_op, rounds_local)
                        ld_roots_relin_2(roots_quo_, r_quo, rounds_local)
                        init_local_ptr_4((gap_local>>1))
                        ld_local_mem_to_reg_4(0)
                        // round1 
                        butterfly_inv_ntt_reg2reg(0, 1, 0)
                        butterfly_inv_ntt_reg2reg(2, 3, 1)

                        // round2
                        compute_ind_2(log_gap_local)
                        ld_roots_relin_2(roots_op_, r_op, (rounds_local+ m_local ))
                        ld_roots_relin_2(roots_quo_, r_quo, (rounds_local+ m_local ))

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
                    init_global_ptr_inv_relin_4((gap_local>>1))
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
            local_accessor<T, 1> ptr_;
            int key_modulus_size_;
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

        template <typename T>
        class RnsInvDwtGapSimdRelin {
        public:
            RnsInvDwtGapSimdRelin(int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo)
            {
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                T data[LOCAL_REG_NUMBERS];
                T r_op[LOCAL_REG_SLOTS];
                T r_quo[LOCAL_REG_SLOTS];
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
                auto i = (idx >> LOG_SUB_GROUP_SIZE);
                auto idx_r_op_offset = ( ((i << LOG_TER_GAP_SIZE) + (idx & (SUB_GROUP_SIZE - 1))) >> log_gap_local );
                auto j = (idx & (SUB_GROUP_SIZE - 1));
                auto r_op_quo_base = rounds_local + (kk<<log_n_);
                auto r_op_quo_base_offset = r_op_quo_base + idx_r_op_offset;
                std::size_t offset = ((i << LOG_TER_GAP_SIZE) << 1) + (j<<1);
                std::size_t poly_offset = ((q_base_size_ *poly_idx) << log_n_);
                auto x = values_ + offset + poly_offset;
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
                    r_op_quo_base = rounds_local + (kk<<log_n_);
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
                    r_op_quo_base = rounds_local + (kk<<log_n_);
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
                x = values_ + offset + poly_offset;
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
            int key_modulus_size_;
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
        class RnsInvDwtGapLocalRelin {
        public:
            RnsInvDwtGapLocalRelin(int TER_LOCAL_MEM_GAP_SIZE, int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo, local_accessor<T, 1>& ptr) : ptr_{ptr}
            {
                TER_LOCAL_MEM_GAP_SIZE_ = TER_LOCAL_MEM_GAP_SIZE;
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
                auto i = (ind >> (log_gap_local - log_Unroll));
                auto j = ind - (i << (log_gap_local - log_Unroll));
                auto i_local = (ind_local >> (log_gap_local - log_Unroll));
                auto j_local = ind_local - (i_local << (log_gap_local - log_Unroll));
                auto r_op = roots_op_[rounds_local + (kk<<log_n_) + i];
                auto r_quo = roots_quo_[rounds_local + (kk<<log_n_) + i];

                std::size_t offset = (i << (log_gap_local + 1)) + (j << log_Unroll);
                std::size_t offset_local = (i_local << (log_gap_local + 1)) + (j_local << log_Unroll);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                auto x = values_ + offset + poly_offset;
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
                    r_op = roots_op_[rounds_local + (kk<<log_n_) + i];
                    r_quo = roots_quo_[rounds_local + (kk<<log_n_) + i];
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
                x = values_ + offset + poly_offset;
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
            local_accessor<T, 1> ptr_;
            int TER_LOCAL_MEM_GAP_SIZE_;
            int key_modulus_size_;
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
        class RnsInvDwtGapLocalRelinFused {
        public:
            RnsInvDwtGapLocalRelinFused(int TER_LOCAL_MEM_GAP_SIZE, int key_modulus_size, int q_base_size, std::size_t log_n, std::size_t log_m, std::size_t log_gap, int rounds, T* values, const T* modulus,
                      const T* roots_op, const T* roots_quo,
                      const T* scalar_op, const T* scalar_quo, local_accessor<T, 1>& ptr, bool lazy = false) : ptr_{ptr}
            {
                TER_LOCAL_MEM_GAP_SIZE_ = TER_LOCAL_MEM_GAP_SIZE;
                key_modulus_size_ = key_modulus_size;
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
                uint64_t kk =  key_modulus_size_ - 1;
                auto m_local = m_;
                auto log_gap_local = log_gap_;
                auto log_m_local = log_m_;
                auto gap_local = gap_;
                auto rounds_local = rounds_;
                auto modulus = modulus_[kk];
                auto two_times_modulus = (modulus << 1);
                auto i = (ind >> (log_gap_local - log_Unroll));
                auto j = ind - (i << (log_gap_local - log_Unroll));
                auto i_local = (ind_local >> (log_gap_local - log_Unroll));
                auto j_local = ind_local - (i_local << (log_gap_local - log_Unroll));
                auto r_op = roots_op_[rounds_local + (kk<<log_n_) + i];
                auto r_quo = roots_quo_[rounds_local + (kk<<log_n_) + i];

                std::size_t offset = (i << (log_gap_local + 1)) + (j << log_Unroll);
                std::size_t offset_local = (i_local << (log_gap_local + 1)) + (j_local << log_Unroll);
                std::size_t poly_offset = ((q_base_size_ * poly_idx) << log_n_);
                auto x = values_ + offset + poly_offset;
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
                    r_op = roots_op_[rounds_local + (kk<<log_n_) + i];
                    r_quo = roots_quo_[rounds_local + (kk<<log_n_) + i];
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
                r_op = roots_op_[rounds_local + (kk<<log_n_) + i];
                r_quo = roots_quo_[rounds_local + (kk<<log_n_) + i];

                auto scalar_op = scalar_op_[kk];
                auto scalar_quo = scalar_quo_[kk];
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
                x = values_ + offset + poly_offset;
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
            local_accessor<T, 1> ptr_;
            int TER_LOCAL_MEM_GAP_SIZE_;
            int key_modulus_size_;
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

        template <typename T>
        class InvRnsDwtGpuRelin : public InvRnsDwtRelin<T> {
        public:
            InvRnsDwtGpuRelin(int key_modulus_size, int poly_num, int q_base_size, int log_n, T* values, const T* modulus,
                         const T* roots_op, const T* roots_quo,
                         const T* scalar_op = nullptr, const T* scalar_quo = nullptr,
                         bool lazy = false, bool wait = false)
                : InvRnsDwtRelin<T>(key_modulus_size, poly_num, q_base_size, log_n, values, modulus, roots_op, roots_quo, scalar_op, scalar_quo, lazy)
            {
                key_modulus_size_ = key_modulus_size;
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
                    //naive implementation
                    if ( (log_n_ % 3) == 0 ){// radix-8
                        gap = 4; log_gap = 2;
                        for (; log_gap <= log_n_ - 1;total_r +=( m + (m>>1) + (m>>2)), m >>= 3, log_m -= 3, gap <<= 3, log_gap += 3)
                        {
                            auto grid_range = sycl::range<2>(poly_num_ ,(n>>3) );
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                            {
                                h.parallel_for({ grid_range }, RnsInvDwtGapRadix8Relin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                        roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapRadix8Relin", e);
                        }
                        auto grid_range = sycl::range<2>(poly_num_, (n >> 1));
                        auto e = queue.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarSeperateRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_n_ -1, total_r, values_, modulus_,
                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, lazy_));
                        });//.wait();
                        EventCollector::add_event("RnsInvDwtLastRoundScalarSeperateRelin", e);
                    }else if ( (log_n_ & 1) == 0 ){// radix-4
                        gap = 2; log_gap = 1;
                        for (; log_gap <= log_n_ - 1;total_r +=( m + (m>>1) ), m >>= 2, log_m -= 2, gap <<= 2, log_gap += 2)
                        {
                            auto grid_range = sycl::range<2>(poly_num_ ,(n>>2) );
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                            {
                                h.parallel_for({ grid_range }, RnsInvDwtGapRadix4Relin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                        roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapRadix4Relin", e);
                        }
                        auto grid_range = sycl::range<2>(poly_num_, (n >> 1));
                        auto e = queue.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarSeperateRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_n_ -1, total_r, values_, modulus_,
                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, lazy_));
                        });//.wait();
                        EventCollector::add_event("RnsInvDwtLastRoundScalarSeperateRelin", e);
                    }else{//radix-2
                        for (; m > 0; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                        {
                            std::cout<<"n ="<<n<<", gap = "<<gap<<", m  = "<<m<<", total_r = "<<total_r<<std::endl;
                            static const int log_unroll = 0;
                            auto grid_range = sycl::range<2>(poly_num_, (1 << (log_m + log_gap - log_unroll)));
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                            {
                                h.parallel_for({ grid_range }, RnsInvDwtGapRelin<T, log_unroll>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                            roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapRelin", e);
                        }
                        auto grid_range = sycl::range<2>(poly_num_, (n >> 1));
                        auto e = queue.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarSeperateRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_n_ -1, total_r, values_, modulus_,
                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, lazy_));
                        });//.wait();
                        EventCollector::add_event("RnsInvDwtLastRoundScalarSeperateRelin", e);
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
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            {
                                h.parallel_for(nd_grid_range, RnsInvDwtGapSimdRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                                roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapSimdRelin", e);
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
                                auto e1 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelinFused<T, log_local_unroll>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
#if (5 < LOG_TER_GAP_SIZE)
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelinFused<T, 5>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelinFused<T, LOG_TER_GAP_SIZE>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#endif
#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRelinFused", e1);
                            }else{
                                if (simd_not_overlap_with_local){
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (1 << (log_m + log_gap - log_local_unroll)))},\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>(WORK_GROUP_SIZE_LOCAL)}\
                                                                    );
                                auto e1 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelin<T, log_local_unroll>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, shared_buffer));
#else
#if (5 < LOG_TER_GAP_SIZE)
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelin<T, 5>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelin<T, LOG_TER_GAP_SIZE>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
#endif
#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRelin", e1);
                                }
                                for (; gap <= TER_LOCAL_MEM_GAP_SIZE; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++) ;
                                for (; m > 1; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                                {
                                    static const int log_unroll = 0;
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << (log_m + log_gap - log_unroll)));
                                    auto e2 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRelin<T, log_unroll>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRelin", e2);

                                }

                                auto grid_range = sycl::range<2>(poly_num_, (1 << log_gap));
                                auto e2 = queue.submit([&](cl::sycl::handler& h)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                {
                                    h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_ , lazy_));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtLastRoundScalarRelin", e2);
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
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8FusedRelin", e);
                            }else{
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (n >> 3) ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8Relin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8Relin", e);
                                total_r += m + (m>>1) + (m>>2); m >>= 3; log_m -= 3;gap <<= 1; log_gap += 1;
                                for (; log_gap <= 11;total_r +=m, m >>= 1, log_m -= 1, gap <<= 1, log_gap += 1) ;
                                if (n == 8192){
                                    // std::cout<<"n ="<<n<<", gap = "<<gap<<", m  = "<<m<<", total_r = "<<total_r<<std::endl;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>1) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix2FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRelin", e1);
                                }else if (n == 16384){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4FusedRelin", e1); 
                                }else if (n == 32768){
                                    gap <<= 2; log_gap += 2;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>3) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix8FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix8FusedRelin", e1);
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
                            auto e = queue.submit([&](cl::sycl::handler& h)
                            {
                                h.parallel_for(nd_grid_range, RnsInvDwtGapSimdRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                                roots_op_, roots_quo_));
                            });//.wait();
                            EventCollector::add_event("RnsInvDwtGapSimdRelin", e);
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
                                auto e1 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelinFused<T, log_local_unroll>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
#if (5 < LOG_TER_GAP_SIZE)
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelinFused<T, 5>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelinFused<T, LOG_TER_GAP_SIZE>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer, lazy_));
#endif
#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRelinFused", e1);
                            }else{
                                if (simd_not_overlap_with_local){
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (1 << (log_m + log_gap - log_local_unroll)))},\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>(WORK_GROUP_SIZE_LOCAL)}\
                                                                    );
                                auto e1 = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(MAX_ELE_NUMBER_LOCAL_MEM, h);
#ifndef WIN32
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelin<T, log_local_unroll>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                            roots_op_, roots_quo_, shared_buffer));
#else
#if (5 < LOG_TER_GAP_SIZE)
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelin<T, 5>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
#else
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRelin<T, LOG_TER_GAP_SIZE>(TER_LOCAL_MEM_GAP_SIZE, key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
#endif
#endif
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRelin", e1);
                                }
                                for (; gap <= TER_LOCAL_MEM_GAP_SIZE; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++) ;
                                for (; m > 1; total_r += m, m >>= 1, log_m--, gap <<= 1, log_gap++)
                                {
                                    static const int log_unroll = 0;
                                    auto grid_range = sycl::range<2>(poly_num_, (1 << (log_m + log_gap - log_unroll)));
                                    auto e2 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRelin<T, log_unroll>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRelin", e2);
                                }

                                auto grid_range = sycl::range<2>(poly_num_, (1 << log_gap));
                                auto e2 = queue.submit([&](cl::sycl::handler& h)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                {
                                    h.parallel_for({ grid_range }, RnsInvDwtLastRoundScalarRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_ , lazy_));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtLastRoundScalarRelin", e2);
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
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8FusedRelin", e);
                            }else{
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (n >> 3) ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix8Relin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix8Relin", e);
                                total_r += m + (m>>1) + (m>>2); m >>= 3; log_m -= 3;gap <<= 1; log_gap += 1;
                                for (; log_gap <= 11;total_r +=m, m >>= 1, log_m -= 1, gap <<= 1, log_gap += 1) ;
                                if (n == 8192){
                                    // std::cout<<"n ="<<n<<", gap = "<<gap<<", m  = "<<m<<", total_r = "<<total_r<<std::endl;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>1) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix2FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRelin", e1);
                                }else if (n == 16384){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4FusedRelin", e1); 
                                }else if (n == 32768){
                                    gap <<= 2; log_gap += 2;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>3) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix8FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix8FusedRelin", e1);
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
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix4FusedRelin<T, 2, 1>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, scalar_op_, scalar_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix4FusedRelin", e);
                                
                            }else{
                                auto nd_grid_range_local = sycl::nd_range<2>({static_cast<uint32_t>(poly_num_),\
                                                                        static_cast<uint32_t>( (n >> 3) ) },\
                                                                        {static_cast<uint32_t>(1),\
                                                                        static_cast<uint32_t>( 512 )}\
                                                                    );
                                auto e = queue.submit([&](cl::sycl::handler& h)
                                {
                                    auto shared_buffer = local_accessor<T, 1>(4096, h);
                                    h.parallel_for(nd_grid_range_local, RnsInvDwtGapLocalRadix4Relin<T, 2, 1>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                        roots_op_, roots_quo_, shared_buffer));
                                });//.wait();
                                EventCollector::add_event("RnsInvDwtGapLocalRadix4Relin", e);
                                total_r += m + (m>>1); m >>= 2; log_m -= 2;gap <<= 1; log_gap += 1;
                                for (; log_gap <= 11;total_r +=m, m >>= 1, log_m -= 1, gap <<= 1, log_gap += 1) ;
                                if (n == 8192){
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>1) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix2FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRelin", e1);
                                }else if (n == 16384){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4FusedRelin", e1); 
                                }else if (n == 32768){
                                    gap <<= 1; log_gap += 1;
                                    auto grid_range = sycl::range<2>(poly_num_ , (n>>2) );
                                    auto e1 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range }, RnsInvDwtGapRadix4Relin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                roots_op_, roots_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix4Relin", e1);
                                    total_r += m + (m>>1); m >>= 2; log_m -= 2;gap <<= 1; log_gap += 1;

                                    auto grid_range1 = sycl::range<2>(poly_num_ , (n>>1) );
                                    auto e2 = queue.submit([&](cl::sycl::handler& h)
                                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                                    {
                                        h.parallel_for({ grid_range1 }, RnsInvDwtGapRadix2FusedRelin<T>(key_modulus_size_, q_base_size_, log_n_, log_m, log_gap, total_r, values_, modulus_,
                                                                                                    roots_op_, roots_quo_, scalar_op_, scalar_quo_));
                                    });//.wait();
                                    EventCollector::add_event("RnsInvDwtGapRadix2FusedRelin", e2);
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
            int key_modulus_size_;
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
#endif
