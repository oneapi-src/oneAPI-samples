/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XEHE_POLY_KERNELS_H
#define XEHE_POLY_KERNELS_H

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "util/defines.h"
#include "util/common.h"
#include "util/xe_uintcore.h"
#include "util/xe_uintarithmod.h"
#include "util/uintarithsmallmod.h"



namespace xehe {
    namespace kernels {

        /********************************************************************************/
        // OLD
        /**********************************************************************************/
        template<typename T>
        void 
            kernel_poly_coeff_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* values, const T* modulus,
                const T* const_ratio, int const_ratio_sz, T* result) {

            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::barrett_reduce_64<T>(values[global_idx], modulus[rns_idx],
                const_ratio + rns_idx * const_ratio_sz);
        }

        template<typename T>
        void 
            kernel_poly_neg_coeff_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* values, const T* modulus, T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::negate_uint_mod<T>(values[global_idx], modulus[rns_idx]);
        }

        template<typename T>
        void
            kernel_poly_add_coeff_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2, const T* modulus,
                T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::add_uint_mod(oprnd1[global_idx], oprnd2[global_idx], modulus[rns_idx]);
        }


        template<typename T>
        void
            kernel_poly_sub_coeff_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2, const T* modulus,
                T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::sub_uint_mod(oprnd1[global_idx], oprnd2[global_idx], modulus[rns_idx]);
        }

        template<typename T>
        void
            kernel_poly_add_coeff_scalar_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2,
                const T* modulus, T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::add_uint_mod(oprnd1[global_idx], oprnd2[0], modulus[rns_idx]);
        }

        template<typename T>
        void
            kernel_poly_sub_coeff_scalar_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::sub_uint_mod(oprnd1[global_idx], oprnd2[0], modulus[rns_idx]);
        }


        template<typename T>
        void
            kernel_poly_mul_coeff_scalar_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1,
                const util::MultiplyUIntModOperand <T>* scalar,
                const T* modulus, T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::multiply_uint_mod(oprnd1[global_idx], scalar[rns_idx], modulus[rns_idx]);
        }



        template<typename T>
        void
            kernel_poly_dyadic_product_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                const T* const_ratio, int const_ratio_sz, T* result) {
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;

            result[global_idx] = util::multiply_uint_mod(oprnd1[global_idx], oprnd2[global_idx], modulus[rns_idx],
                const_ratio + uint64_t(rns_idx) * const_ratio_sz);
        }

        template<typename T>
        T shift_op(uint64_t& index, uint64_t n, uint64_t ind, uint64_t shift, T poly, T modulus) {
            uint64_t coeff_count_mod_mask = n - 1;
            auto i = ind % n;
            uint64_t index_raw = shift + i;
            index = (index_raw & coeff_count_mod_mask);
            T new_poly = (!(index_raw & n) || !poly) ? poly : (modulus - poly);
            return new_poly;
        }



        template<typename T>
        void
            kernel_poly_negacyclic_shift_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const size_t shift,
                const T* modulus,
                T* result) {
            uint64_t index;
            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = uint64_t(coeff_idx) + poly_offset;
            auto poly = oprnd1[global_idx];

            T new_poly = shift_op<T>(index, n, coeff_idx, shift, poly, modulus[rns_idx]);

            result[index + poly_offset] = new_poly;
        }


        template<typename T>
        void
            kernel_poly_negacyclic_mono_mul_mod(int coeff_idx, int rns_idx, int poly_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1,
                const util::MultiplyUIntModOperand <T>* mono_coeff,
                std::size_t mono_exponent,
                const T* modulus, T* result) {
            uint64_t index;

            uint64_t poly_offset = uint64_t(rns_idx) * n + uint64_t(poly_idx) * n * q_base_size;
            uint64_t global_idx = poly_offset + uint64_t(coeff_idx);
            auto poly = oprnd1[global_idx];

            auto mul_poly = multiply_uint_mod(poly, mono_coeff[rns_idx], modulus[rns_idx]);
            T new_poly = shift_op<T>(index, n, coeff_idx, mono_exponent, mul_poly, modulus[rns_idx]);

            result[index + poly_offset] = new_poly;
        }

/* 1D kernels


        template<typename T>
        void
        kernel_poly_neg_coeff_mod(int idx, int q_base_size, int n, const T *values, const T *modulus, T *result) {
            auto p = idx / n;
            result[idx] = util::negate_uint_mod<T>(values[idx], modulus[p]);
        }


        template<typename T>
        void
        kernel_poly_sub_coeff_mod(int idx, int q_base_size, int n, const T *oprnd1, const T *oprnd2, const T *modulus,
                                  T *result) {
            auto p = idx / n;
            result[idx] = util::sub_uint_mod(oprnd1[idx], oprnd2[idx], modulus[p]);
        }


        template<typename T>
        void
        kernel_poly_add_coeff_scalar_mod(int idx, int q_base_size, int n, const T *oprnd1, const T *oprnd2,
                                         const T *modulus, T *result) {
            auto p = idx / n;
            result[idx] = util::add_uint_mod(oprnd1[idx], oprnd2[0], modulus[p]);
        }


        template<typename T>
        void
        kernel_poly_sub_coeff_scalar_mod(int idx, int q_base_size, int n, const T *oprnd1, const T *oprnd2,
                                         const T *modulus, T *result) {
            auto p = idx / n;
            result[idx] = util::sub_uint_mod(oprnd1[idx], oprnd2[0], modulus[p]);
        }


        template<typename T>
        void
        kernel_poly_mul_coeff_scalar_mod(int idx, int q_base_size, int n, const T *oprnd1,
                                         const util::MultiplyUIntModOperand <T> *scalar,
                                         const T *modulus, T *result) {
            auto p = idx / n;
            result[idx] = util::multiply_uint_mod(oprnd1[idx], scalar[p], modulus[p]);
        }

        template<typename T>
        void
        kernel_poly_dyadic_product_mod(int idx, int q_base_size, int n, const T *oprnd1, const T *oprnd2,
                                       const T *modulus,
                                       const T *const_ratio, int const_ratio_sz, T *result) {
            auto p = idx / n;
            result[idx] = util::multiply_uint_mod(oprnd1[idx], oprnd2[idx], modulus[p],
                                                  const_ratio + p * const_ratio_sz);
        }


        template<typename T>
        void
        kernel_poly_negacyclic_shift_mod(int ind, int q_base_size, int n, const T *oprnd1, const size_t shift,
                                         const T *modulus,
                                         T *result) {
            auto p = ind / n;
            uint64_t index;
            auto poly = oprnd1[ind];

            T new_poly = shift_op<T>(index, n, ind, shift, poly, modulus[p]);

            result[index + p * n] = new_poly;
        }


*/
    }
}


#endif //XEHE_POLY_KERNELS_H
