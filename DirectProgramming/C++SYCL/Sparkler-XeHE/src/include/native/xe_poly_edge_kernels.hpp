/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XEHE_POLY_EDGE_KERNELS_HPP
#define XEHE_POLY_EDGE_KERNELS_HPP

#include <algorithm>
#include <cstdint>
#include <stdexcept>


//XeHE
#include "xe_uintarith_core.hpp"


namespace xehe {
    namespace kernels {

        // sub when n_poly1 > n_poly2
        // R1 = A1 - B1
        // R2 = A2 - B2
        // R3 = A3
        template<typename T>
        void
            kernel_coeff_sub_mod_1(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2, const T* modulus,
                T* result, const T* oprnd_longer, int n_poly_max) {

            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx_base = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx_curr = global_idx_base;
            T curr_modulus = modulus[rns_idx];
            int count_n_poly = 0;
            for (; count_n_poly < n_polys; count_n_poly++){
                result[global_idx_curr] = xehe::native::sub_mod<T>(oprnd1[global_idx_curr], oprnd2[global_idx_curr], curr_modulus);
                global_idx_curr += leading_dim;
            }
            for (; count_n_poly < n_poly_max; count_n_poly++){
                result[global_idx_curr] = oprnd_longer[global_idx_curr];
                global_idx_curr += leading_dim;
            }
        }

        // sub when n_poly1 < n_poly2
        // R1 = A1 - B1
        // R2 = A2 - B2
        // R3 = -B3
        template<typename T>
        void
            kernel_coeff_sub_mod_2(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2, const T* modulus,
                T* result, const T* oprnd_longer, int n_poly_max) {

            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx_base = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx_curr = global_idx_base;
            T curr_modulus = modulus[rns_idx];
            int count_n_poly = 0;
            for (; count_n_poly < n_polys; count_n_poly++){
                result[global_idx_curr] = xehe::native::sub_mod<T>(oprnd1[global_idx_curr], oprnd2[global_idx_curr], curr_modulus);
                global_idx_curr += leading_dim;
            }
            for (; count_n_poly < n_poly_max; count_n_poly++){
                result[global_idx_curr] = ~oprnd_longer[global_idx_curr] + 1;
                global_idx_curr += leading_dim;
            }
        }

        // MAD_plain, n_poly_mul > n_poly_add
        // R1 = A1 * B + C1
        // R2 = A2 * B + C2
        // R3 = A3 * B
        template<typename T>
        void
            kernel_coeff_prod_mod_plain_add_1(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_poly_min,
                const T* oprnd_add, const T* oprnd_mul, const T* oprnd_plain,
                const T* modulus, const T* inv_mod2, T* result, int n_poly_max) {

            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx_base = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx_curr = global_idx_base;
            T B = oprnd_plain[global_idx_base];
            T curr_modulus = modulus[rns_idx];
            const T* curr_inv_mod2 = &inv_mod2[rns_idx * 2];
            int count_n_poly = 0;
            for (; count_n_poly < n_poly_min; count_n_poly++){
#if 0
                auto tmp = xehe::native::mul_mod<T>(oprnd_mul[global_idx_curr], B, curr_modulus, curr_inv_mod2);
                result[global_idx_curr] = xehe::native::add_mod(oprnd_add[global_idx_curr], tmp, curr_modulus);
#else
                result[global_idx_curr] = xehe::native::mad_uint_mod(oprnd_mul[global_idx_curr], B, oprnd_add[global_idx_curr], curr_modulus, curr_inv_mod2);
#endif
                global_idx_curr += leading_dim;
            }
            for (; count_n_poly < n_poly_max; count_n_poly++){
                result[global_idx_curr] = xehe::native::mul_mod<T>(oprnd_mul[global_idx_curr], B, curr_modulus, curr_inv_mod2);
                global_idx_curr += leading_dim;
            }
        }

        // MAD_plain, n_poly_mul < n_poly_add
        // R1 = A1 * B + C1
        // R2 = A2 * B + C2
        // R3 = C3
        template<typename T>
        void
            kernel_coeff_prod_mod_plain_add_2(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_poly_min,
                const T* oprnd_add, const T* oprnd_mul, const T* oprnd_plain,
                const T* modulus, const T* inv_mod2, T* result, int n_poly_max) {

            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx_base = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx_curr = global_idx_base;
            T B = oprnd_plain[global_idx_base];
            T curr_modulus = modulus[rns_idx];
            const T* curr_inv_mod2 = &inv_mod2[rns_idx * 2];
            int count_n_poly = 0;
            for (; count_n_poly < n_poly_min; count_n_poly++){
#if 0
                auto tmp = xehe::native::mul_mod<T>(oprnd_mul[global_idx_curr], B, curr_modulus, curr_inv_mod2);
                result[global_idx_curr] = xehe::native::add_mod(oprnd_add[global_idx_curr], tmp, curr_modulus);
#else
                result[global_idx_curr] = xehe::native::mad_uint_mod(oprnd_mul[global_idx_curr], B, oprnd_add[global_idx_curr], curr_modulus, curr_inv_mod2);
#endif
                global_idx_curr += leading_dim;
            }
            for (; count_n_poly < n_poly_max; count_n_poly++){
                result[global_idx_curr] = oprnd_add[global_idx_curr];
                global_idx_curr += leading_dim;
            }
        }

        // MAD, n_poly_add == 0
        // R1 = A1B1
        // R2 = A1B2 + A2B1
        // R3 = A2B2
        template<typename T>
        void
            kernel_coeff_prod_mod_add(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd1, const T* oprnd2,
                const T* modulus, const T* inv_mod2, T* result) {

            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx1 = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx2 = global_idx1 + leading_dim;
            uint64_t global_idx3 = global_idx2 + leading_dim;
            T A1 = oprnd1[global_idx1];
            T A2 = oprnd2[global_idx1];
            T B1 = oprnd1[global_idx2];
            T B2 = oprnd2[global_idx2];
            T curr_modulus = modulus[rns_idx];
            const T* curr_inv_mod2 = &inv_mod2[rns_idx * 2];

#if 0
            // old implementation: non-fused primitives
            // R1 = A1A2;
            result[global_idx1] = xehe::native::mul_mod(A1, A2, curr_modulus, curr_inv_mod2);
            // second poly result: A1B2 + A2B1
            T A1B2 = xehe::native::mul_mod(A1, B2, curr_modulus, curr_inv_mod2);
            T A2B1 = xehe::native::mul_mod(A2, B1, curr_modulus, curr_inv_mod2);
            // R2 = A1B2 + A2B1;
            result[global_idx2] = xehe::native::add_mod(A1B2, A2B1, curr_modulus);
            // third poly result: B1B2
            result[global_idx3] = xehe::native::mul_mod(B1, B2, curr_modulus, curr_inv_mod2);

#else
            // fused primitive: it does lead to overflow risks.
            // first poly result: A1A2
            result[global_idx1] = xehe::native::mul_mod(A1, A2, curr_modulus, curr_inv_mod2);
            T A1B2[2];
            A1B2[0] = xehe::native::mul_uint(A1, B2, A1B2 + 1);
            // second poly result: A1B2 + A2B1
            result[global_idx2] = xehe::native::mad_uint_mod(A2, B1, A1B2, curr_modulus, curr_inv_mod2);
            // third poly result: B1B2
            result[global_idx3] = xehe::native::mul_mod(B1, B2, curr_modulus, curr_inv_mod2);
#endif
        }

        // MAD, n_poly_add == 1
        // R1 = A1B1 + C1
        // R2 = A1B2 + A2B1
        // R3 = A2B2
        template<typename T>
        void
            kernel_coeff_prod_mod_add_1(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd_add, const T* oprnd2, const T* oprnd3,
                const T* modulus, const T* inv_mod2, T* result) {

            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx1 = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx2 = global_idx1 + leading_dim;
            uint64_t global_idx3 = global_idx2 + leading_dim;
            T A1 = oprnd2[global_idx1];
            T A2 = oprnd3[global_idx1];
            T B1 = oprnd2[global_idx2];
            T B2 = oprnd3[global_idx2];
            T curr_modulus = modulus[rns_idx];
            const T* curr_inv_mod2 = &inv_mod2[rns_idx * 2];

#if 0
            // old implementation: non-fused primitives
            // first poly result: A1A2
            T A1A2 = xehe::native::mul_mod(A1, A2, curr_modulus, curr_inv_mod2);
            // R1 = A1A2 + C1;
            result[global_idx1] = xehe::native::add_mod(A1A2, oprnd_add[global_idx1], curr_modulus);
            // second poly result: A1B2 + A2B1
            T A1B2 = xehe::native::mul_mod(A1, B2, curr_modulus, curr_inv_mod2);
            T A2B1 = xehe::native::mul_mod(A2, B1, curr_modulus, curr_inv_mod2);
            // R2 = A1B2 + A2B1;
            result[global_idx2] = xehe::native::add_mod(A1B2, A2B1, curr_modulus);
            // third poly result: B1B2
            result[global_idx3] = xehe::native::mul_mod(B1, B2, curr_modulus, curr_inv_mod2);

#else
            // fused primitive: it does lead to overflow risks.
            // first poly result: A1A2
            result[global_idx1] = xehe::native::mad_uint_mod(A1, A2, oprnd_add[global_idx1], curr_modulus, curr_inv_mod2);
            T A1B2[2];
            A1B2[0] = xehe::native::mul_uint(A1, B2, A1B2 + 1);
            // second poly result: A1B2 + A2B1
            result[global_idx2] = xehe::native::mad_uint_mod(A2, B1, A1B2, curr_modulus, curr_inv_mod2);
            // third poly result: B1B2
            result[global_idx3] = xehe::native::mul_mod(B1, B2, curr_modulus, curr_inv_mod2);
#endif
        }

        // MAD, n_poly_add = 2
        // R1 = A1B1 + C1
        // R2 = A1B2 + A2B1 + C2
        // R3 = A2B2
        template<typename T>
        void
            kernel_coeff_prod_mod_add_2(int coeff_idx, int rns_idx,
                int n, int q_base_size, int n_polys,
                const T* oprnd_add, const T* oprnd2, const T* oprnd3,
                const T* modulus, const T* inv_mod2, T* result) {


#if _INF_MEMORY_
            uint64_t global_idx = 0;
            rns_idx = 0;
            auto tmp = xehe::native::mul_mod<T>(oprnd2[global_idx], oprnd3[global_idx], modulus[rns_idx], &inv_mod2[rns_idx * 2]);
            result[global_idx] = xehe::native::add_mod(oprnd_add[global_idx], tmp, modulus[rns_idx]);
#else
            uint64_t leading_dim = uint64_t(n) * q_base_size;
            uint64_t poly_offset = uint64_t(rns_idx) * n;
            uint64_t global_idx1 = uint64_t(coeff_idx) + poly_offset;
            uint64_t global_idx2 = global_idx1 + leading_dim;
            uint64_t global_idx3 = global_idx2 + leading_dim;
            T A1 = oprnd2[global_idx1];
            T A2 = oprnd3[global_idx1];
            T B1 = oprnd2[global_idx2];
            T B2 = oprnd3[global_idx2];
            T curr_modulus = modulus[rns_idx];
            const T* curr_inv_mod2 = &inv_mod2[rns_idx * 2];

#if 0
            // old implementation: non-fused primitives
            // first poly result: A1A2
            T A1A2 = xehe::native::mul_mod(A1, A2, curr_modulus, curr_inv_mod2);
            // R1 = A1A2 + C1;
            result[global_idx1] = xehe::native::add_mod(A1A2, oprnd_add[global_idx1], curr_modulus);
            // second poly result: A1B2 + A2B1
            T A1B2 = xehe::native::mul_mod(A1, B2, curr_modulus, curr_inv_mod2);
            T A2B1 = xehe::native::mul_mod(A2, B1, curr_modulus, curr_inv_mod2);
            T tmp = xehe::native::add_mod(A1B2, A2B1, curr_modulus);
            // R2 = A1B2 + A2B1 + C2;
            result[global_idx2] = xehe::native::add_mod(tmp, oprnd_add[global_idx2], curr_modulus);
            // third poly result: B1B2
            result[global_idx3] = xehe::native::mul_mod(B1, B2, curr_modulus, curr_inv_mod2);

#else
            // fused primitive: it does lead to overflow risks.
            // first poly result: A1A2
            result[global_idx1] = xehe::native::mad_uint_mod(A1, A2, oprnd_add[global_idx1], curr_modulus, curr_inv_mod2);
            T A1B2[2];
            // second poly result: A1B2 + A2B1
            A1B2[0] = xehe::native::mul_uint(A1, B2, A1B2 + 1);
            // R2 = A1B2 + A2B1 + C2;
            result[global_idx2] = xehe::native::add_mod(xehe::native::mad_uint_mod(A2, B1, A1B2, curr_modulus, curr_inv_mod2), oprnd_add[global_idx2], curr_modulus);
            // third poly result: B1B2
            result[global_idx3] = xehe::native::mul_mod(B1, B2, curr_modulus, curr_inv_mod2);
#endif

#endif
        }

    }
}


#endif //XEHE_POLY_EDGE_KERNELS_H
