/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XEHE_POLY_ARITH_SMALL_MOD_H
#define XEHE_POLY_ARITH_SMALL_MOD_H

#include "util/defines.h"
#include "util/common.h"
#include "util/xe_uintcore.h"
#include "util/xe_uintarithmod.h"
#include "util/uintarithsmallmod.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

#include "util/poly_kernels.h"

namespace xehe {
    namespace util {

        template<typename T>
        void 
            poly_coeff_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* values,
                const T* modulus, const T* const_ratio, int const_ratio_sz,
                T* result) {
            auto n = (size_t(1) << log_n);
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto idx = 0; idx < n; ++idx) {
                    for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                        xehe::kernels::kernel_poly_coeff_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            values, modulus, const_ratio,
                            const_ratio_sz, result);
                    }
                }
            }
        }

        template<typename T>
        void
            poly_neg_coeff_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* values,
                const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_neg_coeff_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            values, modulus, result);
                    }
                }
            }
        }

        template<typename T>
        void
            poly_add_coeff_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);

            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
//                        uint64_t global_idx = idx + rns_idx * n + p * q_base_size * n;
                        xehe::kernels::kernel_poly_add_coeff_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus, result);
                    }
                }
            }
        }

        template<typename T>
        void
            poly_sub_coeff_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);

            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_sub_coeff_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus, result);
                    }
                }
            }
        }

        template<typename T>
        void 
            poly_add_coeff_scalar_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);

            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_add_coeff_scalar_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus,
                            result);
                    }
                }
            }
        }

        template<typename T>
        void 
            poly_sub_coeff_scalar_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);

            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
//                        uint64_t global_idx = idx + rns_idx * n;
                        xehe::kernels::kernel_poly_sub_coeff_scalar_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus,
                            result);
                    }
                }
            }
        }


        template<typename T>
        void 
            poly_mul_coeff_scalar_mod_cpu(int n_polys, int q_base_size, int enc_sz, int log_n,
                const T* oprnd1, const MultiplyUIntModOperand <T>* scalar,
                const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);
            for (int p = 0; p < enc_sz; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_mul_coeff_scalar_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, scalar, modulus, result);
                    }
                }
            }
        }


        template<typename T>
        void
            poly_mul_coeff_scalar_mod_cpu(int n_polys, int q_base_size, int log_n, const T* oprnd1,
                const T* oprnd2,
                const T* modulus, const T* const_ratio, int const_ratio_sz,
                T* result) {

            // TODO: memory allocation should not really happen here!
            auto scalar_mod = new MultiplyUIntModOperand<T>[q_base_size];

            for (int i = 0; i < q_base_size; ++i) {
                set_operand<T>(barrett_reduce_64(oprnd2[0], modulus[i], const_ratio + i * const_ratio_sz),
                    modulus[i],
                    scalar_mod[i]);
            }

            auto n = (size_t(1) << log_n);
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_mul_coeff_scalar_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1,
                            (const MultiplyUIntModOperand<T> *) scalar_mod,
                            modulus,
                            result);
                    }
                }
            }

            delete[] scalar_mod;
        }


        template<typename T>
        void 
            poly_dyadic_product_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* oprnd2,
                const T* modulus,
                const T* const_ratio, int const_ratio_sz,
                T* result) {
            auto n = (size_t(1) << log_n);
            for (int poly = 0; poly < n_polys; ++poly)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_dyadic_product_mod(idx, rns_idx, poly,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus,
                            const_ratio, const_ratio_sz,
                            result);
                    }
                }
            }
        }

        template<typename T>
        void 
            poly_negacyclic_shift_mod_cpu(int n_polys, int q_base_size, int log_n,
                const T* oprnd1,
                const size_t shift, const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_negacyclic_shift_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, shift, modulus,
                            result);
                    }
                }
            }
        }

        template<typename T>
        void 
            poly_negacyclic_mono_mul_mod_cpu(
                int n_polys, int q_base_size, int log_n,
                const T* oprnd1,
                const MultiplyUIntModOperand <T>* mono_coeff, std::size_t mono_exponent,
                const T* modulus, T* result) {
            auto n = (size_t(1) << log_n);
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_negacyclic_mono_mul_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1,
                            mono_coeff, mono_exponent,
                            modulus, result);
                    }
                }
            }
        }


        template<typename T>
        void
            poly_negacyclic_mono_mul_mod_cpu(
                int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* mono_coeff,
                std::size_t mono_exponent, const T* modulus, const T* const_ratio,
                int const_ratio_sz, T* result) {
            //NOTE: mono_coeff is a pointer, only [0] value is used
            auto n = (size_t(1) << log_n);

            auto mono_coeff_mod = new MultiplyUIntModOperand<T>[q_base_size];
            for (int i = 0; i < q_base_size; ++i) {
                set_operand<T>(barrett_reduce_64(mono_coeff[0], modulus[i], const_ratio + i * const_ratio_sz),
                    modulus[i], mono_coeff_mod[i]);
            }

            for (int p = 0; p < n_polys; ++p)
            {
                for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                    for (auto idx = 0; idx < n; ++idx) {
                        xehe::kernels::kernel_poly_negacyclic_mono_mul_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1,
                            (const MultiplyUIntModOperand<T> *) mono_coeff_mod,
                            mono_exponent,
                            modulus, result);
                    }
                }
            }

            delete[] mono_coeff_mod;
        }

    } // namespace util
} // namespace seal

#endif //XEHE_POLY_ARITH_SMALL_MOD_H