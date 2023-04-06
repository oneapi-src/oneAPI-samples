/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XEHE_POLY_ARITH_SMALL_MOD_GPU_H
#define XEHE_POLY_ARITH_SMALL_MOD_GPU_H

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

#ifdef BUILD_WITH_IGPU

#include "dpcpp_utils.h"
#include <CL/sycl.hpp>


        /*
           Utility kernels
        */

        template<typename T>
        class poly_coeff_mod;

        template<typename T>
        class poly_neg_mod;

        template<typename T>
        class poly_add_mod;

        template<typename T>
        class poly_sub_mod;

        template<typename T>
        class poly_dyadic_product_mod;

        template<typename T>
        class poly_negacyclic_shift_mod;

        template<typename T>
        class poly_negacyclic_mono_mul_mod_operand;

        template<typename T>
        class poly_mul_scalar_mod;

        template<typename T>
        class poly_sub_coeff_scalar_mod;

        template<typename T>
        class poly_add_coeff_scalar_mod;

        template<typename T>
        class OperandModPrecomputeGPU {
        public:
            OperandModPrecomputeGPU(int q_base_size, const T value, const T *modulus, const T *const_ratio,
                                    int const_ratio_sz, MultiplyUIntModOperand<T> *result) {
                q_base_size_ = q_base_size;
                value_ = value;
                modulus_ = modulus;
                const_ratio_ = const_ratio;
                const_ratio_sz_ = const_ratio_sz;
                result_ = result;

            }

            void operator()(cl::sycl::id<1> i) const {
                auto value_modulo = barrett_reduce_64(value_, modulus_[i], const_ratio_ + i * const_ratio_sz_);
                set_operand<T>(value_modulo, modulus_[i], result_[i]);
            }

        protected:
            int q_base_size_;
            T value_;
            MultiplyUIntModOperand<T> *result_;
            const T *modulus_;
            const T *const_ratio_;
            int const_ratio_sz_;
        };
#if 0
        template<typename T>
        void SYCL_EXTERNAL calc_global_index(cl::sycl::item<3> it, int q_base_size, int n, T& global_idx)
        {
            int idx = it.get_id(0);
            int rns_idx = it.get_id(1);
            int poly = it.get_id(2);
            global_idx = T(idx) + T(rns_idx * n) + T(poly * q_base_size * n);
        }
#endif
        template<typename T>
        void poly_coeff_mod_gpu(cl::sycl::queue& queue,
            int n_polys, int q_base_size, int log_n,
            const T* values, const T* modulus,
            const T* const_ratio, int const_ratio_sz, T* result) {
            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);
            queue.submit([&](sycl::handler& h) {
                h.parallel_for<poly_coeff_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(2);
                    xehe::kernels::kernel_poly_coeff_mod(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        values, modulus, const_ratio,
                        const_ratio_sz, result);
                    });
                }).wait();
        }

        template<typename T>
        void
            poly_neg_coeff_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n, const T* values, const T* modulus,
                T* result) {
            auto n = (size_t(1) << log_n);

            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);
            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_neg_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(2);
                    xehe::kernels::kernel_poly_neg_coeff_mod(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        values, modulus, result);
                    });
                }).wait();

        }

        template<typename T>
        void
            poly_add_coeff_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n, const T* oprnd1, const T* oprnd2,
                const T* modulus, T* result) {
            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);
            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_add_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(2);
                    xehe::kernels::kernel_poly_add_coeff_mod(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus, result);
                    });
                }).wait();
        }

        template<typename T>
        void
            poly_sub_coeff_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n, const T* oprnd1, const T* oprnd2,
                const T* modulus, T* result) {
            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_sub_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(2);
                    xehe::kernels::kernel_poly_sub_coeff_mod(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus, result);
                    });
                }).wait();
        }

        template<typename T>
        void poly_add_coeff_scalar_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n, const T* oprnd1,
            const T* oprnd2,
            const T* modulus, T* result) {
            // scalar = oprnd2[0]
            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_add_coeff_scalar_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(2);
                    xehe::kernels::kernel_poly_add_coeff_scalar_mod(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus,
                        result);
                    });
                }).wait();
        }

        template<typename T>
        void poly_sub_coeff_scalar_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n, const T* oprnd1,
            const T* oprnd2,
            const T* modulus, T* result) {
            // scalar = oprnd2[0]
            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_sub_coeff_scalar_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(2);
                    xehe::kernels::kernel_poly_sub_coeff_scalar_mod(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus,
                        result);
                    });
                }).wait();
        }


        template<typename T>
        void
            poly_mul_coeff_scalar_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n,
                const T* oprnd1, const T* scalar,
                const T* modulus, const T* const_ratio, int const_ratio_sz,
                T* result) {
            //NOTE: scalar is a pointer

            auto n = (size_t(1) << log_n);
            auto oprnd2 = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(q_base_size, queue);

#if 0
            for (int i = 0; i < q_base_size; ++i)
            {
                set_operand<T>(barrett_reduce_64(scalar, modulus[i], const_ratio + i * const_ratio_sz), modulus[i], oprnd2[i]);
            }
#else
            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for((q_base_size),
                    OperandModPrecomputeGPU(q_base_size, scalar[0], modulus, const_ratio, const_ratio_sz,
                        oprnd2));
                }).wait();
#endif
                auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

                queue.submit([&](cl::sycl::handler& h) {
                    h.parallel_for<class poly_mul_scalar_mod<T>>({ grid_range }, [=](auto it) {
                        int idx = it.get_id(0);
                        int rns_idx = it.get_id(1);
                        int poly_idx = it.get_id(2);
                        xehe::kernels::kernel_poly_mul_coeff_scalar_mod(idx, rns_idx, poly_idx,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus,
                            result);
                        });
                    }).wait();

                    cl::sycl::free(oprnd2, queue);
        }

        template<typename T>
        void poly_dyadic_product_mod_gpu(cl::sycl::queue& queue, int n_polys, int q_base_size, int log_n,
            const T* oprnd1, const T* oprnd2,
            const T* modulus, const T* const_ratio, int const_ratio_sz,
            T* result) {

            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_dyadic_product_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(2);
                    xehe::kernels::kernel_poly_dyadic_product_mod(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus,
                        const_ratio, const_ratio_sz,
                        result);
                    });
                }).wait();
        }


        template<typename T>
        void poly_negacyclic_shift_mod_gpu(cl::sycl::queue& queue,
            int n_polys, int q_base_size, int log_n,
            const T* oprnd1,
            const size_t shift, const T* modulus,
            T* result) {
            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_negacyclic_shift_mod<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(2);
                    xehe::kernels::kernel_poly_negacyclic_shift_mod(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, shift, modulus,
                        result);
                    });
                }).wait();
        }

        template<typename T>
        void poly_negacyclic_mono_mul_mod_gpu(cl::sycl::queue& queue,
            int n_polys, int q_base_size, int log_n,
            const T* oprnd1,
            const MultiplyUIntModOperand<T>* mono_coeff,
            std::size_t mono_exponent,
            const T* modulus, T* result) {

            auto n = (size_t(1) << log_n);
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);

            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class poly_negacyclic_mono_mul_mod_operand<T>>({ grid_range }, [=](auto it) {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(2);
                    xehe::kernels::kernel_poly_negacyclic_mono_mul_mod(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1,
                        mono_coeff, mono_exponent,
                        modulus, result);
                    });
                }).wait();
        }


        template<typename T>
        void poly_negacyclic_mono_mul_mod_gpu(cl::sycl::queue& queue,
            int n_polys, int q_base_size, int log_n,
            const T* oprnd1,
            const T* mono_coeff, std::size_t mono_exponent,
            const T* modulus, const T* const_ratio,
            int const_ratio_sz, T* result) {


            auto mono_coeff_mod = cl::sycl::malloc_shared<MultiplyUIntModOperand<T>>(q_base_size, queue);

#if 0
            for (int i = 0; i < q_base_size; ++i) {
                set_operand<T>(barrett_reduce_64(mono_coeff[0], modulus[i], const_ratio + i * const_ratio_sz),
                    modulus[i], mono_coeff_mod[i]);
            }
#else
            queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for((q_base_size),
                    OperandModPrecomputeGPU(q_base_size, mono_coeff[0], modulus, const_ratio, const_ratio_sz,
                        mono_coeff_mod));
                }).wait();
#endif
                poly_negacyclic_mono_mul_mod_gpu(queue,
                    n_polys, q_base_size, log_n,
                    oprnd1,
                    mono_coeff_mod,
                    mono_exponent,
                    modulus, result);

                cl::sycl::free(mono_coeff_mod, queue);
        }

#endif

#if 0
        std::uint64_t poly_infty_norm_coeffmod(ConstCoeffIter operand, std::size_t coeff_count, const Modulus &modulus);

#endif

    } // namespace util
} // namespace seal

#endif //XEHE_POLY_ARITH_SMALL_MOD_GPU_H
