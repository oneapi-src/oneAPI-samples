/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XEHE_POLY_OPS_HPP
#define XEHE_POLY_OPS_HPP


#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

// XeHE
#include "xe_poly_kernels.hpp"
#include "xe_poly_edge_kernels.hpp"

namespace xehe {
    namespace native {

        template<typename T>
        void
            poly_coeff_mod(int n_polys, int q_base_size, int n,
                const T* values, const T* modulus,
                const T* mod_inv, T* result) {
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto idx = 0; idx < n; ++idx) {
                    for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                        xehe::kernels::kernel_coeff_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            values, modulus, mod_inv, result);
                    }
                }
            }
        }

        template<typename T>
        void poly_coeff_neg_mod(int n_polys, int q_base_size, int n, const T* values, const T* modulus,
            T* result) {
                for (int p = 0; p < n_polys; ++p)
                {
                    for (auto idx = 0; idx < n; ++idx) {
                        for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                            xehe::kernels::kernel_coeff_neg_mod(idx, rns_idx, p,
                                n, q_base_size, n_polys,
                                values, modulus, result);
                        }
                    }
                }

        }


        template<typename T>
        void poly_coeff_add_mod(int n_polys, int q_base_size, int n, const T* oprnd1, const T* oprnd2,
                const T* modulus, T* result) {
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto idx = 0; idx < n; ++idx) {
                    for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                        xehe::kernels::kernel_coeff_add_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus, result);
                    }
                }
            }
        }

        template<typename T>
        void poly_coeff_sub_mod(int n_polys, int q_base_size, int n, const T* oprnd1, const T* oprnd2,
                const T* modulus, T* result) {

                for (int p = 0; p < n_polys; ++p)
                {
                    for (auto idx = 0; idx < n; ++idx) {
                        for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                            xehe::kernels::kernel_coeff_sub_mod(idx, rns_idx, p,
                                n, q_base_size, n_polys,
                                oprnd1, oprnd2, modulus, result);
                        }
                    }
                }

        }

        template<typename T>
        void poly_coeff_add_scalar_mod(int n_polys, int q_base_size, int n, const T* oprnd1,
            const T* scalar,
            const T* modulus, T* result) {

                for (int p = 0; p < n_polys; ++p)
                {
                    for (auto idx = 0; idx < n; ++idx) {
                        for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                            xehe::kernels::kernel_coeff_add_scalar_mod(idx, rns_idx, p,
                                n, q_base_size, n_polys,
                                oprnd1, scalar, modulus,
                                result);
                        }
                    }
                }
        }

        template<typename T>
        void poly_coeff_sub_scalar_mod(int n_polys, int q_base_size, int n, const T* oprnd1,
            const T* scalar,
            const T* modulus, T* result) {

            for (int p = 0; p < n_polys; ++p)
            {
                for (auto idx = 0; idx < n; ++idx) {
                    for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                        xehe::kernels::kernel_coeff_sub_scalar_mod(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, scalar, modulus,
                            result);
                    }
                }
            }
        }

        template<typename T>
        void
            poly_coeff_mul_scalar_mod(int n_polys, int q_base_size, int n,
                const T* oprnd1, const T* scalar,
                const T* modulus, const T* inv_mod,
                T* result) {
            //NOTE: scalar is a pointer

            std::vector<T> op2_mod(q_base_size);
            std::vector<T> op2_byinv_mod(q_base_size);

            for (int i = 0; i < q_base_size; ++i)
            {
                op2_mod[i] = xehe::native::barrett_reduce(*scalar, modulus[i], inv_mod[i]);
                op2_byinv_mod[i] = xehe::native::op_by_mod_inverse(op2_mod[i], modulus[i]);
            }
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto idx = 0; idx < n; ++idx) {
                    for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                        xehe::kernels::kernel_coeff_mul_scalar_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, op2_mod.data(), modulus, op2_byinv_mod.data(),
                            result);
                    }
                }
            }
        }


        template<typename T>
        void poly_coeff_prod_mod(int n_polys, int q_base_size, int n,
            const T* oprnd1, const T* oprnd2,
            const T* modulus, const T* inv_mod2,
            T* result) {
            for (int p = 0; p < n_polys; ++p)
            {
                for (auto idx = 0; idx < n; ++idx) {
                    for (auto rns_idx = 0; rns_idx < q_base_size; ++rns_idx) {
                        xehe::kernels::kernel_coeff_prod_mod<T>(idx, rns_idx, p,
                            n, q_base_size, n_polys,
                            oprnd1, oprnd2, modulus,
                            inv_mod2,
                            result);
                    }
                }
            }

        }

        template<typename T>
        void poly_coeff_prod_mod_add(int n_polys, int q_base_size, int n,
            const T* oprnd1, const T* oprnd2, const T* oprnd3,
            const T* modulus, const T* inv_mod2,
            T* result) {
            for (int rns_idx = 0; rns_idx < q_base_size; ++rns_idx)
            {
                for (int idx = 0; idx < n; ++idx)
                {
                    xehe::kernels::kernel_coeff_prod_mod_add<T>(idx, rns_idx,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, oprnd3, modulus,
                        inv_mod2,
                        result);
                }
            }
        }


#ifdef BUILD_WITH_IGPU

#include "dpcpp_utils.h"
#include <CL/sycl.hpp>


        /*
           Utility kernels
        */


        template<typename T>
        class krnl_coeff_mod;

        template<typename T>
        class krnl_coeff_mod_buf;

        template<typename T>
        class krnl_coeff_neg_mod;

        template<typename T>
        class krnl_coeff_neg_mod_buf;

        template<typename T>
        class krnl_coeff_add_mod1;
        template<typename T>
        class krnl_coeff_add_mod2;
        template<typename T>
        class krnl_coeff_add_mod3;
        template<typename T>
        class krnl_coeff_add_mod4;

        template<typename T>
        class krnl_coeff_add_mod_buf;

        template<typename T>
        class krnl_coeff_sub_mod1;
        template<typename T>
        class krnl_coeff_sub_mod2;
        template<typename T>
        class krnl_coeff_sub_mod3;
        template<typename T>
        class krnl_coeff_sub_mod4;
        template<typename T>
        class krnl_coeff_sub_mod5;

        template<typename T>
        class krnl_coeff_sub_mod_buf;

        template<typename T>
        class krnl_coeff_prod_mod;

        template<typename T>
        class krnl_coeff_prod_mod_plain;

        template<typename T>
        class krnl_coeff_prod_mod_plain_add1;
        template<typename T>
        class krnl_coeff_prod_mod_plain_add2;
        template<typename T>
        class krnl_coeff_prod_mod_plain_add3;

        template<typename T>
        class krnl_coeff_prod_mod_add1;
        template<typename T>
        class krnl_coeff_prod_mod_add2;
        template<typename T>
        class krnl_coeff_prod_mod_add3;
        template<typename T>
        class krnl_coeff_prod_mod_add4;

        template<typename T>
        class krnl_coeff_prod_mod_add_1;

        template<typename T>
        class krnl_coeff_prod_mod_add_2;

        template<typename T>
        class krnl_coeff_prod_mod_buf;

        template<typename T>
        class poly_negacyclic_shift_mod;

        template<typename T>
        class poly_negacyclic_shift_mod_buf;

        template<typename T>
        class poly_negacyclic_mono_mul_mod_operand;

        template<typename T>
        class krnl_coeff_mul_scalar_mod;

        template<typename T>
        class krnl_coeff_add_scalar_mod;

        template<typename T>
        class krnl_coeff_sub_scalar_mod;

        template<typename T>
        class krnl_coeff_mul_scalar_mod_buf;

        template<typename T>
        class krnl_coeff_add_scalar_mod_buf;

        template<typename T>
        class krnl_coeff_sub_scalar_mod_buf;

        template<typename T>
        class krnl_coeff_fused_prod_mod;

        template<typename T>
        class krnl_coeff_square;

        template<typename T>
        void poly_coeff_mod(cl::sycl::queue& queue,
            int n_polys, int q_base_size, int n,
            const T* values, const T* modulus,
            const T* mod_inv, T* result) {
            
            auto grid_range = sycl::range<3>(n, q_base_size, n_polys);
            queue.submit([&](sycl::handler& h) {
                h.parallel_for<krnl_coeff_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(0);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(2);
                    xehe::kernels::kernel_coeff_mod<T>(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        values, modulus, mod_inv,
                        result);
                });
                });
#if !_NO_WAIT_
                queue.wait();
#endif
        }

        template<typename T>
        void poly_coeff_mod(cl::sycl::queue& queue,
            int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T> &values, const cl::sycl::buffer<T> &modulus,
            const cl::sycl::buffer<T> &mod_inv, cl::sycl::buffer<T> &result) {
            //const sycl::property_list rd_only{ read_only };
            //const sycl::property_list wrt_only{ write_only, noinit };
            auto grid_range = sycl::range<3>(n_polys, q_base_size, n );
            queue.submit([&](sycl::handler& h) {
                auto aVal = ((cl::sycl::buffer<T> &)values).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod_inv = ((cl::sycl::buffer<T> &)mod_inv).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);

                h.parallel_for<krnl_coeff_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(0);
                    auto pVal = aVal.get_pointer();
                    auto pMod = aMod.get_pointer();
                    auto pMod_inv = aMod_inv.get_pointer();
                    auto pRes = aRes.get_pointer();
                    xehe::kernels::kernel_coeff_mod<T>(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        pVal.get(), pMod.get(), pMod_inv.get(),
                        pRes.get());
                });
            });
#if !_NO_WAIT_
            queue.wait();
#endif
        }


        template<typename T>
        void poly_coeff_neg_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n, const T* values, const T* modulus,
            T* result, bool wait = false) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_neg_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(0);
                    xehe::kernels::kernel_coeff_neg_mod<T>(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        values, modulus, result);
                });
                });
            EventCollector::add_event("kernel_coeff_neg_mod", e);
            if (wait) {
                queue.wait();
            }

        }

        template<typename T>
        void poly_coeff_neg_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T>& values, const cl::sycl::buffer<T>& modulus,
            cl::sycl::buffer<T>& result) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aVal = ((cl::sycl::buffer<T> &)values).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);

                h.parallel_for<class krnl_coeff_neg_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(0);
                    auto pVal = aVal.get_pointer();
                    auto pMod = aMod.get_pointer();
                    auto pRes = aRes.get_pointer();


                    xehe::kernels::kernel_coeff_neg_mod<T>(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        pVal.get(), pMod.get(), pRes.get());
                });    
            });
            EventCollector::add_event("kernel_coeff_neg_mod", e);
#if !_NO_WAIT_
            queue.wait();
#endif

        }


        template<typename T>
        void poly_coeff_add_mod(cl::sycl::queue& queue, int n_polys, int n_poly1, int n_poly2,
                                int q_base_size, int n, const T* oprnd1, const T* oprnd2,
                                const T* modulus, T* result, bool wait = false) {
            if (n_poly1 == 0 && n_poly2 == 0){
                std::cout<<"invalid input: the length of both input oprnds shouldn't be zero.\n";
            }else if ((n_poly1 == 0)||(n_poly2 == 0)){
                if (n_poly1 == 0){
                    auto grid_range = sycl::range<3>(n_poly2, q_base_size, n);
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_add_mod1<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(2);
                            int rns_idx = it.get_id(1);
                            int poly = it.get_id(0);
                            xehe::kernels::kernel_coeff_add_mod<T>(idx, rns_idx, poly,
                                                                n, q_base_size, n_polys,
                                                                oprnd2, result);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_add_mod", e);
                }else{
                    auto grid_range = sycl::range<3>(n_poly1, q_base_size, n);
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_add_mod2<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(2);
                            int rns_idx = it.get_id(1);
                            int poly = it.get_id(0);
                            xehe::kernels::kernel_coeff_add_mod<T>(idx, rns_idx, poly,
                                                                n, q_base_size, n_polys,
                                                                oprnd1, result);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_add_mod", e);
                }
            }else if (n_poly1 == n_poly2) {
#if _INF_COMP_
                auto grid_range = sycl::range<3>(1,1,1);
#else
                auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
#endif
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_add_mod3<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(2);
                        int rns_idx = it.get_id(1);
                        int poly = it.get_id(0);
                        xehe::kernels::kernel_coeff_add_mod<T>(idx, rns_idx, poly,
                                                            n, q_base_size, n_polys,
                                                            oprnd1, oprnd2, modulus, result);
                    });
                });
                EventCollector::add_event("kernel_coeff_add_mod", e);
            }
            else{
                // TODO: normal kernel for min(n_poly1, n_poly2) and space filling for diff
                bool flag = (n_poly1 < n_poly2);
                int n_poly_min = flag ? n_poly1 : n_poly2;
                int n_poly_max = flag ? n_poly2 : n_poly1;
                auto oprnd_longer = flag ? oprnd2 : oprnd1;
                auto grid_range = sycl::range<2>(q_base_size, n);
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_add_mod4<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(1);
                        int rns_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_add_mod<T>(idx, rns_idx,
                                                            n, q_base_size, n_poly_min,
                                                            oprnd1, oprnd2, modulus,
                                                            result, oprnd_longer, n_poly_max);
                    });
                });
                EventCollector::add_event("kernel_coeff_add_mod", e);
            }
            if (wait){
                queue.wait();
            }
        }


        template<typename T>
        void poly_coeff_add_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T>& oprnd1, const cl::sycl::buffer<T>& oprnd2,
            const cl::sycl::buffer<T>& modulus,
            cl::sycl::buffer<T>& result, bool wait = false) {
            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aOp1 = ((cl::sycl::buffer<T> &)oprnd1).template get_access<cl::sycl::access::mode::read>(h);
                auto aOp2 = ((cl::sycl::buffer<T> &)oprnd2).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);

                h.parallel_for<class krnl_coeff_add_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(0);
                    xehe::kernels::kernel_coeff_add_mod<T>(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        aOp1.get_pointer().get(), aOp2.get_pointer().get(),
                        aMod.get_pointer().get(),
                        aRes.get_pointer().get());
                });
            });
            EventCollector::add_event("kernel_coeff_add_mod", e);
            if (wait) {
                queue.wait();
            }
        }


        template<typename T>
        void
            poly_coeff_sub_mod(cl::sycl::queue& queue, int n_polys, int n_poly1, int n_poly2,
                               int q_base_size, int n, const T* oprnd1, const T* oprnd2,
                               const T* modulus, T* result, bool wait = false) {
            // TODO: probably check oprnd1 == 0 return neg(oprnd2), oprnd2 == 0 return oprnd1
            if (n_poly1 == 0 && n_poly2 == 0){
                std::cout<<"invalid input: the length of both input oprnds shouldn't be zero.\n" << std::endl;
                return;
            }else if ((n_poly1 == 0)||(n_poly2 == 0)){
                if (n_poly1 == 0){
                    auto grid_range = sycl::range<3>(n_poly2, q_base_size, n);
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_sub_mod1<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(2);
                            int rns_idx = it.get_id(1);
                            int poly = it.get_id(0);
                            xehe::kernels::kernel_coeff_sub_mod<T>(idx, rns_idx, poly,
                                                                n, q_base_size, n_polys,
                                                                oprnd2, result);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_sub_mod", e);
                }else{
                    auto grid_range = sycl::range<3>(n_poly1, q_base_size, n);
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_sub_mod2<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(2);
                            int rns_idx = it.get_id(1);
                            int poly = it.get_id(0);
                            xehe::kernels::kernel_coeff_sub_mod<T>(idx, rns_idx, poly,
                                                                n, q_base_size, n_polys,
                                                                oprnd1, result);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_sub_mod", e);
                }
            }else if (n_poly1 == n_poly2) {
                auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_sub_mod3<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(2);
                        int rns_idx = it.get_id(1);
                        int poly = it.get_id(0);
                        xehe::kernels::kernel_coeff_sub_mod<T>(idx, rns_idx, poly,
                                                               n, q_base_size, n_polys,
                                                               oprnd1, oprnd2, modulus, result);
                    });
                });
                EventCollector::add_event("kernel_coeff_sub_mod", e);
            }
            else{
                bool flag = (n_poly1 < n_poly2);
                int n_poly_min = flag ? n_poly1 : n_poly2;
                int n_poly_max = flag ? n_poly2 : n_poly1;
                auto oprnd_longer = flag ? oprnd2 : oprnd1;
                auto grid_range = sycl::range<2>(q_base_size, n);
                if (flag){
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_sub_mod4<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(1);
                            int rns_idx = it.get_id(0);
                            xehe::kernels::kernel_coeff_sub_mod_2<T>(idx, rns_idx,
                                                                n, q_base_size, n_poly_min,
                                                                oprnd1, oprnd2, modulus,
                                                                result, oprnd_longer, n_poly_max);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_sub_mod_2", e);
                }else{
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_sub_mod5<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(1);
                            int rns_idx = it.get_id(0);
                            xehe::kernels::kernel_coeff_sub_mod_1<T>(idx, rns_idx,
                                                                n, q_base_size, n_poly_min,
                                                                oprnd1, oprnd2, modulus,
                                                                result, oprnd_longer, n_poly_max);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_sub_mod_1", e);
                }
            }
            if (wait){
                queue.wait();
            }
        }

        template<typename T>
        void poly_coeff_sub_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T>& oprnd1, const cl::sycl::buffer<T>& oprnd2,
            const cl::sycl::buffer<T>& modulus,
            cl::sycl::buffer<T>& result) {
            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aOp1 = ((cl::sycl::buffer<T> &)oprnd1).template get_access<cl::sycl::access::mode::read>(h);
                auto aOp2 = ((cl::sycl::buffer<T> &)oprnd2).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);

                h.parallel_for<class krnl_coeff_sub_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly = it.get_id(0);
                    xehe::kernels::kernel_coeff_sub_mod<T>(idx, rns_idx, poly,
                        n, q_base_size, n_polys,
                        aOp1.get_pointer().get(), aOp2.get_pointer().get(),
                        aMod.get_pointer().get(),
                        aRes.get_pointer().get());
                });
            });
            EventCollector::add_event("kernel_coeff_sub_mod", e);
#if !_NO_WAIT_
                queue.wait();
#endif
        }

        template<typename T>
        void poly_coeff_add_scalar_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n, const T* oprnd1,
            const T* scalar,
            const T* modulus, T* result) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_add_scalar_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(0);
                    xehe::kernels::kernel_coeff_add_scalar_mod<T>(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, scalar, modulus,
                        result);
                    });
                });
                
            EventCollector::add_event("kernel_coeff_add_scalar_mod", e);
#if !_NO_WAIT_
                queue.wait();
#endif
        }

        template<typename T>
        void poly_coeff_add_scalar_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T>& oprnd1,
            T scalar,
            const cl::sycl::buffer<T>& modulus,
            cl::sycl::buffer<T>& result) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aOp1 = ((cl::sycl::buffer<T> &)oprnd1).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);

                h.parallel_for<class krnl_coeff_add_scalar_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(0);
                    T scalar_val = scalar;
                    xehe::kernels::kernel_coeff_add_scalar_mod<T>(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        aOp1.get_pointer().get(),
                        &scalar_val,
                        aMod.get_pointer().get(),
                        aRes.get_pointer().get());
                });
            });
            EventCollector::add_event("kernel_coeff_add_scalar_mod", e);
#if !_NO_WAIT_
            queue.wait();
#endif
        }


        template<typename T>
        void poly_coeff_sub_scalar_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n, const T* oprnd1,
            const T* scalar,
            const T* modulus, T* result) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_sub_scalar_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(0);
                    xehe::kernels::kernel_coeff_sub_scalar_mod<T>(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, scalar, modulus,
                        result);
                    });
                });
            EventCollector::add_event("kernel_coeff_sub_scalar_mod", e);
#if !_NO_WAIT_
                queue.wait();
#endif
        }


        template<typename T>
        void poly_coeff_sub_scalar_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T>& oprnd1,
            T scalar,
            const cl::sycl::buffer<T>& modulus,
            cl::sycl::buffer<T>& result) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aOp1 = ((cl::sycl::buffer<T> &)oprnd1).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);

                h.parallel_for<class krnl_coeff_sub_scalar_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(0);
                    T scalar_val = scalar;
                    xehe::kernels::kernel_coeff_sub_scalar_mod<T>(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        aOp1.get_pointer().get(),
                        &scalar_val,
                        aMod.get_pointer().get(),
                        aRes.get_pointer().get());
                });
            });
            EventCollector::add_event("kernel_coeff_sub_scalar_mod", e);
#if !_NO_WAIT_
                queue.wait();
#endif
        }



        template<typename T>
        class krnl_precompute_op_mod;

        template<typename T>
        void
            precompute_op_mod(cl::sycl::queue& queue,
                T* op_mod,
                T* op_byinv_mod,
                const T* op,
                const T* modulus,
                const T* inv_mod,
                int count,
                bool wait = false) {

            auto grid_range = sycl::range<1>(count);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_precompute_op_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int i = it.get_id(0);

                    xehe::kernels::kernel_precompute_op_mod<T>(i, op_mod, op_byinv_mod, *op, modulus, inv_mod);

                });
            });//.wait();
            EventCollector::add_event("kernel_precompute_op_mod", e);
            
            if (wait) {
                queue.wait();
            }
        }

        template<typename T>
        void
            poly_coeff_mul_scalar_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
                const T* oprnd1, const T* scalar,
                const T* modulus, const T* inv_mod,
                T* result) {
            //NOTE: scalar is a pointer


            T* op2_mod = nullptr;
        
            T* op2_byinv_mod = nullptr;
     

            {

                op2_mod = cl::sycl::malloc_device<T>(q_base_size, queue);
                op2_byinv_mod = cl::sycl::malloc_device<T>(q_base_size, queue);
    
            }
          
#if 1

            xehe::native::precompute_op_mod<T>(queue, op2_mod, op2_byinv_mod, scalar, modulus, inv_mod, q_base_size);
#else

            std::vector<T> v_op2_mod(q_base_size);
            std::vector<T> v_op2_byinv_mod(q_base_size);
            //std::cout << "Ptr" << std::endl;
            for (int i = 0; i < q_base_size; ++i)
            {
                v_op2_mod[i] = xehe::native::barrett_reduce(*scalar, modulus[i], inv_mod[i]);
                v_op2_byinv_mod[i] = xehe::native::op_by_mod_inverse(v_op2_mod[i], modulus[i]);

                //std::cout << v_op2_mod[i] << " " << v_op2_byinv_mod[i] << " " << modulus[i] << " " <<  inv_mod[i] << std::endl;
            }
            xehe::gpu_copy(queue,op2_mod, v_op2_mod.data(), v_op2_mod.size());
            xehe::gpu_copy(queue, op2_byinv_mod, v_op2_byinv_mod.data(), v_op2_byinv_mod.size());
#endif
                auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

                auto e = queue.submit([&](cl::sycl::handler& h) {
                    h.parallel_for<class krnl_coeff_mul_scalar_mod<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        int idx = it.get_id(2);
                        int rns_idx = it.get_id(1);
                        int poly_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_mul_scalar_mod<T>(idx, rns_idx, poly_idx,
                            n, q_base_size, n_polys,
                            oprnd1, op2_mod, modulus, op2_byinv_mod,
                            result);
                        });
                    });
                    
                EventCollector::add_event("kernel_coeff_mul_scalar_mod", e);
#if !_NO_WAIT_
                    queue.wait();
#endif


                if (op2_mod != nullptr)
                {
                    cl::sycl::free(op2_mod, queue);
                    cl::sycl::free(op2_byinv_mod, queue);
                    op2_mod = nullptr;
                    op2_byinv_mod = nullptr;
                }

        }


        template<typename T>
        class krnl_precompute_op_mod_buf;

        template<typename T>
        void
            precompute_op_mod(cl::sycl::queue& queue,
                cl::sycl::buffer<T> &op_mod,
                cl::sycl::buffer<T> &op_byinv_mod,
                T op,
                const cl::sycl::buffer<T> &modulus,
                const cl::sycl::buffer<T>& inv_mod,
                int count) {

            auto grid_range = sycl::range<1>(count);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod_inv = ((cl::sycl::buffer<T> &)inv_mod).template get_access<cl::sycl::access::mode::read>(h);
                auto aOp_mod = op_mod.template get_access<cl::sycl::access::mode::write>(h);
                auto aOp_byinv_mod = op_byinv_mod.template get_access<cl::sycl::access::mode::write>(h);


                h.parallel_for<class krnl_precompute_op_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        int i = it.get_id(0);
                        xehe::kernels::kernel_precompute_op_mod<T>(i,
                            aOp_mod.get_pointer().get(),
                            aOp_byinv_mod.get_pointer().get(),
                            op,
                            aMod.get_pointer().get(),
                            aMod_inv.get_pointer().get());

                    });
                });
            EventCollector::add_event("krnl_precompute_op_mod_buf", e);
#if !_NO_WAIT_
                queue.wait();
#endif
        }


        template<typename T>
        void
            poly_coeff_mul_scalar_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
                const cl::sycl::buffer<T>& oprnd1, T scalar,
                const cl::sycl::buffer<T>& modulus, const cl::sycl::buffer<T>& inv_mod,
                cl::sycl::buffer<T>& result, bool wait = false) {
            //NOTE: scalar is a pointer


            cl::sycl::buffer<T> op2_mod{ cl::sycl::range<1>{(1024)} };
            cl::sycl::buffer<T> op2_byinv_mod{ cl::sycl::range<1>{(1024)} };

            xehe::native::precompute_op_mod<T>(queue, op2_mod, op2_byinv_mod, scalar, modulus, inv_mod, q_base_size);

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aOp1 = ((cl::sycl::buffer<T> &)oprnd1).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aOp2_mod = op2_mod.template get_access<cl::sycl::access::mode::read>(h);
                auto aOp2_byinv_mod = op2_byinv_mod.template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::write>(h);

                h.parallel_for<class krnl_coeff_mul_scalar_mod_buf<T>>({ grid_range }, [=](auto it)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        int idx = it.get_id(2);
                        int rns_idx = it.get_id(1);
                        int poly_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_mul_scalar_mod<T>(idx, rns_idx, poly_idx,
                            n, q_base_size, n_polys,
                            aOp1.get_pointer().get(),
                            aOp2_mod.get_pointer().get(),
                            aMod.get_pointer().get(),
                            aOp2_byinv_mod.get_pointer().get(),
                            aRes.get_pointer().get());
                    });
                });//.wait();
            EventCollector::add_event("krnl_coeff_mul_scalar_mod_buf", e);
            
            if (wait) {
                queue.wait();
            }
        }

        template<typename T>
        void poly_coeff_prod_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const T* oprnd1, const T* oprnd2,
            const T* modulus, const T* inv_mod2,
            T* result, bool wait = false) {
#if _INF_COMP_
            auto grid_range = sycl::range<3>(1,1,1); 
#else
            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);
#endif            

            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_prod_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(0);
                    xehe::kernels::kernel_coeff_prod_mod<T>(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus,
                        inv_mod2,
                        result);
                    });
                });
            EventCollector::add_event("kernel_coeff_prod_mod", e);
            if (wait) {
                queue.wait();
            }
        }

        template<typename T>
        void poly_coeff_prod_mod_plain(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const T* oprnd1, const T* oprnd2,
            const T* modulus, const T* inv_mod2,
            T* result, bool wait = false) {
#if _INF_COMP_
            auto grid_range = sycl::range<2>(1,1); 
#else
            auto grid_range = sycl::range<2>(q_base_size, n);
#endif
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_prod_mod_plain<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(1);
                    int rns_idx = it.get_id(0);
                    xehe::kernels::kernel_coeff_prod_mod_plain<T>(idx, rns_idx,
                        n, q_base_size, n_polys,
                        oprnd1, oprnd2, modulus,
                        inv_mod2,
                        result);
                    });
                });
            EventCollector::add_event("kernel_coeff_prod_mod_plain", e);
            if (wait) {
                queue.wait();
            }
        }

        template<typename T>
        void poly_coeff_prod_mod_plain_add(cl::sycl::queue& queue, int n_polys, int n_poly_add, int n_poly_mul,
                                           int q_base_size, int n,
                                           const T* oprnd_add, const T* oprnd_mul, const T* oprnd_plain,
                                           const T* modulus, const T* inv_mod2,
                                           T* result, bool wait = false) {
            if (n_poly_add == n_poly_mul) {
#if _INF_COMP_
                auto grid_range = sycl::range<2>(1,1);
#else
                auto grid_range = sycl::range<2>(q_base_size, n);
#endif
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_prod_mod_plain_add1<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(1);
                        int rns_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_prod_mod_plain_add<T>(idx, rns_idx,
                                                                          n, q_base_size, n_polys,
                                                                          oprnd_add, oprnd_mul, oprnd_plain, modulus,
                                                                          inv_mod2,
                                                                          result);
                    });
                });
                EventCollector::add_event("kernel_coeff_prod_mod_plain_add", e);
            }
            else{
                if (n_poly_add < n_poly_mul){
                    int n_poly_min = n_poly_add;
                    int n_poly_max = n_poly_mul;
                    auto grid_range = sycl::range<2>(q_base_size, n);
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_prod_mod_plain_add2<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(1);
                            int rns_idx = it.get_id(0);
                            xehe::kernels::kernel_coeff_prod_mod_plain_add_1<T>(idx, rns_idx,
                                                                            n, q_base_size, n_poly_min,
                                                                            oprnd_add, oprnd_mul, oprnd_plain, modulus,
                                                                            inv_mod2,
                                                                            result, n_poly_max);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_prod_mod_plain_add_1", e);
                }else{
                    int n_poly_min = n_poly_mul;
                    int n_poly_max = n_poly_add;
                    auto grid_range = sycl::range<2>(q_base_size, n);
                    auto e = queue.submit([&](cl::sycl::handler &h) {
                        h.parallel_for<class krnl_coeff_prod_mod_plain_add3<T>>({grid_range}, [=](auto it)
                                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                            int idx = it.get_id(1);
                            int rns_idx = it.get_id(0);
                            xehe::kernels::kernel_coeff_prod_mod_plain_add_2<T>(idx, rns_idx,
                                                                            n, q_base_size, n_poly_min,
                                                                            oprnd_add, oprnd_mul, oprnd_plain, modulus,
                                                                            inv_mod2,
                                                                            result, n_poly_max);
                        });
                    });
                    EventCollector::add_event("kernel_coeff_prod_mod_plain_add_2", e);
                }
            }
            if (wait){
                queue.wait();
            }
        }

        template<typename T>
        void poly_coeff_prod_mod_add(cl::sycl::queue& queue, int n_polys, 
                                     int n_poly_add, int n_poly2, int n_poly3,
                                     int q_base_size, int n,
                                     const T* oprnd_add, const T* oprnd2, const T* oprnd3,
                                     const T* modulus, const T* inv_mod2,
                                     T* result, bool wait = false) {
            if ((n_poly2 * n_poly3 - 1) == n_poly_add) {
#if _INF_COMP_
                auto grid_range = sycl::range<2>(1,1);
#else
                auto grid_range = sycl::range<2>(q_base_size, n);
#endif
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_prod_mod_add1<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(1);
                        int rns_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_prod_mod_add<T>(idx, rns_idx,
                                                                    n, q_base_size, n_polys,
                                                                    oprnd_add, oprnd2, oprnd3, modulus,
                                                                    inv_mod2,
                                                                    result);
                    });
                });
                EventCollector::add_event("kernel_coeff_prod_mod_add", e);
            }
            else if (n_polys > 3){
                std::cout<<"invalid input: n_polys cannot exceed 3.\n"<<std::endl;
                return;
            }else if (n_poly_add == 2){
                auto grid_range = sycl::range<2>(q_base_size, n);
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_prod_mod_add2<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(1);
                        int rns_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_prod_mod_add_2<T>(idx, rns_idx,
                                                                    n, q_base_size, n_polys,
                                                                    oprnd_add, oprnd2, oprnd3, modulus,
                                                                    inv_mod2,
                                                                    result);
                    });
                });
                EventCollector::add_event("kernel_coeff_prod_mod_add_2", e);
            }else if (n_poly_add == 1){
                auto grid_range = sycl::range<2>(q_base_size, n);
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_prod_mod_add3<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(1);
                        int rns_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_prod_mod_add_1<T>(idx, rns_idx,
                                                                    n, q_base_size, n_polys,
                                                                    oprnd_add, oprnd2, oprnd3, modulus,
                                                                    inv_mod2,
                                                                    result);
                    });
                });
                EventCollector::add_event("kernel_coeff_prod_mod_add_1", e);
            }else if (n_poly_add == 0){
                auto grid_range = sycl::range<2>(q_base_size, n);
                auto e = queue.submit([&](cl::sycl::handler &h) {
                    h.parallel_for<class krnl_coeff_prod_mod_add4<T>>({grid_range}, [=](auto it)
                            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] {
                        int idx = it.get_id(1);
                        int rns_idx = it.get_id(0);
                        xehe::kernels::kernel_coeff_prod_mod_add<T>(idx, rns_idx,
                                                                    n, q_base_size, n_polys,
                                                                    oprnd2, oprnd3, modulus,
                                                                    inv_mod2,
                                                                    result);
                    });
                });
                EventCollector::add_event("kernel_coeff_prod_mod_add", e);
            }
            if (wait){
                queue.wait();
            }
        }

        template<typename T>
        void poly_coeff_prod_mod(cl::sycl::queue& queue, int n_polys, int q_base_size, int n,
            const cl::sycl::buffer<T>&  oprnd1, const cl::sycl::buffer<T>&  oprnd2,
            const cl::sycl::buffer<T>&  modulus, const cl::sycl::buffer<T>&  inv_mod2,
            cl::sycl::buffer<T>&  result) {

            auto grid_range = sycl::range<3>(n_polys, q_base_size, n);

            auto e = queue.submit([&](cl::sycl::handler& h) {
                auto aOp1 = ((cl::sycl::buffer<T> &)oprnd1).template get_access<cl::sycl::access::mode::read>(h);
                auto aOp2 = ((cl::sycl::buffer<T> &)oprnd2).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod = ((cl::sycl::buffer<T> &)modulus).template get_access<cl::sycl::access::mode::read>(h);
                auto aMod_inv2 = ((cl::sycl::buffer<T> &)inv_mod2).template get_access<cl::sycl::access::mode::read>(h);
                auto aRes = result.template get_access<cl::sycl::access::mode::discard_write>(h);
                h.parallel_for<class krnl_coeff_prod_mod_buf<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(2);
                    int rns_idx = it.get_id(1);
                    int poly_idx = it.get_id(0);
                    xehe::kernels::kernel_coeff_prod_mod<T>(idx, rns_idx, poly_idx,
                        n, q_base_size, n_polys,
                        aOp1.get_pointer().get(),
                        aOp2.get_pointer().get(),
                        aMod.get_pointer().get(),
                        aMod_inv2.get_pointer().get(),
                        aRes.get_pointer().get());

                    });
                });
            EventCollector::add_event("kernel_coeff_prod_mod", e);
#if !_NO_WAIT_
                queue.wait();
#endif

        }

        template<typename T>
        void ckks_coeff_fused_prod_mod(cl::sycl::queue& queue,
            int q_base_size, int n,
            const T* poly1, const T* poly2,
            const T* xe_modulus, const T* inv_mod2,
            T* poly_res,
            bool wait = false) {
            auto grid_range = sycl::range<2>(q_base_size, n);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_fused_prod_mod<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(1);
                    int rns_idx = it.get_id(0);
                    kernels::kernel_coeff_fused_prod_2x2_mod<T>(idx, rns_idx,
                        n, q_base_size,
                        poly1, poly2, xe_modulus,
                        inv_mod2,
                        poly_res);
                    });
                });
            EventCollector::add_event("kernel_coeff_fused_prod_2x2_mod", e);
            if (wait) {
                queue.wait();
            }
        }

        template<typename T>
        void ckks_coeff_square(cl::sycl::queue& queue, 
            int q_base_size, int n,
            T* poly,
            const T* xe_modulus, const T* inv_mod2, bool wait = false) {
            auto grid_range = sycl::range<2>(q_base_size, n);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class krnl_coeff_square<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    int idx = it.get_id(1);
                    int rns_idx = it.get_id(0);
                    kernels::kernel_coeff_fused_2_square<T>(idx, rns_idx,
                        n, q_base_size,
                        poly, xe_modulus,
                        inv_mod2);
                    });
                });//.wait();
            EventCollector::add_event("kernel_coeff_fused_2_square", e);
            if (wait){
                queue.wait();
            }
        }





#endif //BUILD_WITH_IGPU


    } // namespace util
} // namespace seal

#endif //XEHE_POLY_ARITH_SMALL_MOD_GPU_H
