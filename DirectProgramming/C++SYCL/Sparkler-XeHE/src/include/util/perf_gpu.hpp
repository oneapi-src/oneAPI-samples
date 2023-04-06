/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef _PERF_GPU_HPP_
#define _PERF_GPU_HPP_


#include <memory>
#include <array>
#include <iostream>
#include <iso646.h>
#include <random>

#include "native/xe_modulus.hpp"
#include "util/inline_kernels.hpp"
#include "lib_utils.h"
#include "util/rns_dwt_gpu.hpp"

#pragma once
#ifdef BUILD_WITH_IGPU

#define MUL2(z, y, x) z = ((y)*(x)); y = ((z)*(x));
#define MUL4(z, y, x) MUL2(z, y, x) \
                        MUL2(z, y, x)
#define MUL8(z, y, x) MUL4(z, y, x) \
                        MUL4(z, y, x)
#define MUL16(z, y, x) MUL8(z, y, x) \
                        MUL8(z, y, x)
#define MUL32(z, y, x) MUL16(z, y, x) \
                        MUL16(z, y, x)
#define MUL64(z, y, x) MUL32(z, y, x) \
                        MUL32(z, y, x)
#define MUL128(z, y, x) MUL64(z, y, x) \
                        MUL64(z, y, x)
#define MUL256(z, y, x) MUL128(z, y, x) \
                        MUL128(z, y, x)
#define MUL512(z, y, x) MUL256(z, y, x) \
                        MUL256(z, y, x)
#define MUL1024(z, y, x) MUL512(z, y, x) \
                        MUL512(z, y, x)

#define ADD2(z, y, x)  z = y+x; y = z-x;//{T tmp; z += add_uint64<T>(x, y, &tmp); z += tmp;} //z += y+x;
#define ADD4(z, y, x) ADD2(z, y, x) \
                        ADD2(z, y, x)
#define ADD8(z, y, x) ADD4(z, y, x) \
                        ADD4(z, y, x)
#define ADD16(z, y, x) ADD8(z, y, x) \
                        ADD8(z, y, x)
#define ADD32(z, y, x) ADD16(z, y, x) \
                        ADD16(z, y, x)
#define ADD64(z, y, x) ADD32(z, y, x) \
                        ADD32(z, y, x)
#define ADD128(z, y, x) ADD64(z, y, x) \
                        ADD64(z, y, x)
#define ADD256(z, y, x) ADD128(z, y, x) \
                        ADD128(z, y, x)
#define ADD512(z, y, x) ADD256(z, y, x) \
                        ADD256(z, y, x)
#define ADD1024(z, y, x) ADD512(z, y, x) \
                        ADD512(z, y, x)


#define MULMOD2(z, y, x, m, im) z = xehe::native::barrett_reduce<T>(((y) * (x)), m, im);  y = xehe::native::barrett_reduce<T>(((z) * (x)), m, im);
#define MULMOD4(z, y, x, m, im) MULMOD2(z, y, x, m, im) \
                        MULMOD2(z, y, x, m, im)
#define MULMOD8(z, y, x, m, im) MULMOD4(z, y, x, m, im) \
                        MULMOD4(z, y, x, m, im)
#define MULMOD16(z, y, x, m, im) MULMOD8(z, y, x, m, im) \
                        MULMOD8(z, y, x, m, im)
#define MULMOD32(z, y, x, m, im) MULMOD16(z, y, x, m, im) \
                        MULMOD16(z, y, x, m, im)
#define MULMOD64(z, y, x, m, im) MULMOD32(z, y, x, m, im) \
                        MULMOD32(z, y, x, m, im)
#define MULMOD128(z, y, x, m, im) MULMOD64(z, y, x, m, im) \
                        MULMOD64(z, y, x, m, im)
#define MULMOD256(z, y, x, m, im) MULMOD128(z, y, x, m, im) \
                        MULMOD128(z, y, x, m, im)
#define MULMOD512(z, y, x, m, im) MULMOD256(z, y, x, m, im) \
                        MULMOD256(z, y, x, m, im)
#define MULMOD1024(z, y, x, m, im) MULMOD512(z, y, x, m, im) \
                        MULMOD512(z, y, x, m, im)


#define ADDMOD2(z, y, x, m) z = xehe::native::add_mod(y, x, m); y = xehe::native::add_mod(z, x, m);
#define ADDMOD4(z, y, x, m) ADDMOD2(z, y, x, m) \
                        ADDMOD2(z, y, x, m)
#define ADDMOD8(z, y, x, m) ADDMOD4(z, y, x, m) \
                        ADDMOD4(z, y, x, m)
#define ADDMOD16(z, y, x, m) ADDMOD8(z, y, x, m) \
                        ADDMOD8(z, y, x, m)
#define ADDMOD32(z, y, x, m) ADDMOD16(z, y, x, m) \
                        ADDMOD16(z, y, x, m)
#define ADDMOD64(z, y, x, m) ADDMOD32(z, y, x, m) \
                        ADDMOD32(z, y, x, m)
#define ADDMOD128(z, y, x, m) ADDMOD64(z, y, x, m) \
                        ADDMOD64(z, y, x, m)
#define ADDMOD256(z, y, x, m) ADDMOD128(z, y, x, m) \
                        ADDMOD128(z, y, x, m)
#define ADDMOD512(z, y, x, m) ADDMOD256(z, y, x, m) \
                        ADDMOD256(z, y, x, m)
#define ADDMOD1024(z, y, x, m) ADDMOD512(z, y, x, m) \
                        ADDMOD512(z, y, x, m)

#define MUL2_2(x, r0, r2) r2[0] = xehe::native::mul_uint<T>(x, r0, r2+1); r0 = xehe::native::mul_uint<T>(x, r2[1], &r2[0]); 
                        
#define MUL2_4(x, r0, r2) MUL2_2(x, r0, r2) \
                        MUL2_2(x, r0, r2)
#define MUL2_8(x, r0, r2) MUL2_4(x, r0, r2) \
                        MUL2_4(x, r0, r2)
#define MUL2_16(x, r0, r2) MUL2_8(x, r0, r2) \
                        MUL2_8(x, r0, r2)
#define MUL2_32(x, r0, r2) MUL2_16(x, r0, r2) \
                        MUL2_16(x, r0, r2)
#define MUL2_64(x, r0, r2) MUL2_32(x, r0, r2) \
                        MUL2_32(x, r0, r2)
#define MUL2_128(x, r0, r2) MUL2_64(x, r0, r2) \
                        MUL2_64(x, r0, r2)
#define MUL2_256(x, r0, r2) MUL2_128(x, r0, r2) \
                        MUL2_128(x, r0, r2)
#define MUL2_512(x, r0, r2) MUL2_256(x, r0, r2) \
                        MUL2_256(x, r0, r2)
#define MUL2_1024(x, r0, r2) MUL2_512(x, r0, r2) \
                        MUL2_512(x, r0, r2)


#include <stdexcept>

namespace xehe
{
    namespace util
    {
        template <typename T>
        cl::sycl::buffer<T> &getBuffer(std::shared_ptr<cl::sycl::buffer<T>> buf) {
            return *buf;
  }

        template <typename T>
        class Mul2Mod {
        public:
            Mul2Mod(const T* A, const T* B, const T* modulus, const T* inv_mod2, size_t n, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                modulus_ = modulus;
                inv_mod2_ = inv_mod2;
                n_ = n;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                //auto p = ind / n_;
                uint64_t p = 0;
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                T r = 0;
                auto m = modulus_[p];
                // 128bit inverse
                const T* inv_mod = &inv_mod2_[p*2];
                for (int i = 0; i < loop_sz_/2; i++)
                {
                    r = xehe::native::mul_mod<T>(a, b, m, inv_mod); b = xehe::native::mul_mod(a, r, m, inv_mod);
                }
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            const T* modulus_;
            const T* inv_mod2_;
            int loop_sz_;
            size_t n_;
        };


        template <typename T>
        class Mul2 {
        public:
            Mul2(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                T r[2] = { b,a };
                MUL2_256(a, b, r);
#if 0
                for (int i = 0; i < loop_sz_; i++)
                {
                    multiply_uint64_generic<T>(a, r, r_tmp);
                    r += r_tmp[0];
                }
#endif

                R_[i] = r[1];
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };

        template <typename T>
        class MulMod {
        public:
            MulMod(const T* A, const T* B, const T* modulus, const T* mod_inv, size_t n, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                modulus_ = modulus;
                mod_inv_ = mod_inv;
                n_ = n;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                //auto p = ind / n_;
                uint64_t p = 0;
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                auto m = modulus_[p];
                auto inv_mod = mod_inv_[p];
                T r = 0;
                MULMOD256(r, a, b, m, inv_mod);
#if 0
                for (int i = 0; i < loop_sz_; i++)
                {
                    r += barrett_reduce_64<T>((a * r), m, inv_mod);
                }
#endif
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            const T* modulus_;
            const T* mod_inv_;
            int loop_sz_;
            size_t n_;
        };


        template <typename T>
        class Mul {
        public:
            Mul(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                T r = 0;
                MUL256(r, a, b)
#if 0
                for (int i = 0; i < loop_sz_; i++)
                {
                    r += (a * r);
                }
#endif
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };


        template <typename T>
        class AddMod {
        public:
            AddMod(const T* A, const T* B, const T* modulus, size_t n, int loop_sz,  T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                modulus_ = modulus;
                n_ = n;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                //auto p = ind / n_;
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                auto m = modulus_[0];
                T r = 0;
                ADDMOD256(r, a, b, m);
#if 0
                for (int i = 0; i < loop_sz_; i++)
                {
                    r += add_uint_mod(a, r, m);
                }
#endif
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            const T* modulus_;
            int loop_sz_;
            size_t n_;
        };


        template <typename T>
        class Add {
        public:
            Add(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                T r = 0;
                ADD256(r, a, b);
#if 0
                for (int i = 0; i < loop_sz_; i++)
                {
                    r += (a + r);
                }
#endif
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };


        template <typename T>
        class CalibrateFloatPerf {
        public:
            CalibrateFloatPerf(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                uint64_t i = ind;
                auto a = float(A_[i]);
                auto b = float(B_[i]);
                float r = 0;
                MUL1024(r,a,b);
#if 0
                for (int i = 0; i < loop_sz_; i++)
                {
                    r += (a * r);
                }
#endif
                R_[i] += T(r);
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };


        /**********************************************************************
        
         PERF CLASS

        *************************************************************************/

        template <typename T>
        class BasePerf {
        public:
            BasePerf() {

                log_n_ = 1;
                n_ = (size_t(1) << log_n_);

                outer_loop_ = 1;
                inner_loop_ = 1;
                q_base_sz_ = 1;
               

                //queue_ = 0;
            }

            BasePerf(int outer_loop, int inner_loop, int q_base_sz, int log_n) {

                init(outer_loop, inner_loop, q_base_sz, log_n);

            }

            const cl::sycl::queue& get_queue(void)
            {
                return (queue_);
            }

            void init(int outer_loop, int inner_loop, int q_base_sz, int log_n)
            {
                log_n_ = log_n;
                n_ = (size_t(1) << log_n_);

                outer_loop_ = outer_loop;
                inner_loop_ = inner_loop;
                q_base_sz_ = q_base_sz;

                xehe::dpcpp::Context ctx;
                queue_ = ctx.queue();
                range_ = q_base_sz_ * n_;
                A_ = cl::sycl::malloc_shared<T>(range_, queue_);
                B_ = cl::sycl::malloc_shared<T>(range_, queue_);
                C_ = cl::sycl::malloc_shared<T>(range_, queue_);
                R_ = cl::sycl::malloc_shared<T>(2*range_, queue_);

                std::vector<T> xe_modulus;

                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<T> dis;

                for (int i = 0; i < q_base_sz_;)
                {

                    auto m = T(dis(gen) & ((sizeof(T) == 4)? 0x7fffffff : 0x7fffffffffffffffUL));
                    if (m != 0)
                    {
                        xe_modulus.push_back(m);
                        ++i;
                    }
                }

                modulo_ = cl::sycl::malloc_shared<T>(q_base_sz_, queue_);
                inv_mod_ = cl::sycl::malloc_shared<T>(q_base_sz_, queue_);
                inv_mod2_ = cl::sycl::malloc_shared<T>(2 * q_base_sz_, queue_);


                std::memcpy((void*)modulo_, xe_modulus.data(), q_base_sz_ * sizeof(T));
                for (int j = 0; j < q_base_sz_; ++j)
                {
                    ((T*)inv_mod_)[j] = xehe::native::mod_inverse1(xe_modulus[j]);
                    ((T*)inv_mod2_)[2 * j] = xehe::native::mod_inverse2(xe_modulus[j], &((T*)inv_mod2_)[2 * j + 1]);
                    for (int i = 0; i < n_; i++) {
                        A_[i + j * n_] = std::rand() % xe_modulus[j];
                        B_[i + j * n_] = std::rand() % xe_modulus[j];
                        C_[i + j * n_] = std::rand() % xe_modulus[j];
                    }
                }

            }

            void calibrate_float_perf(double& duration_count)
            {


                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(1), CalibrateFloatPerf<T>(A_, B_, inner_loop_, R_));
                    }).wait();

                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 0; i < outer_loop_; i++) {
                        queue_.submit([&](sycl::handler& h) {
                            h.parallel_for<>(sycl::range<1>(range_), CalibrateFloatPerf<T>(A_, B_, inner_loop_, R_));
                            });
                    }
                    queue_.wait();

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    duration_count = duration.count();
            }

            void add_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < outer_loop_; i++) {
                    queue_.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_), Add<T>(A_, B_, inner_loop_, R_));
                        });
                }
                queue_.wait();

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }


            void add_mod_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();
                
                for (int i = 0; i < outer_loop_; i++) {
                    queue_.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_), AddMod<T>(A_, B_, modulo_, n_, inner_loop_, R_));
                        });
                }
                queue_.wait();
                
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }

            void mul_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < outer_loop_; i++) {
                    queue_.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_), Mul<T>(A_, B_, inner_loop_, R_));
                        });
                }
                queue_.wait();

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }


            void mul_mod_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < outer_loop_; i++) {
                    queue_.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_), MulMod<T>(A_, B_, modulo_, inv_mod_, n_, inner_loop_, R_));
                        });
                }
                queue_.wait();

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }

            void mul2_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < outer_loop_; i++) {
                    queue_.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_), Mul2<T>(A_, B_, inner_loop_, R_));
                        });
                }
                queue_.wait();

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }

            void mul2_mod_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < outer_loop_; i++) {
                    queue_.submit([&](sycl::handler& h) {
                        h.parallel_for<>(sycl::range<1>(range_), Mul2Mod<T>(A_, B_, modulo_, inv_mod2_, n_, 128, R_));
                        });
                }
                queue_.wait();

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }

            void ntt_negacyclic_forward_perf(double& duration_count)
            {
                auto start = std::chrono::high_resolution_clock::now();
                #if 0
                for (int i = 0; i < outer_loop_; i++) {
                    ntt_negacyclic_harvey<T, MultiplyUIntModOperand<T>, MultiplyUIntModOperand<T>>(queue_, int(q_base_sz_),
                        log_n_, A_,
                        modulo_, roots_);
                }
                #endif
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }

            void mul_native_perf(double& duration_count)
            {

                auto start = std::chrono::high_resolution_clock::now();

                //gpu mul
                queue_.submit([&](sycl::handler& h) 
                {
                    auto A = A_;
                    auto B = B_;
                    auto R = R_;

                    h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        R[2*ind] = xehe::native::mul_uint<T>(A[ind], B[ind], R + 2*ind + 1);
                    }
                    );
                });
                queue_.wait();

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                duration_count = duration.count();
            }

        protected:
            cl::sycl::queue queue_;
            size_t n_;
            int log_n_;
            int q_base_sz_;
            uint64_t range_;
            const T* modulo_;
            const T* inv_mod_;
            const T* inv_mod2_;
            T* A_;
            T* B_;
            T* C_;
            T* R_;
            int outer_loop_;
            int inner_loop_;
        };


        /**********************************************************************

         CODE DUMP

        *************************************************************************/
// #if 0

        template <typename T>
        class Mul2ModCode {
        public:
            Mul2ModCode(const T* A, const T* B, const T* modulus, const T* inv_mod2, size_t n, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                modulus_ = modulus;
                inv_mod2_ = inv_mod2; 
                n_ = n;
                loop_sz_ = loop_sz;
            }

            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                //auto p = ind / n_;
                uint64_t p = 0;
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                auto m = modulus_[p];
                const T* inv_mod = &inv_mod2_[p*2];
                T r = b;
                for (int i = 0; i < loop_sz_/2; i++)
                {
                    // r = xehe::native::mul_mod<T>(a, b, m, inv_mod);
                    r = xehe::native::mul_mod<T>(a, r, m, inv_mod);
                }
                //auto r = multiply_uint_mod(a, b, m, inv_mod);
                R_[i] = r;
            }

        protected:


            const T* A_;
            const T* B_;
            T* R_;
            const T* modulus_;
            const T* inv_mod2_;
            int loop_sz_;
            size_t n_;
        };


        template <typename T>
        class Mul2Code {
        public:
            Mul2Code(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                T r[2] = { b,a };
                // T r_tmp[2];
                // MUL2_256(a, b, r);
                for (int i = 0; i < loop_sz_; i++)
                {
                    // multiply_uint64_generic<T>(a, r, r_tmp);
                    r[0] = xehe::native::mul_uint<T>(a, r[1], &r[1]);
                }

                R_[i] = r[1];
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };

        template <typename T>
        class MulModCode {
        public:
            MulModCode(const T* A, const T* B, const T* modulus, const T* mod_inv, size_t n, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                modulus_ = modulus;
                mod_inv_ = mod_inv;
                n_ = n;
                loop_sz_ = 3;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                //auto p = ind / n_;
                uint64_t p = 0;
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                auto m = modulus_[p];
                auto inv_mod = mod_inv_[p]; 
                T r = b;
                // MULMOD256(r, a, b, m, inv_mod);
                for (int i = 0; i < loop_sz_; i++)
                {
                    r += xehe::native::barrett_reduce<T>((a * r), m, inv_mod);
                }
                //auto r = barrett_reduce_64<T>((a * b), m, inv_mod);
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            const T* modulus_;
            const T* mod_inv_;
            int loop_sz_;
            size_t n_;
        };


        template <typename T>
        class MulCode {
        public:
            MulCode(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
                T r = a;
                // MUL256(r, a, b)
                for (int i = 0; i < loop_sz_; i++)
                {
                    // r += (a * r);
                    r = r * b;
                }
                //auto r = (a * b);
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };


        template <typename T>
        class AddModCode {
        public:
            AddModCode(const T* A, const T* B, const T* modulus, size_t n, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                modulus_ = modulus;
                n_ = n;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const {
                //auto p = ind / n_;
                uint64_t i = ind;
                auto a = A_[i];
                auto b = B_[i];
         
                auto m = modulus_[0];
                T r = b;
                // ADDMOD256(r, a, b, m);
                for (int i = 0; i < loop_sz_; i++)
                {
                    // r += add_uint_mod(a, r, m);
                    r += xehe::native::add_mod<T>(a, r, m);
                }
                //auto r = add_uint_mod(a, b, m);
                R_[i] = r;
            }

        protected:

            const T* A_;
            const T* B_;
            T* R_;
            const T* modulus_;
            int loop_sz_;
            size_t n_;
        };

        template <typename T>
        class AddCode {
        public:
            AddCode(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const 
            {
                uint64_t i = ind;
                // auto a = A_[i];
                // auto b = B_[i];
                // T r = 0;
                // T tmp;
                // ADD256(r, a, b);
                for (int j = 0; j < loop_sz_; j++)
                {
                    // r += (a + r);
                xehe::native::add_uint<T>(A_[i],B_[i],&(R_[i]));
                }
                // R_[i] = r;
            }
        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };

        template <typename T>
        class InlineAddCode {
        public:
            InlineAddCode(const T* A, const T* B, int loop_sz, T* R)
            {
                A_ = A;
                B_ = B;
                R_ = R;
                loop_sz_ = loop_sz;
            }
            void operator() [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]] (cl::sycl::id<1> ind) const 
            {
                
                // uint64_t i = ind;
                // auto a = A_[i];
                // auto b = B_[i];
                // T r = 0;
 
                for (int j = 0; j < loop_sz_; j++)
                {
#ifdef __SYCL_DEVICE_ONLY__
                    asm("add (M1, 16) %0(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>"
                        : "=rw"(R_[ind])
                        : "rw"(A_[ind]), "rw"(B_[ind])
                    );
#endif
                }
            }
        protected:

            const T* A_;
            const T* B_;
            T* R_;
            int loop_sz_;
        };

// #endif

        template <typename T>
        class BaseAsmDump : public BasePerf<T> {
        public:
            //from BasePerf class
            BaseAsmDump(int q_base_sz, int log_n)
            : BasePerf<T>()
            {
                BasePerf<T>::init(1, 1, q_base_sz, log_n);
            }

            //using modulus and const_ratio obtained from modulus.h
            BaseAsmDump(int q_base_sz, int log_n, const std::vector<xehe::native::Modulus<T>> &modulus)
            {
                log_n_ = log_n;
                n_ = (size_t(1) << log_n_);

                outer_loop_ = 1;
                inner_loop_ = 1;
                q_base_sz_ = q_base_sz;


                xehe::dpcpp::Context ctx;
                queue_ = ctx.queue();
                range_ = q_base_sz_ * n_;
                A_ = cl::sycl::malloc_shared<T>(range_, queue_);
                B_ = cl::sycl::malloc_shared<T>(range_, queue_);
                C_ = cl::sycl::malloc_shared<T>(range_, queue_);
                R_ = cl::sycl::malloc_shared<T>(range_, queue_);

                std::vector<T> xe_modulus;
                for (int i = 0; i < q_base_sz; ++i)
                {
                    xe_modulus.push_back(modulus[i].value());
                    // auto temp = gen_modulus[i].const_ratio().data();
                    // for (int j = 0; j < const_ratio_sz; ++j)
                    // {
                    //     xe_const_ratio.push_back(temp[i]);
                    // }
                }

                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<T> dis;

                modulo_ = cl::sycl::malloc_shared<T>(q_base_sz_, queue_);
                inv_mod_ = cl::sycl::malloc_shared<T>(q_base_sz_, queue_);
                inv_mod2_ = cl::sycl::malloc_shared<T>(2 * q_base_sz_, queue_);


                std::memcpy((void*)modulo_, xe_modulus.data(), q_base_sz_ * sizeof(T));
                
                const T * const_ratio;
                for (int j = 0; j < q_base_sz_; ++j)
                {
                    const_ratio = modulus[j].const_ratio().data();
                    ((T*)inv_mod_)[j] = const_ratio[0];
                    ((T*)inv_mod2_)[2 * j] = const_ratio[1];
                    // ((T*)inv_mod_)[j] = xehe::native::mod_inverse1(xe_modulus[j]);
                    // ((T*)inv_mod2_)[2 * j] = xehe::native::mod_inverse2(xe_modulus[j], &((T*)inv_mod2_)[2 * j + 1]);
                    for (int i = 0; i < n_; i++) {
                        A_[i + j * n_] = std::rand() % xe_modulus[j];
                        B_[i + j * n_] = std::rand() % xe_modulus[j];
                        C_[i + j * n_] = std::rand() % xe_modulus[j];
                    }
                }

            }

            void add_code(void)
            {
                
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), AddCode<T>(A_, B_, 1, R_));
                    });
                queue_.wait();
            }


            void add_mod_code(void)
            {
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), AddModCode<T>(A_, B_, modulo_, n_, 1, R_));
                    });
                queue_.wait();
            }

            void mul_code(void)
            {
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), MulCode<T>(A_, B_, 1, R_));
                    });
                queue_.wait();

            }

            void mul_mod_code(void)
            {
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), MulModCode<T>(A_, B_, modulo_, inv_mod_, n_, 1, R_));
                    });
                queue_.wait();
            }

            void mul2_code(void)
            {
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), Mul2Code<T>(A_, B_, 1, R_));
                    });
                queue_.wait();
            }

            void mul2_mod_code(void)
            {
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), Mul2ModCode<T>(A_, B_, modulo_, inv_mod2_, n_, 1, R_));
                    });
                queue_.wait();
            }

            void inline_add_code(void)
            {
                queue_.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::range<1>(range_), InlineAddCode<T>(A_, B_, 1, R_));
                    });
                queue_.wait();
            }

        protected:
       
            cl::sycl::queue queue_;
            size_t n_;
            int log_n_;
            int q_base_sz_;
            uint64_t range_;
            const T* modulo_;
            const T* inv_mod_;
            const T* inv_mod2_;
            T* A_;
            T* B_;
            T* C_;
            T* R_;
            int outer_loop_;
            int inner_loop_;
        };
        
        
        template <typename T>
        class BaseInlineAsm : public BasePerf<T> {
        public:
            //from BasePerf class
            BaseInlineAsm(int q_base_sz, int log_n)
            {
                init(1, 1, q_base_sz, log_n);
            }

            void init(int outer_loop, int inner_loop, int q_base_sz, int log_n)
            {
                log_n_ = log_n;
                n_ = (size_t(1) << log_n_);

                outer_loop_ = outer_loop;
                inner_loop_ = inner_loop;
                q_base_sz_ = q_base_sz;

                xehe::dpcpp::Context ctx;
                queue_ = ctx.queue();

#ifdef XeHE_DEBUG
                // check if device has the reqd group size extension
                cl::sycl::device device = queue_.get_device();

                if (!device.has_extension("cl_intel_required_subgroup_size")) 
                {
                    throw std::invalid_argument("Device doesn't have cl_intel_required_subgroup_size extension");
                }
#endif

                range_ = q_base_sz_ * n_;
                A_ = cl::sycl::malloc_shared<T>(range_, queue_);
                B_ = cl::sycl::malloc_shared<T>(range_, queue_);
                C_ = cl::sycl::malloc_shared<T>(range_, queue_);

                ref_R_ = cl::sycl::malloc_shared<T>(2*range_, queue_);
                inline_R_ = cl::sycl::malloc_shared<T>(2*range_, queue_);;

                std::vector<T> xe_modulus;

                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<T> dis;

                for (int i = 0; i < q_base_sz_;)
                {
                    auto m = T(dis(gen) & ((sizeof(T) == 4)? 0x7fffffff : 0x7fffffffffffffffUL));
                    if (m != 0)
                    {
                        xe_modulus.push_back(m);
                        ++i;
                    }
                }

                modulo_ = cl::sycl::malloc_shared<T>(q_base_sz_, queue_);
                inv_mod_ = cl::sycl::malloc_shared<T>(q_base_sz_, queue_);
                inv_mod2_ = cl::sycl::malloc_shared<T>(2 * q_base_sz_, queue_);
                std::memcpy((void*)modulo_, xe_modulus.data(), q_base_sz_ * sizeof(T));

                for (int j = 0; j < q_base_sz_; ++j)
                {
                    ((T*)inv_mod_)[j] = xehe::native::mod_inverse1(xe_modulus[j]);
                    ((T*)inv_mod2_)[2 * j] = xehe::native::mod_inverse2(xe_modulus[j], &((T*)inv_mod2_)[2 * j + 1]);
                    for (int i = 0; i < n_; i++) {
                        A_[i + j * n_] = std::rand() % xe_modulus[j];
                        B_[i + j * n_] = std::rand() % xe_modulus[j];
                        C_[i + j * n_] = std::rand() % xe_modulus[j];
                    }
                }

                //fill inline_R_
                for(int k = 0; k < range_; k++)
                {
                    inline_R_[k] = 0xbeef;
                }
            }

            void reset_vector(T* A, size_t length, T value)
            {
                for(int i = 0; i < range_; i++)
                {
                    A[i] = value;
                }
            }

            //function to check 2 vectors are equal
            void verify_equal_vectors(const T * A, const T* B, size_t length,
                        bool print_first_64_vals = false) 
            {
                int err = 0;
                for (int i = 0; i < length; ++i){
                    if(print_first_64_vals && i < 64){
                        std::cout << "\t\tinlined_val = " << A[i] << std::endl;
                        std::cout << "\t\tref_val = " << B[i] << std::endl;

                    }
                    if (A[i] != B[i]) {
                        // std::cerr << "At index: " << i << " ";
                        // std::cerr << A[i] << " != " << B[i] << "\n";
                        // return false;
                        err++;
                    }
                    
                }
                std::cout << "\t" << err << " errors out of " << length << "\n\n";
            }

            void verify_inline_add_uint(void)
            {
                //reset both result vectors with different values
                reset_vector(ref_R_, range_, 0);
                reset_vector(inline_R_, range_, T(0xbeef));

                //get results using the native xehe op
                for (int i = 0; i < range_; i++)
                {
                    xehe::native::add_uint<T>(A_[i],B_[i],&(ref_R_[i]));
                }

                // get results from inlined kernel
                queue_.submit([&](sycl::handler& h) 
                {
                    auto A = A_;
                    auto B = B_;
                    auto inline_R = inline_R_;

                    // sycl::stream out(1024, 256, h);

                    h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
#ifdef __SYCL_DEVICE_ONLY__
                        asm(ADD_STR(_SIMD_WIDTH_)
                            : "=rw"(inline_R[ind])
                            : "rw"(A[ind]), "rw"(B[ind])
                        );
#else
                        inline_R[ind] = A[ind] * B[ind];
#endif
                        // out << "inline_R = " << inline_R[ind] << cl::sycl::endl;
                    } 
                    );
                });

                queue_.wait();

                std::cout << "add_uint_" << sizeof(T)*8 << ":" << std::endl;
                verify_equal_vectors(inline_R_, ref_R_, range_, false);
            }


            void verify_inline_addmod(void)
            {
                //reset both result vectors with different values
                reset_vector(ref_R_, range_, 0);
                reset_vector(inline_R_, range_, T(0xbeef));

                //get results using native xehe op
                for (int i = 0; i < range_; i++)
                {
                    ref_R_[i] = xehe::native::add_mod<T>(A_[i], B_[i], modulo_[0]);
                }

                //get results from inlined kernel
                queue_.submit([&](sycl::handler& h) 
                {
                    auto A = A_;
                    auto B = B_;
                    auto MOD = modulo_;
                    auto inline_R = inline_R_;

                    sycl::stream out(1024, 256, h);

                    h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
#if defined(__SYCL_DEVICE_ONLY__) && defined(XeHE_INLINE_ASM)
                        if constexpr (sizeof(T) == 4)
                        {
                            asm(ADDMOD_32_STR(_SIMD_WIDTH_)
                            : "=rw"(inline_R[ind])
                            : "rw"(A[ind]), "rw"(B[ind]), "rw"(MOD[0]));
                        }
                        else if constexpr (sizeof(T) == 8)
                        {
                            asm(ADDMOD_64_STR(_SIMD_WIDTH_)
                            : "+rw"(inline_R[ind])
                            : "rw"(A[ind]), "rw"(B[ind]), "rw"(MOD[0]));
                        }
#else
                        inline_R[ind] = xehe::native::add_mod<T>(A[ind], B[ind], MOD[0]);
#endif
                        // out << "A = " << A[ind] << "\n" 
                        //     << "B = " << B[ind] << "\n"
                        //     << "inline_R = " << inline_R[ind] << sycl::endl;
                    }
                    );
                });

                queue_.wait();

                std::cout << "add_mod_" << sizeof(T)*8 << ":" << std::endl;
                verify_equal_vectors(inline_R_, ref_R_, range_, false);
            }

            void verify_inline_addmod_opt(void)
            {
                //reset both result vectors with different values
                reset_vector(ref_R_, range_, 0);
                reset_vector(inline_R_, range_, T(0xbeef));

                //get results using native xehe op
                for (int i = 0; i < range_; i++)
                {
                    ref_R_[i] = xehe::native::add_mod<T>(A_[i], B_[i], modulo_[0]);
                }
                
                //get results from inlined kernel
                queue_.submit([&](sycl::handler& h) 
                {
                    auto A = A_;
                    auto B = B_;
                    auto MOD = modulo_;
                    auto inline_R = inline_R_;

                    // sycl::stream out(1024, 256, h);

                    h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        inline_R[ind] = xehe::native::add_mod<T>(A[ind], B[ind], MOD[0]);
                    }
                    );
                });

                queue_.wait();

                std::cout << "add_mod_opt_" << sizeof(T)*8 << ":" << std::endl;
                verify_equal_vectors(inline_R_, ref_R_, range_, false);

            }

            void verify_inline_mul_uint_low(void)
            {
                //reset both result vectors with different values
                reset_vector(ref_R_, range_, 0);
                reset_vector(inline_R_, range_, T(0xbeef));

                //get results using native xehe op
                for (int i = 0; i < range_; i++)
                {
                    ref_R_[i] = xehe::native::mul_uint_low<T>(A_[i], B_[i]);
                }
                
                //get results from inlined kernel
                queue_.submit([&](sycl::handler& h) 
                {
                    auto A = A_;
                    auto B = B_;
                    auto inline_R = inline_R_;

                    // sycl::stream out(1024, 256, h);

                    h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        inline_R[ind] = xehe::native::mul_uint_low<T>(A[ind], B[ind]);
                    }
                    );
                });

                queue_.wait();

                std::cout << "mul_uint_" << sizeof(T)*8 << "_low :" << std::endl;
                verify_equal_vectors(inline_R_, ref_R_, range_, false);
            }

            void verify_inline_mul_uint(void)
            {
                //reset both result vectors with different values
                reset_vector(ref_R_, 2*range_, 0);
                reset_vector(inline_R_, 2*range_, T(0xbeef));

                //cpu
                for (int i = 0; i < range_; i++)
                {
                    ref_R_[2*i] = xehe::native::mul_uint<T>(A_[i], B_[i], ref_R_+ 2*i + 1);
                }

                // // //get results using the new mul_uint
                // queue_.submit([&](sycl::handler& h) 
                // {
                //     auto A = A_;
                //     auto B = B_;
                //     auto ref_R = ref_R_;

                //     sycl::stream out(1024, 256, h);

                //     h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                //     {
                //         ref_R[2*ind] = xehe::native::mul_uint<T>(A[ind], B[ind], ref_R + 2*ind + 1);
                //     }
                //     );
                // });
                // queue_.wait();

                // //get results using the new mul_uint
                queue_.submit([&](sycl::handler& h) 
                {
                    auto A = A_;
                    auto B = B_;
                    auto inline_R = inline_R_;

                    sycl::stream out(1024, 256, h);

                    h.parallel_for<>(sycl::range<1>(range_), [=] (cl::sycl::id<1> ind) [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        inline_R[2*ind] = xehe::native::mul_uint<T>(A[ind], B[ind], inline_R + 2*ind + 1);
                    }
                    );
                });
                queue_.wait();

                // for (int i = 0; i < 5 ; i++)
                // {
                //     // if(inline_R_[i]==ref_R_[i])
                //     // {
                //         std::cout << "at index " << i << std::endl;
                //         std::cout << "A = " << A_[i] << std::endl;
                //         std::cout << "B = " << B_[i] << std::endl;
                //         std::cout << "inline = " << inline_R_[i*2] << std::endl;
                //         std::cout << "ref = " << ref_R_[i*2] << std::endl;
                //         // break;
                //     // }
                // }

                std::cout << "mul_uint_" << sizeof(T)*8 << "_opt :" << std::endl;
                verify_equal_vectors(inline_R_, ref_R_, 2*range_, false);
                // std::cout << "A[60] = " << T(A_[60]) << std::endl;
                // std::cout << "B[60] = " << T(B_[60]) << std::endl;
                // std::cout << "ref[60] = " << T(ref_R_[60]) << std::endl;
                // std::cout << "inline[60] = " << T(inline_R_[60]) << std::endl;

            }

            void kernel_tester(void)
            {
                auto grid_range = sycl::range<3>(10, 3 ,(1 << (2 + 1)));
                queue_.submit([&](cl::sycl::handler& h)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                        {
                            auto A = A_;
                            auto B = B_;
                            auto C = C_;
                            auto modulus = modulo_;
                            h.parallel_for({ grid_range }, RnsDwtGap<T, 0>(20, log_n_, 1, 2, 100, A, modulus,
                                                                                    B, C));
                }).wait();

            }

            protected:
       
            cl::sycl::queue queue_;
            size_t n_;
            int log_n_;
            int q_base_sz_;
            uint64_t range_;
            const T* modulo_;
            const T* inv_mod_;
            const T* inv_mod2_;
            T* A_;
            T* B_;
            T* C_;
            T* ref_R_;
            T* inline_R_;
            int outer_loop_;
            int inner_loop_;
        };
    } // namespace util
} // namespace xehe

#endif //#ifdef BUILD_WITH_IGPU


#endif //#ifndef _PERF_GPU_HPP_