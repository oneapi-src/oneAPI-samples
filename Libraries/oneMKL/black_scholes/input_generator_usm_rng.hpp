//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*******************************************************************************
!  Content:
!      Black-Scholes formula Intel(r) Math Kernel Library (Intel(r) MKL) VML based Example
!******************************************************************************/

#pragma once

namespace input_generator {
namespace usm {
namespace rng {
namespace impl {

using std::int64_t;
using std::uint64_t;
using std::size_t;


template <typename T>
struct device_ptr {
    sycl::queue q_;
    T * ptr_;

    device_ptr(int64_t s, sycl::queue & q) { 
        ptr_ = sycl::malloc_device<T>(s, q); 
        if (nullptr == ptr_) { throw std::runtime_error("bad device alloc"); }
        q_ = q;
    }

    T * operator()()  { return ptr_; }
    ~device_ptr() { sycl::free(ptr_, q_); }
};

template <typename T>
void run(
        int64_t nopt,
        T s0_low,
        T s0_high,
        T * s0,
        T x_low,
        T x_high,
        T * x,
        T t_low,
        T t_high,
        T * t,
        uint64_t seed,
        sycl::queue & q
    ) {

    mkl::rng::philox4x32x10 engine(q, seed);
    mkl::rng::uniform distr_s0 ( s0_low, s0_high );
    mkl::rng::uniform distr_x  ( x_low, x_high );
    mkl::rng::uniform distr_t  ( t_low, t_high );

    device_ptr<T> dptr_s0 ( nopt, q );
    device_ptr<T> dptr_x ( nopt, q );
    device_ptr<T> dptr_t ( nopt, q );

    T * dev_s0 = dptr_s0();
    T * dev_x  = dptr_x();
    T * dev_t  = dptr_t();
    
    mkl::rng::generate(distr_s0, engine, nopt, dev_s0);
    mkl::rng::generate(distr_x, engine, nopt, dev_x);
    mkl::rng::generate(distr_t, engine, nopt, dev_t);

    q.wait_and_throw();


    q.memcpy(s0, dev_s0, nopt * sizeof(T));
    q.memcpy(x, dev_x, nopt * sizeof(T));
    q.memcpy(t, dev_t, nopt * sizeof(T));
    q.wait_and_throw();
}

} // namespace impl

using impl::run;

} // namespace rng
} // namespace usm
} // namespace input_generator

