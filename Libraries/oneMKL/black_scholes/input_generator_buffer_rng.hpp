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
namespace buffer {
namespace rng {
namespace impl {

using std::int64_t;
using std::uint64_t;
using std::size_t;



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


    sycl::buffer<T, 1> buf_s0 ( s0, nopt );
    sycl::buffer<T, 1> buf_x ( x, nopt );
    sycl::buffer<T, 1> buf_t ( t, nopt );

    mkl::rng::generate(distr_s0, engine, nopt, buf_s0);
    mkl::rng::generate(distr_x, engine, nopt, buf_x);
    mkl::rng::generate(distr_t, engine, nopt, buf_t);
}

} //namespace impl

using impl::run;

} // namespace rng
} // namespace buffer
} // namespace input_generator

