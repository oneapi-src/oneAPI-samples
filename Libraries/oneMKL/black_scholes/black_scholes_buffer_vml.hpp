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

namespace black_scholes {
namespace buffer {
namespace vml {
namespace impl {

using std::int64_t;
using std::uint64_t;
using std::size_t;

template <typename T>
void run(
        int64_t nopt,
        T risk_free,
        T volatility,
        T * s0,
        T * x,
        T * t,
        T * opt_call,
        T * opt_put,
        mkl::vm::mode vml_accuracy,
        sycl::queue & q
    ) {

    // buffers created from inputs
    sycl::buffer<T, 1> buf_s0(s0, s0 + nopt);
    sycl::buffer<T, 1> buf_x(x, x + nopt);
    sycl::buffer<T, 1> buf_t(t, t + nopt);

    // buffers created for outputs (will be copied back to host)
    sycl::buffer<T, 1> buf_opt_call(opt_call, nopt);
    sycl::buffer<T, 1> buf_opt_put(opt_put, nopt);

    sycl::buffer<T, 1> buf_a(nopt);
    sycl::buffer<T, 1> buf_b(nopt);
    sycl::buffer<T, 1> buf_c(nopt);
    sycl::buffer<T, 1> buf_e(nopt);
    sycl::buffer<T, 1> buf_y(nopt);
    sycl::buffer<T, 1> buf_z(nopt);
    sycl::buffer<T, 1> buf_d1(nopt);
    sycl::buffer<T, 1> buf_d2(nopt);
    sycl::buffer<T, 1> buf_w1(nopt);
    sycl::buffer<T, 1> buf_w2(nopt);

    mkl::vm::set_mode(q, vml_accuracy);

    mkl::vm::div(q, nopt, buf_s0, buf_x, buf_a);
    mkl::vm::ln(q, nopt, buf_a, buf_a);

    q.submit(
    [&](sycl::handler & cgh) {
        auto acc_t = buf_t.template get_access<sycl::access::mode::read>(cgh);

        auto acc_a = buf_a.template get_access<sycl::access::mode::read_write>(cgh);

        auto acc_b = buf_b.template get_access<sycl::access::mode::write>(cgh);
        auto acc_c = buf_c.template get_access<sycl::access::mode::write>(cgh);
        auto acc_z = buf_z.template get_access<sycl::access::mode::write>(cgh);

        sycl::range<1> range { static_cast<size_t>(nopt) };


        constexpr T one_over_four { 0.25 };
        constexpr T two            { 2.0 };
        
        T sig_sig_two = volatility * volatility * two;
        T mr = -risk_free;

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);

            acc_b[i] = acc_t[i] * mr;
            acc_a[i] = acc_a[i] - acc_b[i];
            T z = acc_t[i] * sig_sig_two;
            acc_z[i] = z;
            acc_c[i] = one_over_four * z;
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit
 
    mkl::vm::invsqrt(q, nopt, buf_z, buf_y);
    mkl::vm::exp(q, nopt, buf_b, buf_e);

    q.submit(
    [&](sycl::handler & cgh) {
        auto acc_a = buf_a.template get_access<sycl::access::mode::read>(cgh);
        auto acc_c = buf_c.template get_access<sycl::access::mode::read>(cgh);
        auto acc_y = buf_y.template get_access<sycl::access::mode::read>(cgh);

        auto acc_w1 = buf_w1.template get_access<sycl::access::mode::write>(cgh);
        auto acc_w2 = buf_w2.template get_access<sycl::access::mode::write>(cgh);

        sycl::range<1> range { static_cast<size_t>(nopt) };

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);
            T ai = acc_a[i];
            T ci = acc_c[i];

            acc_w1[i] = ( ai + ci ) * acc_y[i];
            acc_w2[i] = ( ai - ci ) * acc_y[i];
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit
 

    mkl::vm::erf(q, nopt, buf_w1, buf_d1);
    mkl::vm::erf(q, nopt, buf_w2, buf_d2);

    q.submit(
    [&](sycl::handler & cgh) {
        auto acc_d1 = buf_d1.template get_access<sycl::access::mode::read>(cgh);
        auto acc_d2 = buf_d2.template get_access<sycl::access::mode::read>(cgh);
        auto acc_e  = buf_e.template get_access<sycl::access::mode::read>(cgh);
        auto acc_s0 = buf_s0.template get_access<sycl::access::mode::read>(cgh);
        auto acc_x  = buf_x.template get_access<sycl::access::mode::read>(cgh);

        auto acc_opt_call = buf_opt_call.template get_access<sycl::access::mode::write>(cgh);
        auto acc_opt_put  = buf_opt_put.template get_access<sycl::access::mode::write>(cgh);

        sycl::range<1> range { static_cast<size_t>(nopt) };
        constexpr T one_over_two  { 0.5 };

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);
            T d1 = one_over_two + one_over_two * acc_d1[i];
            T d2 = one_over_two + one_over_two * acc_d2[i];
            T call =  acc_s0[i] * d1 - acc_x[i] * acc_e[i] * d2;
            acc_opt_call[i] = call; 
            acc_opt_put[i]  = call - acc_s0[i] + acc_x[i] * acc_e[i];
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit

    // work will be done when result buffer is destructed
} // run 

} // namespace impl

using impl::run;

} // namespace vml 
} // namespace buffer
} // namespace black_scholes
