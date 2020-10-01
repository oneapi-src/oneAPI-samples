//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*******************************************************************************
!  Content:
!      Black-Scholes formula MKL VML based Example
!******************************************************************************/

#pragma once

namespace black_scholes {
namespace buffer {
namespace dpcpp {
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
        sycl::queue & q
    ) {

    // buffers created from inputs
    sycl::buffer<T, 1> buf_s0(s0, s0 + nopt);
    sycl::buffer<T, 1> buf_x(x, x + nopt);
    sycl::buffer<T, 1> buf_t(t, t + nopt);

    // buffers created for outputs (will be copied back to host)
    sycl::buffer<T, 1> buf_opt_call(opt_call, nopt);
    sycl::buffer<T, 1> buf_opt_put(opt_put, nopt);

    q.submit(
    [&](sycl::handler & cgh) {
        auto acc_s0 = buf_s0.template get_access<sycl::access::mode::read>(cgh);
        auto acc_x  = buf_x.template get_access<sycl::access::mode::read>(cgh);
        auto acc_t  = buf_t.template get_access<sycl::access::mode::read>(cgh);

        auto acc_opt_call = buf_opt_call.template get_access<sycl::access::mode::write>(cgh);
        auto acc_opt_put  = buf_opt_put.template get_access<sycl::access::mode::write>(cgh);

        sycl::range<1> range { static_cast<size_t>(nopt) };

        constexpr T one_over_four { 0.25 };
        constexpr T one_over_two  { 0.5 };
        constexpr T two            { 2.0 };

        T sig_sig_two = volatility * volatility * two;
        T mr = -risk_free;

        cgh.parallel_for(range,
        [=](sycl::id<1> id) {
            size_t i = id.get(0);
            T a, b, c, y, z, e;
            T d1, d2, w1, w2;

            a = sycl::log(acc_s0[i] / acc_x[i]);
            b = acc_t[i] * mr;
            z = acc_t[i] * sig_sig_two;

            c = one_over_four * z;
            e = sycl::exp(b);
            y = sycl::rsqrt(z);

            w1 = (a - b + c) * y;
            w2 = (a - b - c) * y;

            d1 = sycl::erf(w1);
            d2 = sycl::erf(w2);
            d1 = one_over_two + one_over_two * d1;
            d2 = one_over_two + one_over_two * d2;

            T call = acc_s0[i] * d1 - acc_x[i] * e * d2;    
            acc_opt_call[i] = call;
            acc_opt_put[i]  = call - acc_s0[i] + acc_x[i] * e;
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit

    // work will be done when result buffer is destructed
} // run 

} // namespace impl



using impl::run;

} // namespace dpcpp
} // namespace buffer
} // namespace black_scholes

