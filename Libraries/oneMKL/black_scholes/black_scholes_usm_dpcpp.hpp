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
namespace usm {
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
// allocate memory on device
    T * dev_s0 = sycl::malloc_device<T>(nopt, q);
    T * dev_x  = sycl::malloc_device<T>(nopt, q);
    T * dev_t  = sycl::malloc_device<T>(nopt, q);
    T * dev_opt_call = sycl::malloc_device<T>(nopt, q);
    T * dev_opt_put  = sycl::malloc_device<T>(nopt, q);

// check allocation
    if (nullptr ==  dev_s0
        || nullptr ==  dev_x
        || nullptr ==  dev_t
        || nullptr == dev_opt_call
        || nullptr == dev_opt_put) {
        std::cerr << "failed to allocate USM memory" << std::endl;
        throw std::runtime_error("failed to allocate USM");
    }

// copy inputs to device
    q.memcpy(dev_s0, s0, nopt * sizeof(T));
    q.memcpy(dev_x, x, nopt * sizeof(T));
    q.memcpy(dev_t, t, nopt * sizeof(T));

// calculate
    q.submit(
    [&](sycl::handler & cgh) {
        sycl::range<1> range { static_cast<size_t>(nopt) };

        constexpr T one_above_four { 0.25 };
        constexpr T one_above_two  { 0.5 };
        constexpr T two            { 2.0 };
        
        T sig_sig_two = volatility * volatility * two;
        T mr = -risk_free;

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);
            T a, b, c, y, z, e;
            T d1, d2, w1, w2;

            a = sycl::log(dev_s0[i] / dev_x[i]);
            b = dev_t[i] * mr;
            z = dev_t[i] * sig_sig_two;

            c = one_above_four * z;
            e = sycl::exp(b);
            y = sycl::rsqrt(z);

            w1 = (a - b + c) * y;
            w2 = (a - b - c) * y;

            d1 = sycl::erf(w1);
            d2 = sycl::erf(w2);
            d1 = one_above_two + one_above_two * d1;
            d2 = one_above_two + one_above_two * d2;

            T call = dev_s0[i] * d1 - dev_x[i] * e * d2;    
            dev_opt_call[i] = call;
            dev_opt_put[i]  = call - dev_s0[i] + dev_x[i] * e;
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit

    q.wait_and_throw();

// copy back
    q.memcpy(opt_call, dev_opt_call, nopt * sizeof(T));
    q.memcpy(opt_put, dev_opt_put,   nopt * sizeof(T));
    q.wait_and_throw();

    sycl::free(dev_opt_put, q);
    sycl::free(dev_opt_call, q);
    sycl::free(dev_t, q);
    sycl::free(dev_x, q);
    sycl::free(dev_s0, q);
} 

} // namespace impl

using impl::run;

} // namespace dpcpp
} // namespace usm
} // namespace black_scholes

