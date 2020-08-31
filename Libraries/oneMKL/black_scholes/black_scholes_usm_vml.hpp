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
namespace usm {
namespace vml {
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
        T risk_free,
        T volatility,
        T * s0,
        T * x,
        T * t,
        T * opt_call,
        T * opt_put,
        uint64_t vml_accuracy,
        sycl::queue & q 
    ) {
// allocate memory on device
    device_ptr<T> dptr_s0(nopt, q);
    device_ptr<T> dptr_t(nopt, q);
    device_ptr<T> dptr_x(nopt, q);
    device_ptr<T> dptr_opt_call(nopt, q);
    device_ptr<T> dptr_opt_put(nopt, q);

// temporaries
    device_ptr<T> dptr_a(nopt, q);
    device_ptr<T> dptr_b(nopt, q);
    device_ptr<T> dptr_c(nopt, q);
    device_ptr<T> dptr_e(nopt, q);
    device_ptr<T> dptr_y(nopt, q);
    device_ptr<T> dptr_z(nopt, q);
    device_ptr<T> dptr_d1(nopt, q);
    device_ptr<T> dptr_d2(nopt, q);
    device_ptr<T> dptr_w1(nopt, q);
    device_ptr<T> dptr_w2(nopt, q);

    T * dev_s0 = dptr_s0();
    T * dev_x  = dptr_x();
    T * dev_t  = dptr_t();
    T * dev_opt_call = dptr_opt_call();
    T * dev_opt_put  = dptr_opt_put();

    T * dev_a = dptr_a();
    T * dev_b = dptr_b();
    T * dev_c = dptr_c();
    T * dev_e = dptr_e();
    T * dev_y = dptr_y();
    T * dev_z = dptr_z();
    T * dev_d1 = dptr_d1();
    T * dev_d2 = dptr_d2();
    T * dev_w1 = dptr_w1(); 
    T * dev_w2 = dptr_w2();


// copy inputs to device
    q.memcpy(dev_s0, s0, nopt * sizeof(T));
    q.memcpy(dev_x, x, nopt * sizeof(T));
    q.memcpy(dev_t, t, nopt * sizeof(T));

    mkl::vm::set_mode(q, vml_accuracy);

// calculate
    auto ev1 = mkl::vm::ln(q, nopt, dev_a, dev_a, { mkl::vm::div(q, nopt, dev_s0, dev_x, dev_a)} );
    auto ev2 = q.submit(
    [&](sycl::handler & cgh) {
        cgh.depends_on(ev1);

        sycl::range<1> range { static_cast<size_t>(nopt) };

        constexpr T one_over_four { 0.25 };
        constexpr T two            { 2.0 };
        
        T sig_sig_two = volatility * volatility * two;
        T mr = -risk_free;

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);

            dev_b[i] = dev_t[i] * mr;
            dev_a[i] = dev_a[i] - dev_b[i];
            dev_z[i] = dev_t[i] * sig_sig_two;
            dev_c[i] = one_over_four * dev_z[i];
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit
    
    auto ev3 = mkl::vm::invsqrt(q, nopt, dev_z, dev_y, { ev2 });
    auto ev4 = mkl::vm::exp(q, nopt, dev_b, dev_e, { ev2 });

    auto ev5 = q.submit(
    [&](sycl::handler & cgh) {
        cgh.depends_on(ev3);

        sycl::range<1> range { static_cast<size_t>(nopt) };

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);
            T ai = dev_a[i];
            T ci = dev_c[i];

            dev_w1[i] = ( ai + ci ) * dev_y[i];
            dev_w2[i] = ( ai - ci ) * dev_y[i];
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit
 

    auto ev6 = mkl::vm::erf(q, nopt, dev_w1, dev_d1, { ev5 });
    auto ev7 = mkl::vm::erf(q, nopt, dev_w2, dev_d2, { ev5 });

    auto ev8 = q.submit(
    [&](sycl::handler & cgh) {
        cgh.depends_on({ev4, ev6, ev7});


        sycl::range<1> range { static_cast<size_t>(nopt) };
        constexpr T one_over_two  { 0.5 };

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t i = id.get(0);
            T d1 = one_over_two + one_over_two * dev_d1[i];
            T d2 = one_over_two + one_over_two * dev_d2[i];
            T call =  dev_s0[i] * d1 - dev_x[i] * dev_e[i] * d2;
            dev_opt_call[i] = call; 
            dev_opt_put[i]  = call - dev_s0[i] + dev_x[i] * dev_e[i];
        } // [=]
        ); // parallel_for
    } // [&]
    ); // submit

    q.wait_and_throw();
 
// copy back
    q.memcpy(opt_call, dev_opt_call, nopt * sizeof(T));
    q.memcpy(opt_put, dev_opt_put,   nopt * sizeof(T));
    q.wait_and_throw();
} 
} // namespace impl

using impl::run;

} // namespace vml
} // namespace usm
} // namespace black_scholes

