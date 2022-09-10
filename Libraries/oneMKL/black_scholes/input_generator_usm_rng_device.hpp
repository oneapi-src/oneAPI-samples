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
namespace rng_device {
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

    device_ptr<T> dptr_s0 ( nopt, q );
    device_ptr<T> dptr_x ( nopt, q );
    device_ptr<T> dptr_t ( nopt, q );

    T * dev_s0 = dptr_s0();
    T * dev_x  = dptr_x();
    T * dev_t  = dptr_t();



    q.submit(
    [&](sycl::handler & cgh) {
        sycl::range<1> range(nopt / 4);

        cgh.parallel_for(range, 
        [=](sycl::id<1> id) {
            size_t k = id.get(0);
            size_t off = k * 4;
            sycl::vec<T, 4> rn1;
            sycl::vec<T, 4> rn2;
            sycl::vec<T, 4> rn3;

            mkl::rng::device::uniform distr1(s0_low, s0_high);
            mkl::rng::device::uniform distr2(x_low, x_high);
            mkl::rng::device::uniform distr3(t_low, t_high);


            mkl::rng::device::philox4x32x10<4> engine1(seed, 3 * off);
            mkl::rng::device::philox4x32x10<4> engine2(seed, 3 * off + 1);
            mkl::rng::device::philox4x32x10<4> engine3(seed, 3 * off + 2);


            rn1 = mkl::rng::device::generate(distr1, engine1);
            rn2 = mkl::rng::device::generate(distr2, engine2);
            rn3 = mkl::rng::device::generate(distr3, engine3);

            rn1.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(dev_s0 + off));
            rn2.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(dev_x + off));
            rn3.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(dev_t + off));
        }
        );
    } // [&]
    ); // submit


    /* this is not recommended */
    int64_t rem = (nopt % 4);

    if (rem > 0) {
        // tail
        q.submit(
        [&](sycl::handler & cgh) {
            cgh.single_task( 
            [=]() {
                size_t k = nopt - rem;
                size_t off = k / 4;

                sycl::vec<T, 4> rn1;
                sycl::vec<T, 4> rn2;
                sycl::vec<T, 4> rn3;

                mkl::rng::device::uniform distr1(s0_low, s0_high);
                mkl::rng::device::uniform distr2(x_low, x_high);
                mkl::rng::device::uniform distr3(t_low, t_high);

                mkl::rng::device::philox4x32x10<4> engine1(seed, 3 * off);
                mkl::rng::device::philox4x32x10<4> engine2(seed, 3 * off + 1);
                mkl::rng::device::philox4x32x10<4> engine3(seed, 3 * off + 2);

                rn1 = mkl::rng::device::generate(distr1, engine1);
                rn2 = mkl::rng::device::generate(distr2, engine2);
                rn3 = mkl::rng::device::generate(distr3, engine3);

                for (int i = 0; i < rem; ++i, ++k)  {
                    dev_s0[k] = rn1[i];
                    dev_x[k] = rn2[i];
                    dev_t[k] = rn3[i];
                }
            } // [=]
            ); // single_task
        } // [&]
        ); // submit
    } // if (rem > 0)

    q.wait_and_throw();


    q.memcpy(s0, dev_s0, nopt * sizeof(T));
    q.memcpy(x, dev_x, nopt * sizeof(T));
    q.memcpy(t, dev_t, nopt * sizeof(T));
    q.wait_and_throw();
}


} // namespace impl

using impl::run;

} // rng_device
} // usm
} // namespace input_generator

