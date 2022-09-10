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
namespace rng_device {

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

    sycl::buffer<T, 1> buf_s0 ( s0, nopt );
    sycl::buffer<T, 1> buf_x ( x, nopt );
    sycl::buffer<T, 1> buf_t ( t, nopt );


    q.submit(
    [&](sycl::handler & cgh) {
        sycl::range<1> range(nopt / 4);

        auto acc_s0 = buf_s0.template get_access<sycl::access::mode::write>(cgh);
        auto acc_x = buf_x.template get_access<sycl::access::mode::write>(cgh);
        auto acc_t = buf_t.template get_access<sycl::access::mode::write>(cgh);


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

            rn1.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(acc_s0.get_pointer() + off));
            rn2.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(acc_x.get_pointer()  + off));
            rn3.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(acc_t.get_pointer()  + off));
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
            auto acc_s0 = buf_s0.template get_access<sycl::access::mode::write>(cgh);
            auto acc_x = buf_x.template get_access<sycl::access::mode::write>(cgh);
            auto acc_t = buf_t.template get_access<sycl::access::mode::write>(cgh);

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
                    acc_s0[k] = rn1[i];
                    acc_x[k] = rn2[i];
                    acc_t[k] = rn3[i];
                }
            } // [=]
            ); // single_task
        } // [&]
        ); // submit
    } // if (rem > 0)
}



} // namespace rng_device
} // namespace buffer
} // namespace input_generator

