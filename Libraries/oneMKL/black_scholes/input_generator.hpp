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

#include "input_generator_buffer_rng_device.hpp"
#include "input_generator_buffer_rng.hpp"
#include "input_generator_usm_rng_device.hpp"
#include "input_generator_usm_rng.hpp"

namespace input_generator {

using std::int64_t;
using std::uint64_t;
using std::size_t;



template <typename T>
void run(
        int var,
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

    if (var < 0 || var > 3) { throw std::runtime_error("invalid variant"); }

    const char * variants[] = { "USM mkl::rng", "USM mkl::rng_device", "Buffer mkl::rng", "Buffer mkl::rng_device" };

    std::cerr << "running " << variants[var] << std::endl;

    switch (var) {
        case 0:  usm::rng::run(nopt, s0_low, s0_high, s0, x_low, x_high, x, t_low, t_high, t, seed, q); break;
        case 1:  usm::rng_device::run(nopt, s0_low, s0_high, s0, x_low, x_high, x, t_low, t_high, t, seed, q); break;
        case 2:  buffer::rng::run(nopt, s0_low, s0_high, s0, x_low, x_high, x, t_low, t_high, t, seed, q); break;
        case 3:  buffer::rng_device::run(nopt, s0_low, s0_high, s0, x_low, x_high, x, t_low, t_high, t, seed, q); break;
    }
}



} // namespace input_generator

