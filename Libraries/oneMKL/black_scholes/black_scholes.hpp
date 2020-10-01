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

#include "black_scholes_usm_dpcpp.hpp"
#include "black_scholes_usm_vml.hpp"
#include "black_scholes_buffer_dpcpp.hpp"
#include "black_scholes_buffer_vml.hpp"

namespace black_scholes {

using std::int64_t;
using std::uint64_t;
using std::size_t;

template <typename T>
void run(
        int var,
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

    if (var < 0 || var > 3) { throw std::runtime_error("invalid variant"); }

    const char * variants[] = { "USM mkl::vm", "USM dpcpp", "Buffer mkl::vm", "Buffer dpcpp" };

    std::cerr << "running " << variants[var] << std::endl;

    switch (var) {
        case 0:  usm::vml::run(nopt, risk_free, volatility, s0, x, t, opt_call, opt_put, vml_accuracy, q); break;
        case 1:  usm::dpcpp::run(nopt, risk_free, volatility, s0, x, t, opt_call, opt_put, q); break;
        case 2:  buffer::vml::run(nopt, risk_free, volatility, s0, x, t, opt_call, opt_put, vml_accuracy, q); break;
        case 3:  buffer::dpcpp::run(nopt, risk_free, volatility, s0, x, t, opt_call, opt_put, q); break;
    }
}

} // namespace black_scholes

