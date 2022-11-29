//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

#ifndef DataType
#define DataType double
#endif

#ifndef ITEMS_PER_WORK_ITEM
#define ITEMS_PER_WORK_ITEM 4
#endif

#ifndef VEC_SIZE
#define VEC_SIZE 8
#endif

//Should be > 1
constexpr int num_options = 384000;
//Should be > 16
constexpr int path_length = 262144;
//Test iterations
constexpr int num_iterations = 5;

constexpr DataType risk_free  = 0.06f;
constexpr DataType volatility = 0.10f;

constexpr DataType RLog2E = -risk_free * M_LOG2E;
constexpr DataType MuLog2E = M_LOG2E * (risk_free - 0.5 * volatility * volatility);
constexpr DataType VLog2E = M_LOG2E * volatility;

template<typename MonteCarlo_vector>
void check(const MonteCarlo_vector& h_CallResult, const MonteCarlo_vector& h_CallConfidence,
const MonteCarlo_vector& h_StockPrice, const MonteCarlo_vector& h_OptionStrike, const MonteCarlo_vector& h_OptionYears)
{
    std::vector<DataType> h_CallResultRef(num_options);

    auto BlackScholesRefImpl = [](
        DataType Sf, //Stock price
        DataType Xf, //Option strike
        DataType Tf //Option years
        ) {
            // BSM Formula: https://www.nobelprize.org/prizes/economic-sciences/1997/press-release/
            // N(d)=1/2 + 1/2*ERF(d/sqrt(2)), https://software.intel.com/en-us/node/531898
            DataType S = Sf, L = Xf, t = Tf, r = risk_free, sigma = volatility;
            DataType N_d1 = 1. / 2. + 1. / 2. * std::erf(((std::log(S / L) + (r + 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t))) / std::sqrt(2.));
            DataType N_d2 = 1. / 2. + 1. / 2. * std::erf(((std::log(S / L) + (r - 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t))) / std::sqrt(2.));
            return S * N_d1 - L * std::exp(-r * t) * N_d2;
    };

    for (int opt = 0; opt < num_options; opt++)
    {
        h_CallResultRef[opt] = BlackScholesRefImpl(h_StockPrice[opt], h_OptionStrike[opt], h_OptionYears[opt]);
    }

    std::cout << "Running quality test..." << std::endl;

    DataType sum_delta = 0.0,
        sum_ref = 0.0,
        max_delta = 0.0,
        sum_reserve = 0.0;

    for (int opt = 0; opt < num_options; opt++)
    {
        DataType ref = h_CallResultRef[opt];
        DataType delta = std::fabs(h_CallResultRef[opt] - h_CallResult[opt]);
        if (delta > max_delta)
        {
            max_delta = delta;
        }
        sum_delta += delta;
        sum_ref += std::fabs(ref);
        if (delta > 1e-6)
        {
            sum_reserve += h_CallConfidence[opt] / delta;
        }
        max_delta = std::max(delta, max_delta);

    }
    sum_reserve /= static_cast<double>(num_options);
    DataType L1_norm = sum_delta / sum_ref;

    std::cout << "L1_Norm          = "<< L1_norm << std::endl;
    std::cout << "Average RESERVE  = "<< sum_reserve << std::endl;
    std::cout << "Max Error        = "<< max_delta << std::endl;
    std::cout << (sum_reserve > 1.0f ? "TEST PASSED!" : "TEST FAILED!") << std::endl;
}
