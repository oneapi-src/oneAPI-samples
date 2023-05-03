//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cstdio>
#include <algorithm>
#include <cmath>

#if _OPENMP
#include <omp.h>
#endif

#include <mkl.h>
#include "binomial.hpp"

#ifndef BINOMIAL_USE_TBB_ALLOCATOR
#define BINOMIAL_USE_TBB_ALLOCATOR 0
#endif

#ifndef BINOMIAL_USE_OMP_SIMD
#define BINOMIAL_USE_OMP_SIMD 1
#endif

#if BINOMIAL_USE_TBB_ALLOCATOR
#include <tbb/cache_aligned_allocator.h>
#define MALLOC(x) (DATA_TYPE*)scalable_malloc(x*sizeof(DATA_TYPE));
#else
#define MALLOC(x) (DATA_TYPE*)malloc(x*sizeof(DATA_TYPE));
#endif

void Binomial::body() {
#pragma omp parallel for
    for (int opt = 0; opt < opt_n; opt++) {
        DATA_TYPE Call[num_steps + 1];
        const DATA_TYPE       Sx = h_stock_price[opt];
        const DATA_TYPE       Xx = h_option_strike[opt];
        const DATA_TYPE       Tx = h_option_years[opt];
        const DATA_TYPE      dt = Tx / static_cast<DATA_TYPE>(num_steps);
        const DATA_TYPE     vDt = volatility * std::sqrt(dt);
        const DATA_TYPE     rDt = risk_free * dt;
        const DATA_TYPE      If = std::exp(rDt);
        const DATA_TYPE      Df = std::exp(-rDt);
        const DATA_TYPE       u = std::exp(vDt);
        const DATA_TYPE       d = std::exp(-vDt);
        const DATA_TYPE      pu = (If - d) / (u - d);
        const DATA_TYPE      pd = 1.0f - pu;
        const DATA_TYPE  puByDf = pu * Df;
        const DATA_TYPE  pdByDf = pd * Df;
        const DATA_TYPE mul_c = vDt * static_cast<DATA_TYPE>(2.0);
        DATA_TYPE id = vDt * static_cast<DATA_TYPE>(-num_steps);
#if BINOMIAL_USE_OMP_SIMD
#pragma omp simd
#endif
#pragma unroll(8)
        for (int i = 0; i <= num_steps; i++) {
            DATA_TYPE d = Sx * std::exp(id) - Xx;
            Call[i] = (d > 0)?d:0;
            id += mul_c;
        }
        // Start at the final tree time step nodes(leaves) and walk backwards
        // to calculate the call option price.
        for (int i = num_steps; i > 0; i--)
#if BINOMIAL_USE_OMP_SIMD
#pragma omp simd
#endif
#pragma unroll(8)
            for (int j = 0; j <= i - 1; j++)
                Call[j] = puByDf * Call[j + 1] + pdByDf * Call[j];
        h_call_result[opt] = Call[0];
    }
}

void Binomial::run()
{
#if _OPENMP
    kmp_set_defaults("KMP_AFFINITY=scatter,granularity=thread");
#endif

#pragma omp parallel
    {
    }

    std::printf("%s Precision Binomial Option Pricing, version %d.%d\n", sizeof(DATA_TYPE) > 4 ? "Double" : "Single", MAJOR, MINOR);
    std::printf("Compiler Version: %s, LLVM %d.%d based.\n", __VERSION__, __clang_major__, __clang_minor__);
    std::printf("Build Time      : %s %s\n", __DATE__, __TIME__);
    std::printf("Input Dataset   : %d\n", opt_n);
#if _OPENMP
    std::printf("Using %d OpenMP thread(s)%s.\n", omp_get_max_threads(), BINOMIAL_USE_OMP_SIMD ? " and SIMD Vectorization" : "");
#endif
    std::printf("Pricing %d Options with time step of %d.\n", opt_n, num_steps); fflush(stdout);

    timer t{};
    t.start();
    body();
    t.stop();

    std::printf("Completed in %10.5f seconds. Options per second: %10.5f\n", t.duration(), static_cast<double>(opt_n) / (t.duration()));
    std::printf("Time Elapsed =  %10.5f seconds\n", t.duration()); fflush(stdout);
}

Binomial::Binomial()
{
    h_call_result = MALLOC(opt_n);
    h_stock_price = MALLOC(opt_n);
    h_option_strike = MALLOC(opt_n);
    h_option_years = MALLOC(opt_n);

    std::fill_n(h_call_result, opt_n, 0.0);
    constexpr int rand_seed = 777;
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_PHILOX4X32X10, rand_seed);
    if (sizeof(DATA_TYPE) == sizeof(double)) {
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, opt_n, reinterpret_cast<double*>(h_stock_price), static_cast<double>(5.0), static_cast<double>(50.0));
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, opt_n, reinterpret_cast<double*>(h_option_strike), static_cast<double>(10.0), static_cast<double>(25.0));
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, opt_n, reinterpret_cast<double*>(h_option_years), static_cast<double>(1.0), static_cast<double>(5.0));
    } else {
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, opt_n, reinterpret_cast<float*>(h_stock_price), static_cast<float>(5.0), static_cast<float>(50.0));
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, opt_n, reinterpret_cast<float*>(h_option_strike), static_cast<float>(10.0), static_cast<float>(25.0));
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, opt_n, reinterpret_cast<float*>(h_option_years), static_cast<float>(1.0), static_cast<float>(5.0));
    }
    vslDeleteStream(&stream);
}

Binomial::~Binomial() {}
