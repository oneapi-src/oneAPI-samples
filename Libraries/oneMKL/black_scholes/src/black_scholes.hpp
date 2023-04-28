//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef __BLACK_SCHOLES_HPP__
#define __BLACK_SCHOLES_HPP__

#include <vector>
#include <chrono>

/******* VERSION *******/
#define MAJOR 1
#define MINOR 6
/******* VERSION *******/

#ifndef _PRECISION
#define _PRECISION double
#endif

#ifndef VERBOSE
#define VERBOSE 1
#endif

constexpr float VOLATILITY = 0.30f;
constexpr float RISKFREE = 0.02f;

constexpr size_t OPT_N =
#if SMALL_OPT_N
    480;
#else
    8 * 1024 * 1024;
#endif

#ifndef ITER_N
#define ITER_N 512
#endif

#ifndef __clang_major__
#define __clang_major__ 0
#endif
#ifndef __clang_minor__
#define __clang_minor__ 0
#endif
#ifndef __VERSION__
#define __VERSION__ __clang_major__
#endif

class BlackScholes {
public:
    BlackScholes();
    ~BlackScholes();
    void run();
    void check();
    inline bool isDP() { return sizeof(_PRECISION) > 4; }

private:
    _PRECISION* h_CallResult;
    _PRECISION* h_PutResult;
    _PRECISION* h_StockPrice;
    _PRECISION* h_OptionStrike;
    _PRECISION* h_OptionYears;
    void body();
};

// Black-Scholes Reference Implementation
void BlackScholesRefImpl(
    double& callResult,
    double Sf, //Stock price
    double Xf, //Option strike
    double Tf, //Option years
    double Rf, //Riskless rate
    double Vf  //Volatility rate
)
{
    // BSM Formula: https://www.nobelprize.org/prizes/economic-sciences/1997/press-release/
    double S = Sf, L = Xf, t = Tf, r = Rf, sigma = Vf;
    double N_d1 = 1. / 2. + 1. / 2. * std::erf(((std::log(S / L) + (r + 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t))) / std::sqrt(2.));
    double N_d2 = 1. / 2. + 1. / 2. * std::erf(((std::log(S / L) + (r - 0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t))) / std::sqrt(2.));
    callResult = (S * N_d1 - L * std::exp(-r * t) * N_d2);
}

void BlackScholes::check()
{
    if (VERBOSE) {
        std::printf("Creating the reference result...\n");
        std::vector<double> h_CallResultCPU(OPT_N);

        for (int opt = 0; opt < OPT_N; opt++)
            BlackScholesRefImpl(h_CallResultCPU[opt], h_StockPrice[opt], h_OptionStrike[opt], h_OptionYears[opt], RISKFREE, VOLATILITY);

        double sum_delta = 0.0,
            sum_ref = 0.0,
            max_delta = 0.0,
            sumReserve = 0.0,
            errorVal = 0.0;

        for (auto i = 0; i < OPT_N; i++) {
            auto ref = h_CallResultCPU[i];
            auto delta = std::fabs(h_CallResultCPU[i] - h_CallResult[i]);
            if (delta > max_delta) {
                max_delta = delta;
            }
            sum_delta += delta;
            sum_ref += std::fabs(ref);
        }
        sumReserve /= OPT_N;
        if (sum_ref > 1E-5)
            std::printf("L1 norm: %E\n", errorVal = sum_delta / sum_ref);
        else
            std::printf("Avg. diff: %E\n", errorVal = sum_delta / OPT_N);
        std::printf((errorVal < 5e-4) ? "TEST PASSED\n" : "TEST FAILED\n");

    }
}

class timer {
public:
    timer() { start(); }
    void start() { t1_ = std::chrono::steady_clock::now(); }
    void stop() { t2_ = std::chrono::steady_clock::now(); }
    auto duration() { return std::chrono::duration<double>(t2_ - t1_).count(); }
private:
    std::chrono::steady_clock::time_point t1_, t2_;
};

#endif // __BLACK_SCHOLES_HPP__
