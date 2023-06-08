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

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

#ifndef VERBOSE
#define VERBOSE 1
#endif

constexpr float volatility = 0.30f;
constexpr float risk_free = 0.02f;

constexpr size_t opt_n =
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

private:
    DATA_TYPE* h_call_result;
    DATA_TYPE* h_put_result;
    DATA_TYPE* h_stock_price;
    DATA_TYPE* h_option_strike;
    DATA_TYPE* h_option_years;
    void body();
};

// Black-Scholes Reference Implementation
void BlackScholesRefImpl(
    double& call_result,
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
    call_result = (S * N_d1 - L * std::exp(-r * t) * N_d2);
}

void BlackScholes::check()
{
    if (VERBOSE) {
        std::printf("Creating the reference result...\n");
        std::vector<double> h_CallResultCPU(opt_n);

        for (size_t opt = 0; opt < opt_n; opt++)
            BlackScholesRefImpl(h_CallResultCPU[opt], h_stock_price[opt], h_option_strike[opt], h_option_years[opt], risk_free, volatility);

        double sum_delta = 0.0,
            sum_ref = 0.0,
            max_delta = 0.0,
            errorVal = 0.0;

        for (size_t i = 0; i < opt_n; i++) {
            auto ref = h_CallResultCPU[i];
            auto delta = std::fabs(h_CallResultCPU[i] - h_call_result[i]);
            if (delta > max_delta) {
                max_delta = delta;
            }
            sum_delta += delta;
            sum_ref += std::fabs(ref);
        }
        if (sum_ref > 1E-5)
            std::printf("L1 norm: %E\n", errorVal = sum_delta / sum_ref);
        else
            std::printf("Avg. diff: %E\n", errorVal = sum_delta / opt_n);
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
