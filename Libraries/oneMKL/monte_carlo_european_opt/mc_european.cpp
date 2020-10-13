//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This file contains Monte Carlo simulatuon for European option pricing
*       for DPC++ interface of random number generators.
*
*******************************************************************************/

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_rng_sycl.hpp"
#endif

using namespace oneapi;

// Temporary code for beta08 compatibility. Reduce routine is moved from intel::
// to ONEAPI:: namespace
#if __SYCL_COMPILER_VERSION < 20200902L
using sycl::intel::reduce;
#else
using sycl::ONEAPI::reduce;
#endif

// Default number of options
static const auto num_options_default = 2048;
// Default number of independent samples
static const auto num_samples_default = 262144;

// Initialization value for random number generator
static const auto seed = 7777;

// European options pricing parameters
static const auto initial_stock_price_l = 10.0;
static const auto initial_stock_price_h = 50.0;
static const auto strike_price_l = 10.0;
static const auto strike_price_h = 50.0;
static const auto t_l = 0.2;
static const auto t_h = 2.0;
static const auto default_risk_neutral_rate = 0.05;
static const auto default_volatility = 0.2;

void init_data(std::vector<double>& initial_stock_price, std::vector<double>& strike_price,
               std::vector<double>& t, std::vector<double>& vcall, std::vector<double>& vput);

// Black-Scholes formula to verify Monte Carlo computation
void bs_ref(double risk_neutral_rate, double volatility, std::vector<double>& initial_stock_price,
            std::vector<double>& strike_price, std::vector<double>& t, std::vector<double>& vcall,
            std::vector<double>& vput);

double estimate_error(std::vector<double>& v_put_ref, std::vector<double>& v_put);

template <typename EngineType>
static void mc_kernel(sycl::queue& q, EngineType& engine, size_t num_samples,
                      double risk_neutral_rate, double volatility, double initial_stock_price,
                      double strike_price, double t, double& vcall, double& vput,
                      sycl::buffer<double>& rng_buf) {
    double a, nu;
    double sc_dp, sp_dp;
    double st;

    a = (risk_neutral_rate - volatility * volatility * 0.5) * t;
    nu = volatility * sqrt(t);

    mkl::rng::lognormal<double, mkl::rng::lognormal_method::box_muller2> distr(a, nu);

    mkl::rng::generate(distr, engine, num_samples, rng_buf);

    size_t wg_size =
        std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(), num_samples);
    size_t wg_num = num_samples / wg_size;

    std::vector<double> count_sc(wg_num);
    std::vector<double> count_sp(wg_num);

    {
        sycl::buffer<double, 1> count_sc_buf(count_sc.data(), count_sc.size());
        sycl::buffer<double, 1> count_sp_buf(count_sp.data(), count_sp.size());

        q.submit([&](sycl::handler& h) {
            auto rng_acc = rng_buf.get_access<sycl::access::mode::read>(h);
            auto count_sc_acc = count_sc_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::nd_range<1>(wg_size * wg_num, wg_size),
                           [=](sycl::nd_item<1> item) {
                               double rng;
                               double sc;

                               rng = rng_acc[item.get_global_linear_id()] * initial_stock_price;
                               sc = sycl::max(rng - strike_price, 0.0);

                               count_sc_acc[item.get_group_linear_id()] =
                                   reduce(item.get_group(), sc, std::plus<double>());
                           });
        });

        q.submit([&](sycl::handler& h) {
            auto rng_acc = rng_buf.get_access<sycl::access::mode::read>(h);
            auto count_sp_acc = count_sp_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::nd_range<1>(wg_size * wg_num, wg_size),
                           [=](sycl::nd_item<1> item) {
                               double rng;
                               double sp;

                               rng = rng_acc[item.get_global_linear_id()] * initial_stock_price;
                               sp = sycl::max(strike_price - rng, 0.0);

                               count_sp_acc[item.get_group_linear_id()] =
                                   reduce(item.get_group(), sp, std::plus<double>());
                           });
        });
    }

    sc_dp = std::accumulate(count_sc.begin(), count_sc.end(), 0);
    sp_dp = std::accumulate(count_sp.begin(), count_sp.end(), 0);

    vcall = sc_dp / num_samples * exp(-risk_neutral_rate * t);
    vput = sp_dp / num_samples * exp(-risk_neutral_rate * t);
}

void mc_calculate(sycl::queue& q, size_t num_options, size_t num_samples, double risk_neutral_rate,
                  double volatility, std::vector<double>& initial_stock_price,
                  std::vector<double>& strike_price, std::vector<double>& t,
                  std::vector<double>& vcall, std::vector<double>& vput) {
    // Creating random number engine
    mkl::rng::philox4x32x10 engine(q, seed);
    // Create buffer for random numbers
    sycl::buffer<double> rng_buf(num_samples);
    // Price options
    for (size_t j = 0; j < num_options; j++) {
        mc_kernel(q, engine, num_samples, risk_neutral_rate, volatility, initial_stock_price[j],
                  strike_price[j], t[j], vcall[j], vput[j], rng_buf);
    }
}

int main(int argc, char** argv) {
    std::cout << std::endl;
    std::cout << "Monte Carlo European Option Pricing Simulation" << std::endl;
    std::cout << "Buffer Api" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    size_t num_options = num_options_default;
    size_t num_samples = num_samples_default;
    if (argc >= 2) {
        num_options = atol(argv[1]);
        if (num_options == 0) {
            num_options = num_options_default;
        }
        if (argc >= 3) {
            num_samples = atol(argv[2]);
            if (num_samples == 0) {
                num_samples = num_samples_default;
            }
        }
    }
    std::cout << "Number of options = " << num_options << std::endl;
    std::cout << "Number of samples = " << num_samples << std::endl;

    std::vector<double> initial_stock_price(num_options), strike_price(num_options), t(num_options),
        vcall(num_options), vput(num_options);
    std::vector<double> vcall_ref(num_options), vput_ref(num_options);

    init_data(initial_stock_price, strike_price, t, vcall, vput);

    // This exception handler with catch async exceptions
    auto exception_handler = [&](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
                std::terminate();
            }
        }
    };

    try {
        // Queue constructor passed exception handler
        sycl::queue q(sycl::default_selector{}, exception_handler);

        // Check for double precision support
        auto device = q.get_device();
        if (device.get_info<sycl::info::device::double_fp_config>().empty()) {
            std::cerr << "The sample uses double precision, which is not supported" << std::endl;
            std::cerr << "by the selected device. Quitting." << std::endl;
            return 0;
        }

        // Launch Monte Carlo calculation
        mc_calculate(q, num_options, num_samples, default_risk_neutral_rate, default_volatility,
                     initial_stock_price, strike_price, t, vcall, vput);
    }
    catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Validate results
    bs_ref(default_risk_neutral_rate, default_volatility, initial_stock_price, strike_price, t,
           vcall_ref, vput_ref);

    std::cout << "put_abs_err  = " << estimate_error(vput_ref, vput) << std::endl;
    std::cout << "call_abs_err = " << estimate_error(vcall_ref, vcall) << std::endl;

    return 0;
}

static double RandDouble(double a, double b) {
    return rand() / (double)RAND_MAX * (b - a) + a;
}

void init_data(std::vector<double>& initial_stock_price, std::vector<double>& strike_price,
               std::vector<double>& t, std::vector<double>& vcall, std::vector<double>& vput) {
    srand(seed);

    // Initialize data
    for (size_t i = 0; i < initial_stock_price.size(); i++) {
        initial_stock_price[i] = RandDouble(initial_stock_price_l, initial_stock_price_h);
        strike_price[i] = RandDouble(strike_price_l, strike_price_h);
        t[i] = RandDouble(t_l, t_h);

        vcall[i] = 0.0;
        vput[i] = 0.0;
    }
}

static double CNDF(double x) {
    const double b1 = 0.319381530;
    const double b2 = -0.356563782;
    const double b3 = 1.781477937;
    const double b4 = -1.821255978;
    const double b5 = 1.330274429;
    const double p = 0.2316419;
    const double c = 0.39894228;

    if (x >= 0.0) {
        double t = 1.0 / (1.0 + p * x);
        return (1.0 - c * exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
    }
    else {
        double t = 1.0 / (1.0 - p * x);
        return (c * exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
    }
}

static double erf_ref(double x) {
    return 2.0 * CNDF(1.4142135623730950488016887242097 * x) - 1.0;
}

static const auto inv_sqrt2 = 0.7071067811865475727373109293694142252207;

void bs_ref(double risk_neutral_rate, double volatility, std::vector<double>& initial_stock_price,
            std::vector<double>& strike_price, std::vector<double>& t, std::vector<double>& vcall,
            std::vector<double>& vput) {
    double a, b, c, y, z, e, d1, d2, w1, w2;

    for (size_t i = 0; i < initial_stock_price.size(); i++) {
        a = log(initial_stock_price[i] / strike_price[i]);
        b = t[i] * risk_neutral_rate;
        z = t[i] * volatility * volatility;

        c = 0.5 * z;
        e = exp(-b);
        y = 1.0 / sqrt(z);

        w1 = (a + b + c) * y;
        w2 = (a + b - c) * y;
        d1 = erf_ref(inv_sqrt2 * w1);
        d2 = erf_ref(inv_sqrt2 * w2);
        d1 = 0.5 + 0.5 * d1;
        d2 = 0.5 + 0.5 * d2;

        vcall[i] = initial_stock_price[i] * d1 - strike_price[i] * e * d2;
        vput[i] = vcall[i] - initial_stock_price[i] + strike_price[i] * e;
    }
}

double estimate_error(std::vector<double>& v_put_ref, std::vector<double>& v_put) {
    double abs_err = 0.0;
    double abs_err_res = 0.0;

    for (size_t i = 0; i < v_put_ref.size(); ++i) {
        abs_err = (v_put_ref[i] - v_put[i]);
        if (abs_err < 0) {
            abs_err = -abs_err;
        }
        if (abs_err_res < abs_err) {
            abs_err_res = abs_err;
        }
    }
    return abs_err_res;
}
