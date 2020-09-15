//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This file contains Monte Carlo simulatuon for European option pricing
*       for DPC++ USM-based interface of random number generators.
*
*******************************************************************************/

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "mkl_rng_sycl.hpp"

// Temporary code for compatibility with beta08.
// oneMKL moves to the oneapi namespace in beta09.
namespace oneapi {}
using namespace oneapi;

// Default number of options
#define N_OPT  2048
// Default number of independent samples
#define N_SAMPLES 262144

// Initialization value for random number generator
#define SEED        7777

#define S0L     10.0
#define S0H     50.0
#define XL      10.0
#define XH      50.0
#define TLL      0.2
#define TH       2.0
#define RISK_FREE  0.05
#define VOLATILITY  0.2

void init_data(std::vector<double>& s0, std::vector<double>& x, std::vector<double>& t,
                std::vector<double>& vcall, std::vector<double>& vput);

// Black-Scholes formula to verify Monte Carlo computation
void bs_ref(double r, double sig, std::vector<double>& s0, std::vector<double>& x, std::vector<double>& t,
            std::vector<double>& vcall, std::vector<double>& vput);

double estimate_error(std::vector<double>& v_put_ref, std::vector<double>& v_put);

template<typename EngineType>
static void mc_kernel(sycl::queue& q, EngineType& engine, size_t n_samples,
                        double r, double sig, double s0, double x, double t, double& vcall, double& vput,
                        double* rng_ptr) {
    double a, nu;
    double sc_dp, sp_dp;
    double st;

    a  = (r - sig * sig * 0.5) * t;
    nu = sig * sqrt(t);

    mkl::rng::lognormal<double, mkl::rng::lognormal_method::box_muller2> distr(a, nu);

    auto event = mkl::rng::generate(distr, engine, n_samples, rng_ptr);

    size_t wg_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(), n_samples);
    size_t wg_num = n_samples / wg_size;

    double* count_sc  = sycl::malloc_shared<double>(wg_num, q);
    double* count_sp  = sycl::malloc_shared<double>(wg_num, q);

    event.wait_and_throw();

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(wg_size * wg_num, wg_size),
            [=](sycl::nd_item<1> item) {
            double rng;
            double sc;

            rng = rng_ptr[item.get_global_linear_id()];
            rng *= s0;
            sc = sycl::max(rng - x, 0.0);

            count_sc[item.get_group_linear_id()] = sycl::intel::reduce(item.get_group(), sc, std::plus<double>());
        });
    });

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(wg_size * wg_num, wg_size),
            [=](sycl::nd_item<1> item) {
            double rng;
            double sp;

            rng = rng_ptr[item.get_global_linear_id()];
            rng *= s0;
            sp = sycl::max(x - rng, 0.0);

            count_sp[item.get_group_linear_id()] = sycl::intel::reduce(item.get_group(), sp, std::plus<double>());
        });
    });
    q.wait_and_throw();

    sc_dp = std::accumulate(count_sc, count_sc + wg_num, 0);
    sp_dp = std::accumulate(count_sp, count_sp + wg_num, 0);

    vcall = sc_dp / n_samples * exp(-r * t);
    vput  = sp_dp / n_samples * exp(-r * t);

    sycl::free(count_sc, q);
    sycl::free(count_sp, q);
}

void mc_calculate(sycl::queue& q, size_t n_opt, size_t n_samples, double r, double sig,
    std::vector<double>& s0, std::vector<double>& x, std::vector<double>& t,
    std::vector<double>& vcall, std::vector<double>& vput) {
    // Creating random number engine
    mkl::rng::philox4x32x10 engine(q, SEED);
    // Allocate memory for random numbers
    double* rng_ptr = sycl::malloc_device<double>(n_samples, q);
    // Price options
    for (size_t j = 0; j < n_opt; j++) {
        mc_kernel(q, engine, n_samples, r, sig, s0[j], x[j], t[j], vcall[j], vput[j], rng_ptr);
    }
    sycl::free(rng_ptr, q);
}

int main(int argc, char ** argv) {

    size_t n_opt = N_OPT;
    size_t n_samples = N_SAMPLES;
    if(argc >= 2) {
        n_opt = atol(argv[1]);
        if(n_opt == 0) {
            n_opt = N_OPT;
        }
        if(argc >= 3) {
            n_samples = atol(argv[2]);
            if(n_samples == 0) {
                n_samples = N_SAMPLES;
            }
        }
    }
    std::cout << "Number of options = " << n_opt << std::endl;
    std::cout << "Number of samples = " << n_samples << std::endl;

    std::vector<double> s0(n_opt), x(n_opt), t(n_opt), vcall(n_opt), vput(n_opt);
    std::vector<double> vcall_ref(n_opt), vput_ref(n_opt);

    init_data(s0, x, t, vcall, vput);

    // This exception handler with catch async exceptions
    auto exception_handler = [&](sycl::exception_list exceptions) {
        for(std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const& e) {
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
        
        // Launch calculation
        mc_calculate(q, n_opt, n_samples, RISK_FREE, VOLATILITY, s0, x, t, vcall, vput);
    } catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Validate results
    bs_ref(RISK_FREE, VOLATILITY, s0, x, t, vcall_ref, vput_ref);

    std::cout << "put_abs_err  = " << estimate_error(vput_ref, vput) << std::endl;
    std::cout << "call_abs_err = " << estimate_error(vcall_ref, vcall) << std::endl;

    return 0;
}

static double RandDouble(double a, double b) {
    return rand()/(double)RAND_MAX*(b-a) + a;
}

void init_data(std::vector<double>& s0, std::vector<double>& x, std::vector<double>& t,
    std::vector<double>& vcall, std::vector<double>& vput) {

    srand(SEED);

    // Initialize data
    for(size_t i = 0; i < s0.size(); i++) {
        s0[i] = RandDouble(S0L, S0H);
        x[i]  = RandDouble(XL, XH);
        t[i]  = RandDouble(TLL, TH);

        vcall[i] = 0.0;
        vput[i] = 0.0;
    }
}

static double CNDF(double x) {
    const double b1 =  0.319381530;
    const double b2 = -0.356563782;
    const double b3 =  1.781477937;
    const double b4 = -1.821255978;
    const double b5 =  1.330274429;
    const double p  =  0.2316419;
    const double c  =  0.39894228;

    if(x >= 0.0) {
        double t = 1.0 / ( 1.0 + p * x );
        return (1.0 - c * exp( -x * x / 2.0 ) * t *
        (t *(t * (t * (t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
    }
    else {
        double t = 1.0 / ( 1.0 - p * x );
        return (c * exp( -x * x / 2.0 ) * t *
        (t *(t * (t * (t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
    }
}

static double erf_ref(double x) {
    return 2.0 * CNDF(1.4142135623730950488016887242097*x) - 1.0;
}

#define INV_SQRT2 0.7071067811865475727373109293694142252207

void bs_ref(double r, double sig, std::vector<double>& s0, std::vector<double>& x, std::vector<double>& t,
            std::vector<double>& vcall, std::vector<double>& vput) {

    double a, b, c, y, z, e, d1, d2, w1, w2;

    for(size_t i = 0; i < s0.size(); i++) {
        a = log(s0[i] / x[i]);
        b = t[i] * r;
        z = t[i]*sig*sig;
    
        c = 0.5 * z;
        e = exp(-b);
        y = 1.0 / sqrt(z);
                         
        w1 = (a + b + c) * y;
        w2 = (a + b - c) * y;
        d1 = erf_ref(INV_SQRT2 * w1);
        d2 = erf_ref(INV_SQRT2 * w2);
        d1 = 0.5 + 0.5 * d1;
        d2 = 0.5 + 0.5 * d2;

        vcall[i] = s0[i] * d1 - x[i] * e * d2;
        vput[i]  = vcall[i] - s0[i] + x[i] * e;
    }
}

double estimate_error(std::vector<double>& v_put_ref, std::vector<double>& v_put) {
    double abs_err = 0.0;
    double abs_err_res = 0.0;

    for(size_t i = 0; i < v_put_ref.size(); ++i) {
        abs_err = (v_put_ref[i] - v_put[i]);
        if(abs_err < 0) {
            abs_err = -abs_err;
        }
        if(abs_err_res < abs_err) {
            abs_err_res = abs_err;
        }
    }
    return abs_err_res;
}
