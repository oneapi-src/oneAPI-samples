//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include<cstdio>

#if !SYCL_LANGUAGE_VERSION
#error "SYCL is not enabled""
#endif

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "black_scholes.hpp"

constexpr int sg_size = 32;
#if NON_DEFAULT_SIZE
constexpr int wg_size = 128;
constexpr int block_size = 1;
#else
constexpr int wg_size = 256;
constexpr int block_size = 4;
#endif

sycl::queue* black_scholes_queue;

template<typename Type, int>
class k_BlackScholes;

#if USE_CNDF_C

template <typename T>
__attribute__((always_inline))
static inline T CNDF_C(T input)
{
    constexpr T inv_sqrt_2xPI = 0.39894228040143270286;
    constexpr T CNDF_C1 = 0.2316419;
    constexpr T CNDF_C2 = 0.319381530;
    constexpr T CNDF_C3 = -0.356563782;
    constexpr T CNDF_C4 = 1.781477937;
    constexpr T CNDF_C5 = -1.821255978;
    constexpr T CNDF_C6 = 1.330274429;
    constexpr T CNDF_LN2 = 0.693147180559945309417;
    constexpr T INV_LN2X2 = 1.0 / (CNDF_LN2 * 2.0);

    T x = (input < 0.0) ? -input : input;

    T k2 = 1.0 / (1.0 + CNDF_C1 * x);
    T k2_2 = k2 * k2;
    T k2_3 = k2_2 * k2;
    T k2_4 = k2_3 * k2;
    T k2_5 = k2_4 * k2;

    T output = 1.0 - (inv_sqrt_2xPI * sycl::exp2(-x * x * INV_LN2X2) * ((CNDF_C2 * k2) +
        ((CNDF_C3 * (k2_2)) + (CNDF_C4 * (k2_3)) + (CNDF_C5 * (k2_4)) + (CNDF_C6 * (k2_5)))));
    if (input < 0.0)
        output = (1.0 - output);

    return output;
}
#endif // USE_CNDF_C

void BlackScholes::body() {
    // this can not be captured to the kernel. So, we need to copy internals of the class to local variables
    DATA_TYPE* h_stock_price_local = this->h_stock_price;
    DATA_TYPE* h_option_years_local = this->h_option_years;
    DATA_TYPE* h_option_strike_local = this->h_option_strike;
    DATA_TYPE* h_call_result_local = this->h_call_result;
    DATA_TYPE* h_put_result_local = this->h_put_result;

    black_scholes_queue->parallel_for<k_BlackScholes<DATA_TYPE, block_size>>(sycl::nd_range(sycl::range<1>(opt_n / block_size), sycl::range<1>(wg_size)),
                [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(sg_size)]] {
                        auto local_id = item.get_local_linear_id();
                        auto group_id = item.get_group_linear_id();
#pragma unroll
                        for (size_t opt = group_id * block_size * wg_size + local_id, i = 0; i < block_size; opt += wg_size, i++) {
                            constexpr DATA_TYPE sigma = volatility;
                            const DATA_TYPE s = h_stock_price_local[opt];
                            const DATA_TYPE t = h_option_years_local[opt];
                            const DATA_TYPE x = h_option_strike_local[opt];
                            const DATA_TYPE XexpRT = x * sycl::exp(-risk_free * t);
#if USE_CNDF_C
                            const DATA_TYPE v_sqrt = sigma * sycl::sqrt(t);
                            const DATA_TYPE d1 = (sycl::log(s / x) + (risk_free + DATA_TYPE(0.5) * sigma * sigma) * t) / v_sqrt;
                            const DATA_TYPE d2 = d1 - v_sqrt;
                            const DATA_TYPE n_d1 = CNDF_C(d1);
                            const DATA_TYPE n_d2 = CNDF_C(d2);
#else
                            constexpr DATA_TYPE sqrt1_2 = 0.707106781186547524401;
                            DATA_TYPE n_d1 = DATA_TYPE(1. / 2.) + DATA_TYPE(1. / 2.) * sycl::erf(((sycl::log(s / x) + (risk_free + DATA_TYPE(0.5) * sigma * sigma) * t) / (sigma * sycl::sqrt(t))) * sqrt1_2);
                            DATA_TYPE n_d2 = DATA_TYPE(1. / 2.) + DATA_TYPE(1. / 2.) * sycl::erf(((sycl::log(s / x) + (risk_free - DATA_TYPE(0.5) * sigma * sigma) * t) / (sigma * sycl::sqrt(t))) * sqrt1_2);
#endif // USE_CNDF_C
                            const DATA_TYPE call_val = s * n_d1 - XexpRT * n_d2;
                            const DATA_TYPE put_val = call_val + XexpRT - s;
                            h_call_result_local[opt] = call_val;
                            h_put_result_local[opt] = put_val;
                        }
                });
}

BlackScholes::BlackScholes()
{
    black_scholes_queue = new sycl::queue;

    h_call_result = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
    h_put_result = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
    h_stock_price = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
    h_option_strike = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);
    h_option_years = sycl::malloc_shared<DATA_TYPE>(opt_n, *black_scholes_queue);

    black_scholes_queue->fill(h_call_result, 0.0, opt_n);
    black_scholes_queue->fill(h_put_result, 0.0, opt_n);

    constexpr int rand_seed = 777;
    namespace mkl_rng = oneapi::mkl::rng;
    // create random number generator object
    mkl_rng::philox4x32x10 engine(
#if !INIT_ON_HOST
        *black_scholes_queue,
#else
        sycl::queue{sycl::cpu_selector_v},
#endif // !INIT_ON_HOST
        rand_seed);

    sycl::event event_1 = mkl_rng::generate(mkl_rng::uniform<DATA_TYPE>(5.0, 50.0), engine, opt_n, h_stock_price);
    sycl::event event_2 = mkl_rng::generate(mkl_rng::uniform<DATA_TYPE>(10.0, 25.0), engine, opt_n, h_option_strike);
    sycl::event event_3 = mkl_rng::generate(mkl_rng::uniform<DATA_TYPE>(1.0, 5.0), engine, opt_n, h_option_years);
    sycl::event::wait({event_1, event_2, event_3});
}

BlackScholes::~BlackScholes()
{
    sycl::free(h_call_result, *black_scholes_queue);
    sycl::free(h_put_result, *black_scholes_queue);
    sycl::free(h_stock_price, *black_scholes_queue);
    sycl::free(h_option_strike, *black_scholes_queue);
    sycl::free(h_option_years, *black_scholes_queue);
    delete black_scholes_queue;
}

void BlackScholes::run()
{
    std::printf("%s Precision Black&Scholes Option Pricing version %d.%d running on %s using DPC++, workgroup size %d, sub-group size %d.\n",
        sizeof(DATA_TYPE) > 4 ? "Double" : "Single", MAJOR, MINOR, black_scholes_queue->get_device().get_info<sycl::info::device::name>().c_str(), wg_size, sg_size);

    std::printf("Compiler Version: %s, LLVM %d.%d based.\n", __VERSION__, __clang_major__, __clang_minor__);
    std::printf("Driver Version  : %s\n", black_scholes_queue->get_device().get_info<sycl::info::device::driver_version>().c_str());
    std::printf("Build Time      : %s %s\n", __DATE__, __TIME__);
    std::printf("Input Dataset   : %zu\n", opt_n);
    size_t total_options = 2 * opt_n   /*Pricing Call and Put options at the same time, so 2*num_options*/ * ITER_N;

    body();
    black_scholes_queue->wait();

    std::printf("Pricing %zu Options in %d iterations, %zu Options in total.\n", 2 * opt_n, ITER_N, total_options); fflush(stdout);
    timer t{};
    t.start();

    for (int i = 0; i < ITER_N; i++) {
        body();
    }
    black_scholes_queue->wait();

    t.stop();

    std::printf("Completed in %10.5f seconds. GOptions per second: %10.5f\n", t.duration(), static_cast<double>(total_options) / t.duration() / 1e9);
    std::printf("Time Elapsed =  %10.5f seconds\n", t.duration()); fflush(stdout);
}

int main(int const argc, char const* argv[])
{
    BlackScholes test{};
    test.run();
    test.check();

    return 0;
}
