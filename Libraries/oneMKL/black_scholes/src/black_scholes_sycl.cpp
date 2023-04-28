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
template<typename DataType, int>
class k_BlackScholes;

template <typename T>
__attribute__((always_inline))
static inline T CNDF_C(T InputX)
{
    constexpr T inv_sqrt_2xPI = 0.39894228040143270286;
    constexpr T CNDF_C1 = 0.2316419;
    constexpr T CNDF_C2 = 0.319381530;
    constexpr T CNDF_C3 = -0.356563782;
    constexpr T CNDF_C4 = 1.781477937;
    constexpr T CNDF_C5 = -1.821255978;
    constexpr T CNDF_C6 = 1.330274429;
    constexpr T _M_LN2 = 0.693147180559945309417;
    constexpr T INV_LN2X2 = 1.0 / (_M_LN2 * 2.0);

    T X = (InputX < 0.0) ? -InputX : InputX;

    T k2 = 1.0 / (1.0 + CNDF_C1 * X);
    T k2_2 = k2 * k2;
    T k2_3 = k2_2 * k2;
    T k2_4 = k2_3 * k2;
    T k2_5 = k2_4 * k2;

    T OutputX = 1.0 - (inv_sqrt_2xPI * sycl::exp2(-X * X * INV_LN2X2) * ((CNDF_C2 * k2) +
        ((CNDF_C3 * (k2_2)) + (CNDF_C4 * (k2_3)) + (CNDF_C5 * (k2_4)) + (CNDF_C6 * (k2_5)))));
    if (InputX < 0.0)
        OutputX = (1.0 - OutputX);

    return OutputX;
}

using DataType = BS_PRECISION;

void BlackScholes::body() {
    // this can not be captured to the kernel. So, we need to copy internals of the class to local variables
    DataType* h_StockPrice_local = this->h_StockPrice;
    DataType* h_OptionYears_local = this->h_OptionYears;
    DataType* h_OptionStrike_local = this->h_OptionStrike;
    DataType* h_CallResult_local = this->h_CallResult;
    DataType* h_PutResult_local = this->h_PutResult;

    black_scholes_queue->parallel_for<k_BlackScholes<DataType, block_size>>(sycl::nd_range(sycl::range<1>(OPT_N / block_size), sycl::range<1>(wg_size)),
                [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(sg_size)]] {
                        auto local_id = item.get_local_linear_id();
                        auto group_id = item.get_group_linear_id();
#pragma unroll
                        for (size_t opt = group_id * block_size * wg_size + local_id, i = 0; i < block_size; opt += wg_size, i++) {
                            const DataType S = h_StockPrice_local[opt];
#if USE_CNDF_C
                            const DataType T = h_OptionYears_local[opt];
                            const DataType X = h_OptionStrike_local[opt];
                            const DataType v_sqrtT = VOLATILITY * sycl::sqrt(T);
                            const DataType d1 = (sycl::log(S / X) + (RISKFREE + 0.5 * VOLATILITY * VOLATILITY) * T) / v_sqrtT;
                            const DataType d2 = d1 - v_sqrtT;
                            const DataType CNDD1 = CNDF_C(d1);
                            const DataType CNDD2 = CNDF_C(d2);
                            const DataType XexpRT = X * sycl::exp(-RISKFREE * T);
                            const DataType CallVal = S * CNDD1 - XexpRT * CNDD2;
                            const DataType PutVal = CallVal + XexpRT - S;
#else

                            const DataType L = h_OptionStrike_local[opt];
                            const DataType t = h_OptionYears_local[opt];
                            constexpr DataType r = RISKFREE;
                            constexpr DataType sigma = VOLATILITY;
                            constexpr DataType _M_SQRT1_2 = 0.707106781186547524401;
                            double N_d1 = DataType(1. / 2.) + DataType(1. / 2.) * sycl::erf(((sycl::log(S / L) + (r + DataType(0.5) * sigma * sigma) * t) / (sigma * sycl::sqrt(t))) * _M_SQRT1_2);
                            double N_d2 = DataType(1. / 2.) + DataType(1. / 2.) * sycl::erf(((sycl::log(S / L) + (r - DataType(0.5) * sigma * sigma) * t) / (sigma * sycl::sqrt(t))) * _M_SQRT1_2);

                            DataType expRT = sycl::exp(-r * t);
                            const DataType CallVal = (S * N_d1 - L * expRT * N_d2);
                            const DataType PutVal = (L * expRT * (1.0 - N_d2) - S * (1.0 - N_d1));
#endif
                            h_CallResult_local[opt] = CallVal;
                            h_PutResult_local[opt] = PutVal;
                        }
                });
}

BlackScholes::BlackScholes()
{
    black_scholes_queue = new sycl::queue;

    h_CallResult = sycl::malloc_shared<DataType>(OPT_N, *black_scholes_queue);
    h_PutResult = sycl::malloc_shared<DataType>(OPT_N, *black_scholes_queue);
    h_StockPrice = sycl::malloc_shared<DataType>(OPT_N, *black_scholes_queue);
    h_OptionStrike = sycl::malloc_shared<DataType>(OPT_N, *black_scholes_queue);
    h_OptionYears = sycl::malloc_shared<DataType>(OPT_N, *black_scholes_queue);

    black_scholes_queue->fill(h_CallResult, 0.0, OPT_N);
    black_scholes_queue->fill(h_PutResult, 0.0, OPT_N);

    constexpr int rand_seed = 777;
    namespace mkl_rng = oneapi::mkl::rng;
    mkl_rng::philox4x32x10 engine(
#if !INIT_ON_HOST
        *black_scholes_queue,
#else
        sycl::queue{sycl::cpu_selector_v},
#endif
        rand_seed); // random number generator object

    sycl::event event_1 = mkl_rng::generate(mkl_rng::uniform<DataType>(5.0, 50.0), engine, OPT_N, h_StockPrice);
    sycl::event event_2 = mkl_rng::generate(mkl_rng::uniform<DataType>(10.0, 25.0), engine, OPT_N, h_OptionStrike);
    sycl::event event_3 = mkl_rng::generate(mkl_rng::uniform<DataType>(1.0, 5.0), engine, OPT_N, h_OptionYears);
    sycl::event::wait({event_1, event_2, event_3});
}

BlackScholes::~BlackScholes()
{
    sycl::free(h_CallResult, *black_scholes_queue);
    sycl::free(h_PutResult, *black_scholes_queue);
    sycl::free(h_StockPrice, *black_scholes_queue);
    sycl::free(h_OptionStrike, *black_scholes_queue);
    sycl::free(h_OptionYears, *black_scholes_queue);
    delete black_scholes_queue;
}

void BlackScholes::run()
{
    std::printf("%s Precision Black&Scholes Option Pricing version %d.%d running on %s using DPC++, workgroup size %d, sub-group size %d.\n",
        sizeof(DataType) > 4 ? "Double" : "Single", MAJOR, MINOR, black_scholes_queue->get_device().get_info<sycl::info::device::name>().c_str(), wg_size, sg_size);

    std::printf("Compiler Version: %s, LLVM %d.%d based.\n", __VERSION__, __clang_major__, __clang_minor__);
    std::printf("Driver Version  : %s\n", black_scholes_queue->get_device().get_info<sycl::info::device::driver_version>().c_str());
    std::printf("Build Time      : %s %s\n", __DATE__, __TIME__);
    std::printf("Input Dataset   : %zu\n", OPT_N);
    size_t total_options = 2 * OPT_N   /*Pricing Call and Put options at the same time, so 2*num_options*/ * ITER_N;

    body();
    black_scholes_queue->wait();

    std::printf("Pricing %zu Options in %d iterations, %zu Options in total.\n", 2 * OPT_N, ITER_N, total_options); fflush(stdout);
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
