//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*******************************************************************************
!  Content:
!      Black-Scholes formula Intel(r) Math Kernel Library (Intel(r) MKL) VML based Example
!******************************************************************************/

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/rng/device.hpp"

using namespace oneapi;

#include "input_generator.hpp"
#include "black_scholes.hpp"
#include "code_wrapper.tpp"

namespace {

using std::int64_t;
using std::uint64_t;
using std::size_t;




#if (defined (ACC_ep))
auto vml_accuracy = mkl::vm::mode::ep;
#elif (defined (ACC_la))
auto vml_accuracy = mkl::vm::mode::la;
#elif(defined (ACC_ha))
auto vml_accuracy = mkl::vm::mode::ha;
#else
auto vml_accuracy = mkl::vm::mode::not_defined;
#endif

enum class dev_select : int {
    cpu = 1,
    gpu = 2
};

constexpr uint64_t seed = UINT64_C(0x1234'5678'09ab'cdef);

constexpr double s0_low  = 10.0;
constexpr double s0_high = 50.0;

constexpr double x_low   = 10.0;
constexpr double x_high  = 50.0;

constexpr double t_low   = 1.0;
constexpr double t_high  = 2.0;

constexpr double risk_free = 1.0;
constexpr double volatility = 2.0;

void preamble(sycl::device & dev) {
    std::string dev_name       = dev.template get_info<sycl::info::device::name>();
    std::string driver_version = dev.template get_info<sycl::info::device::version>();

    std::cerr << std::endl
              << "running on:         " << std::endl
              << "       device name: " << dev_name << std::endl
              << "    driver version: " << driver_version << std::endl;
}

void async_sycl_error(sycl::exception_list el) {
    std::cerr << "async exceptions caught: " << std::endl;

    for (auto l = el.begin(); l != el.end(); ++l) {
        try {
            std::rethrow_exception(*l);
        } catch(const sycl::exception & e) {
            std::cerr << "SYCL exception occured with code " << code_wrapper(e) << " with " << e.what() << std::endl;
        }
    }
}

template <typename T>
void print_input_stats(int64_t nopt, T * s0, T * x, T * t) {
    double avg_s0 = std::accumulate(s0, s0 + nopt, 0.0) / nopt;
    double avg_x  = std::accumulate(x, x + nopt, 0.0) / nopt;
    double avg_t  = std::accumulate(t, t + nopt, 0.0) / nopt;

    std::cerr << "    <s0> = " << avg_s0 << std::endl;
    std::cerr << "    <x> = "  << avg_x << std::endl;
    std::cerr << "    <t> = "  << avg_t << std::endl;
}

template <typename T>
void print_result(int64_t nopt, T * opt_call, T * opt_put) {
    double avg_opt_call  = std::accumulate(opt_call, opt_call + nopt, 0.0) / nopt;
    double avg_opt_put   = std::accumulate(opt_put, opt_put + nopt, 0.0) / nopt;

    std::cerr << "    <opt_call> = "  << avg_opt_call << std::endl
              << "    <opt_put>  = "  << avg_opt_put  << std::endl;
}

template <typename T>
void run(int64_t nopt, sycl::device & dev) {
    sycl::queue q { dev, async_sycl_error };
    std::vector<T> s0(nopt);
    std::vector<T> x(nopt);
    std::vector<T> t(nopt);

    std::vector<T> opt_call(nopt);
    std::vector<T> opt_put(nopt);



    for (int var = 0; var < 4; ++var) {

#ifdef _WIN32
        if (var & 1) continue; // Skip RNG device APIs on Windows for now.
#endif

        std::fill(s0.begin(), s0.end(), static_cast<T>(0.0));
        std::fill(x.begin(), x.end(), static_cast<T>(0.0));
        std::fill(t.begin(), t.end(), static_cast<T>(0.0));


        std::cerr << "\n\n";
        input_generator::run(
            var,
            nopt,
            static_cast<T>(s0_low), static_cast<T>(s0_high), s0.data(),
            static_cast<T>(x_low), static_cast<T>(x_high), x.data(),
            static_cast<T>(t_low), static_cast<T>(t_high), t.data(),
            seed,
            q);

        print_input_stats(nopt, s0.data(), x.data(), t.data());

        std::fill(opt_call.begin(), opt_call.end(), static_cast<T>(std::nan("")));
        std::fill(opt_put.begin(), opt_put.end(), static_cast<T>(std::nan("")));

        black_scholes::run(
            var,
            nopt,
            static_cast<T>(risk_free),
            static_cast<T>(volatility),
            s0.data(),
            x.data(),
            t.data(),
            opt_call.data(),
            opt_put.data(),
            vml_accuracy,
            q);
         print_result(nopt, opt_call.data(), opt_put.data());
    }
}


int sample_run(int64_t nopt) {
    try {
        sycl::device dev{sycl::default_selector{}};

        preamble(dev);

        std::cerr << std::endl
                  << "running floating-point type float" << std::endl;
        run<float>(nopt, dev);

        // check if double is supported and run if it is supported
        auto fp64_conf = dev.template get_info<sycl::info::device::double_fp_config>();
        if (0 != fp64_conf.size()) {
            std::cerr << std::endl
                      << "running floating-point type double" << std::endl;
            run<double>(nopt, dev);
        } else {
            std::cerr << "floating-point type double is not supported on this device" << std::endl;
        }
    }
    catch (sycl::exception const & re) {
        std::cerr << "SYCL exception occured with code " << code_wrapper(re) << " with " << re.what() << std::endl;
        return -1;
    }

    return 0;
}

} // anon. namespace

int main(int argc, char * argv[]) {
    std::int64_t nopt;

    if (argc > 1) {
        auto nopt_param = std::string { argv[1] };
        nopt = std::stol(nopt_param);
        if (nopt <= 0) {
            std::cerr << "nopt <= 0" << std::endl;
            return 1;
        }
    } else {
        nopt = 10'000'000;
    }

    return sample_run(nopt);
}
