//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This file contains Monte Carlo Pi number evaluation benchmark for DPC++ 
*       USM-based interface of random number generators.
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

// Value of Pi with many exact digits to compare with estimated value of Pi
static const auto pi = 3.1415926535897932384626433832795;

// Initialization value for random number generator
static const auto seed = 7777;

// Default Number of 2D points
static const auto n_samples = 120000000;

double estimate_pi(sycl::queue& q, size_t n_points) {
    double estimated_pi;         // Estimated value of Pi
    size_t n_under_curve = 0;    // Number of points fallen under the curve

    // Step 1. Generate n_points * 2 random numbers
    // 1.1. Generator initialization
    // Create an object of basic random numer generator (engine)
    mkl::rng::philox4x32x10 engine(q, seed);
    // Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
    mkl::rng::uniform distr;

    float* rng_ptr = sycl::malloc_device<float>(n_points * 2, q);

    // 1.2. Random number generation
    auto event = mkl::rng::generate(distr, engine, n_points * 2, rng_ptr);

    // Step 2. Count points under curve (x ^ 2 + y ^ 2 < 1.0f)
    size_t wg_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(), n_points);
    size_t max_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    size_t wg_num = (n_points > wg_size * max_compute_units) ? max_compute_units : 1;

    size_t count_per_thread = n_points / (wg_size * wg_num);

    size_t* count_ptr = sycl::malloc_shared<size_t>(wg_num, q);

    // Make sure, that generation is finished
    event.wait_and_throw();

    event = q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(wg_size * wg_num, wg_size),
            [=](sycl::nd_item<1> item) {
            sycl::vec<float, 2> r;
            size_t count = 0;
            for(int i = 0; i < count_per_thread; i++) {
                r.load(i + item.get_global_linear_id() * count_per_thread, sycl::global_ptr<float>(rng_ptr));
                if(sycl::length(r) <= 1.0f) {
                    count += 1;
                }
            }
            count_ptr[item.get_group_linear_id()] = reduce(item.get_group(), count, std::plus<size_t>());
        });
    });

    event.wait_and_throw();

    n_under_curve = std::accumulate(count_ptr, count_ptr + wg_num, 0);

    // Step 3. Calculate approximated value of Pi
    estimated_pi = n_under_curve / ((double)n_points) * 4.0;

    sycl::free(rng_ptr, q);
    sycl::free(count_ptr, q);

    return estimated_pi;

}

int main(int argc, char ** argv) {

    std::cout << std::endl;
    std::cout << "Monte Carlo pi Calculation Simulation" << std::endl;
    std::cout << "Unified Shared Memory Api" << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    double estimated_pi;
    size_t n_points = n_samples;
    if(argc >= 2) {
        n_points = atol(argv[1]);
        if(n_points == 0) {
            n_points = n_samples;
        }
    }
    std::cout << "Number of points = " << n_points << std::endl;

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
        // Launch Pi number calculation
        estimated_pi = estimate_pi(q, n_points);
    } catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Printing results
    std::cout << "Estimated value of Pi = " << estimated_pi << std::endl;
    std::cout << "Exact value of Pi = " << pi << std::endl;
    std::cout << "Absolute error = " << fabs(pi-estimated_pi) << std::endl;
    std::cout << std::endl;

    return 0;
}
