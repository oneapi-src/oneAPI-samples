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
#include <numeric>
#include <vector>
#include <numeric>

#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

using namespace oneapi;

// Value of Pi with many exact digits to compare with estimated value of Pi
static const auto pi = 3.1415926535897932384626433832795;

// Initialization value for random number generator
static const auto seed = 7777;

// Default Number of 2D points
static const auto n_samples = 120'000'000;

double estimate_pi(sycl::queue& q, size_t n_points) {
    double estimated_pi;         // Estimated value of Pi

    // Step 1. Generate n_points * 2 random numbers
    // 1.1. Generator initialization
    // Create an object of basic random numer generator (engine)
    mkl::rng::philox4x32x10 engine(q, seed);
    // Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
    mkl::rng::uniform distr;

    float* rng_ptr = sycl::malloc_shared<float>(n_points * 2, q);

    // 1.2. Random number generation
    auto event = mkl::rng::generate(distr, engine, n_points * 2, rng_ptr);

    // Step 2. Count points under curve (x ^ 2 + y ^ 2 < 1.0f)
    constexpr size_t count_per_thread = 32;
    size_t *n_under_curve = sycl::malloc_host<size_t>(1, q); // Number of points fallen under the curve
    *n_under_curve = 0;
    auto reductor = sycl::reduction(n_under_curve, size_t(0), std::plus<size_t>{});

    q.parallel_for(sycl::range<1>(n_points / count_per_thread), event, reductor,
                   [=](sycl::item<1> item, auto& sum) {
                        sycl::vec<float, 2> r;
                        size_t count = 0;
                        for(int i = 0; i < count_per_thread; i++) {
                            r.load(i + item.get_id(0) * count_per_thread, sycl::global_ptr<float>(rng_ptr));
                            if(sycl::length(r) <= 1.0f) {
                                count++;
                            }
                        }
                        sum += count;
                   }).wait_and_throw();

    // Step 3. Calculate approximated value of Pi
    estimated_pi = *n_under_curve / ((double)n_points) * 4.0;

    sycl::free(rng_ptr, q);
    sycl::free(n_under_curve, q);

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
        sycl::queue q(sycl::default_selector_v, exception_handler);
        // Launch Pi number calculation
        estimated_pi = estimate_pi(q, n_points);
    } catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Printing results
    double abs_error = std::fabs(pi - estimated_pi);
    std::cout << "Estimated value of Pi = " << estimated_pi << std::endl;
    std::cout << "Exact value of Pi = " << pi << std::endl;
    std::cout << "Absolute error = " << abs_error << std::endl;
    std::cout << std::endl;

    if(abs_error > 1.0e-4) {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }

    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
