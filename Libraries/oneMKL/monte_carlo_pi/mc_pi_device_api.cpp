//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This file contains Monte Carlo Pi number evaluation benchmark for DPC++ 
*       device interface of random number generators.
*
*******************************************************************************/

#include <iostream>
#include <numeric>
#include <vector>

#include <sycl/sycl.hpp>
#include "oneapi/mkl/rng/device.hpp"

using namespace oneapi;

// Value of Pi with many exact digits to compare with estimated value of Pi
static const auto pi = 3.1415926535897932384626433832795;

// Initialization value for random number generator
static const auto seed = 7777;

// Default Number of 2D points
static const auto n_samples = 120000000;

// Test Iterations
constexpr int num_iterations = 10;

template<typename T>
double estimate_pi(sycl::queue& q, std::size_t n_points) {
    double estimated_pi;         // Estimated value of Pi
    std::size_t n_under_curve = 0;    // Number of points fallen under the curve

    constexpr std::size_t vec_size = 2;

    std::size_t wg_size = std::min(q.get_device().get_info<sycl::info::device::max_work_group_size>(), n_points);
    std::size_t max_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    std::size_t wg_num = (n_points > wg_size * max_compute_units) ? max_compute_units : 1;

    std::size_t count_per_thread = n_points / (wg_size * wg_num);

    std::size_t* count_ptr = sycl::malloc_shared<std::size_t>(wg_num, q);

    auto event = q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(wg_size * wg_num, wg_size),
        [=](sycl::nd_item<1> item) {
            std::size_t id_global = item.get_global_linear_id();
            sycl::vec<T, vec_size> r;
            
            std::size_t count = 0;

            // Create an object of basic random numer generator (engine)
            mkl::rng::device::philox4x32x10<vec_size> engine(seed, id_global * count_per_thread * vec_size);
            // Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
            mkl::rng::device::uniform<T> distr;

            for(int i = 0; i < count_per_thread; i++) {
                // Step 1. Generate 2D point
                r = mkl::rng::device::generate(distr, engine);
                // Step 2. Increment counter if point is under curve (x ^ 2 + y ^ 2 < 1.0f)
                if(sycl::length(r) <= 1.0f) {
                    count += 1;
                }
            }
            count_ptr[item.get_group_linear_id()] = reduce_over_group(item.get_group(), count, std::plus<std::size_t>());
        });
    });

    event.wait_and_throw();

    n_under_curve = std::accumulate(count_ptr, count_ptr + wg_num, 0);

    // Step 3. Calculate approximated value of Pi
    estimated_pi = n_under_curve / ((double)n_points) * 4.0;

    sycl::free(count_ptr, q);

    return estimated_pi;

}

int main(int argc, char ** argv) {

    std::cout << std::endl;
    std::cout << "Monte Carlo pi Calculation Simulation" << std::endl;
    std::cout << "Device Api" << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    double time = 0.0;
    double estimated_pi;
    std::size_t n_points = n_samples;
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
        estimated_pi = estimate_pi<float>(q, n_points);

        for (int i = 0; i < num_iterations; i++) {
            auto start = std::chrono::steady_clock::now();
            estimated_pi = estimate_pi<float>(q, n_points);
            auto end = std::chrono::steady_clock::now();
            time +=  std::chrono::duration<double>(end - start).count();
        }
    } catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Printing results
    std::cout << "Estimated value of Pi = " << estimated_pi << std::endl;
    std::cout << "Exact value of Pi = " << pi << std::endl;
    std::cout << "Absolute error = " << fabs(pi-estimated_pi) << std::endl;
    std::cout << "Completed in " << time / num_iterations << " seconds" << std::endl;
    std::cout << std::endl;

    return 0;
}
