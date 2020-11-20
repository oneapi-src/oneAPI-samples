//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This file contains Multiple Simple Random Sampling without replacement
*       for DPC++ USM-based interface of random number generators
*
*******************************************************************************/

#include <iostream>

#include <CL/sycl.hpp>

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_rng_sycl.hpp"
#endif

using namespace oneapi;

// Initialization value for random number generator
static const auto seed = 777;

// Lottery default parameters
static const auto m_def = 6; // lottery first parameter
static const auto n_def = 49; // lottery second parameter
static const auto num_exp_def = 11969664; // number of lottery experiments

void lottery(sycl::queue& q, size_t m, size_t n, size_t num_exp, size_t* result_ptr) {

    // Generate (m * num_exp) random numbers
    // Generator initialization
    // Create an object of basic random numer generator (engine)
    oneapi::mkl::rng::philox4x32x10 engine(q, seed);
    // Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
    oneapi::mkl::rng::uniform distr;

    float* rng_buf = sycl::malloc_device<float>(m * num_exp, q);

    // Random number generation
    auto event = oneapi::mkl::rng::generate(distr, engine, m * num_exp, rng_buf);
    // Make sure, that generation is finished
    event.wait_and_throw();

    {

        event = q.submit([&] (sycl::handler& h) {
            sycl::accessor<size_t, 1, sycl::access::mode::read_write, sycl::access::target::local>
                local_buf(sycl::range<1>{n}, h);
            h.parallel_for(sycl::nd_range<1>(num_exp, 1),
                [=](sycl::nd_item<1> item) {
                size_t id = item.get_group(0);
                // Let buf contain natural numbers 1, 2, ..., N
                for (size_t i = 0; i < n; ++i) {
                    local_buf[i] = i + 1;
                }
                // Shuffle copied buffer
                for (size_t i = 0; i < m; ++i) {
                    // Generate random natural number j from {i,...,N-1}
                    auto j = i + (size_t)(rng_buf[id * m + i] * (float)(n - i));
                    // Swap local_buf[i] and local_buf[j]
                    auto tmp = local_buf[i];
                    local_buf[i] = local_buf[j];

                    local_buf[j] = tmp;
                }
                for (size_t i = 0; i < m; ++i) {
                    // Copy shuffled buffer
                    result_ptr[id * m + i] = local_buf[i];
                }
            });
        });
    }

    event.wait_and_throw();
}

// Prints last 3 lottery samples
void print_results(size_t* result_ptr, size_t m, size_t num_exp) {
    for (size_t i = num_exp - 3; i < num_exp; ++i) {
        std::cout << "Sample " << i << " of lottery of " << num_exp << ": ";
        for (size_t j = 0; j < m; ++j) {
            std::cout << result_ptr[i * m + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv) {

    std::cout << std::endl;
    std::cout << "Multiple Simple Random Sampling without replacement" << std::endl;
    std::cout << "Unified Shared Memory Api" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    size_t m = m_def;
    size_t n = n_def;
    size_t num_exp = num_exp_def;
    if(argc >= 4) {
        m = atol(argv[1]);
        n = atol(argv[2]);
        num_exp = atol(argv[3]);
        if(m == 0 || n == 0 || num_exp == 0 || m > n) {
            m = m_def;
            n = n_def;
            num_exp = num_exp_def;
        }
    }
    std::cout << "M = " << m << ", N = " << n << ", Number of experiments = " << num_exp << std::endl;
    // This exception handler will catch async exceptions
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

    // Pointer to result storage
    size_t* result_ptr;

    try {
        // Queue constructor passed exception handler
        sycl::queue q(sycl::default_selector{}, exception_handler);
        // Allocate memory
        result_ptr = sycl::malloc_shared<size_t>(m * num_exp, q);
        // Launch lottery for Host USM API
        lottery(q, m, n, num_exp, result_ptr);
    } catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Print output
    std::cout << "Results with Host API:" << std::endl;
    print_results(result_ptr, m, num_exp);

    return 0;
}
