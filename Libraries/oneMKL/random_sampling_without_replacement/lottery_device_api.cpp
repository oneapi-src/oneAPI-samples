//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This file contains Multiple Simple Random Sampling without replacement
*       for DPC++ device interface of random number generators
*
*******************************************************************************/

#include <iostream>

#include <CL/sycl.hpp>

#if __has_include("oneapi/mkl/rng/device.hpp")
#include "oneapi/mkl/rng/device.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_rng_sycl_device.hpp"
#endif

using namespace oneapi;

// Initialization value for random number generator
static const auto seed = 777;

// Lottery default parameters
static const auto m_def = 6; // lottery first parameter
static const auto n_def = 49; // lottery second parameter
static const auto num_exp_def = 11969664; // number of lottery experiments

void lottery_device_api(sycl::queue& q, size_t m, size_t n, size_t num_exp, std::vector<size_t>& result_vec) {

    {
        sycl::buffer<size_t, 1> result_buf(result_vec.data(), result_vec.size());

        q.submit([&](sycl::handler& h) {
            auto res_acc = result_buf.template get_access<sycl::access::mode::write>(h);
            sycl::accessor<size_t, 1, sycl::access::mode::read_write, sycl::access::target::local>
                local_buf(sycl::range<1>{n}, h);
            h.parallel_for(sycl::nd_range<1>(num_exp, 1),
                [=](sycl::nd_item<1> item) {
                size_t id = item.get_group(0);
                // Let buf contain natural numbers 1, 2, ..., N
                for (size_t i = 0; i < n; ++i) {
                    local_buf[i] = i + 1;
                }
                // Create an object of basic random numer generator (engine)
                oneapi::mkl::rng::device::philox4x32x10 engine(seed, id * m);
                // Create an object of distribution (by default float, a = 0.0f, b = 1.0f)
                oneapi::mkl::rng::device::uniform distr;

                for (size_t i = 0; i < m; ++i) {
                    auto res = oneapi::mkl::rng::device::generate(distr, engine);
                    // Generate random natural number j from {i,...,N-1}
                    auto j = i + (size_t)(res * (float)(n - i));
                    // Swap local_buf[i] and local_buf[j]
                    auto tmp = local_buf[i];
                    local_buf[i] = local_buf[j];

                    local_buf[j] = tmp;
                }
                for (size_t i = 0; i < m; ++i) {
                    // Copy shuffled buffer
                    res_acc[id * m + i] = local_buf[i];
                }
            });
        });
    }
}

// Prints last 3 lottery samples
void print_results(std::vector<size_t>& res, size_t m) {
    for (size_t i = res.size() / m - 3; i < res.size() / m; ++i) {
        std::cout << "Sample " << i << " of lottery of " << res.size() / m << ": ";
        for (size_t j = 0; j < m; ++j) {
            std::cout << res[i * m + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv) {

    std::cout << std::endl;
    std::cout << "Multiple Simple Random Sampling without replacement" << std::endl;
    std::cout << "Device Api" << std::endl;
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

    // Result storage
    std::vector<size_t> result_vec(m * num_exp);

    try {
        // Queue constructor passed exception handler
        sycl::queue q(sycl::default_selector{}, exception_handler);
        // Launch lottery for device API
        lottery_device_api(q, m, n, num_exp, result_vec);
    } catch (...) {
        // Some other exception detected
        std::cout << "Failure" << std::endl;
        std::terminate();
    }

    // Print output
    print_results(result_vec, m);

    return 0;
}
