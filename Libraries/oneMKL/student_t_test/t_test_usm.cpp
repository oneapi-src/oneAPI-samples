//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
 *
 *  Content:
 *       This file contains Student's T-test DPC++ implementation with
 *       USM APIs.
 *
 *******************************************************************************/

#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "oneapi/mkl.hpp"

using fp_type = float;
// Initialization value for random number generator
static const auto seed = 7777;
// Quantity of samples to check using Students' T-test
static const auto n_samples = 1000000;
// Expected mean value of random samples
static const auto expected_mean = 0.0f;
// Expected standard deviation of random samples
static const auto expected_std_dev = 1.0f;
// T-test threshold which corresponds to 5% significance level and infinite
// degrees of freedom
static const auto threshold = 1.95996f;

// T-test function with expected mean
// Returns: -1 if something went wrong, 1 - in case of NULL hypothesis should be
// accepted, 0 - in case of NULL hypothesis should be rejected
template <typename RealType>
std::int32_t t_test(sycl::queue& q, RealType* r, std::int64_t n,
                    RealType expected_mean) {
  std::int32_t res = -1;
  RealType sqrt_n_observations = sycl::sqrt(static_cast<RealType>(n));

  // Allocate memory to be passed inside oneMKL stats functions
  RealType* mean = sycl::malloc_shared<RealType>(1, q);
  RealType* variance = sycl::malloc_shared<RealType>(1, q);
  // Perform computations of mean and variance
  auto dataset =
      oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
          1, n, r);
  oneapi::mkl::stats::mean(q, dataset, mean);
  q.wait_and_throw();
  oneapi::mkl::stats::central_moment(q, mean, dataset, variance);
  q.wait_and_throw();
  // Check the condition
  if ((sycl::abs(mean[0] - expected_mean) * sqrt_n_observations /
       sycl::sqrt(variance[0])) < static_cast<RealType>(threshold)) {
    res = 1;
  } else {
    res = 0;
  }
  // Free allocated memory
  sycl::free(mean, q);
  sycl::free(variance, q);
  return res;
}

// T-test function with two input arrays
// Returns: -1 if something went wrong, 1 - in case of NULL hypothesis should be
// accepted, 0 - in case of NULL hypothesis should be rejected
template <typename RealType>
std::int32_t t_test(sycl::queue& q, RealType* r1, std::int64_t n1,
                    RealType* r2, std::int64_t n2) {
  std::int32_t res = -1;
  // Allocate memory to be passed inside oneMKL stats functions
  RealType* mean1 = sycl::malloc_shared<RealType>(1, q);
  RealType* variance1 = sycl::malloc_shared<RealType>(1, q);
  RealType* mean2 = sycl::malloc_shared<RealType>(1, q);
  RealType* variance2 = sycl::malloc_shared<RealType>(1, q);
  // Perform computations of mean and variance
  auto dataset1 =
      oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
          1, n1, r1);
  auto dataset2 =
      oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
          1, n2, r2);
  oneapi::mkl::stats::mean(q, dataset1, mean1);
  q.wait_and_throw();
  oneapi::mkl::stats::central_moment(q, mean1, dataset1, variance1);
  oneapi::mkl::stats::mean(q, dataset2, mean2);
  q.wait_and_throw();
  oneapi::mkl::stats::central_moment(q, mean2, dataset2, variance2);
  q.wait_and_throw();
  // Check the condition
  bool almost_equal =
      (variance1[0] < 2 * variance2[0]) || (variance2[0] < 2 * variance1[0]);
  if (almost_equal) {
    if ((sycl::abs(mean1[0] - mean2[0]) /
         sycl::sqrt((static_cast<RealType>(1.0) / static_cast<RealType>(n1) +
                     static_cast<RealType>(1.0) / static_cast<RealType>(n2)) *
                    ((n1 - 1) * (n1 - 1) * variance1[0] +
                     (n2 - 1) * (n2 - 1) * variance2[0]) /
                    (n1 + n2 - 2))) < static_cast<RealType>(threshold)) {
      res = 1;
    } else {
      res = 0;
    }
  } else {
    if ((sycl::abs(mean1[0] - mean2[0]) /
         sycl::sqrt((variance1[0] + variance2[0]))) <
        static_cast<RealType>(threshold)) {
      res = 1;
    } else {
      res = 0;
    }
  }
  // Free allocated memory
  sycl::free(mean1, q);
  sycl::free(variance1, q);
  sycl::free(mean2, q);
  sycl::free(variance2, q);
  return res;
}

int main(int argc, char** argv) {
  std::cout << "\nStudent's T-test Simulation\n";
  std::cout << "Unified Shared Memory Api\n";
  std::cout << "-------------------------------------\n";

  size_t n_points = n_samples;
  fp_type mean = expected_mean;
  fp_type std_dev = expected_std_dev;

  if (argc >= 2) {
    n_points = std::atol(argv[1]);
    if (n_points == 0) {
      n_points = n_samples;
    }
  }

  if (argc >= 3) {
    mean = std::atof(argv[2]);
    if (std::isnan(mean) || std::isinf(mean)) {
      mean = expected_mean;
    }
  }

  if (argc >= 4) {
    std_dev = std::atof(argv[3]);
    if (std_dev <= static_cast<fp_type>(0.0f)) {
      std_dev = expected_std_dev;
    }
  }

  std::cout << "Number of random samples = " << n_points
            << " with mean = " << mean << ", std_dev = " << std_dev << "\n";

  // This exception handler with catch async exceptions
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception during generation:\n"
                  << e.what() << std::endl;
      }
    }
  };

  std::int32_t res0, res1;

  try {
    // Queue constructor passed exception handler
    sycl::queue q(sycl::default_selector{}, exception_handler);
    // Allocate memory for random output
    fp_type* rng_arr0 = sycl::malloc_shared<fp_type>(n_points, q);
    fp_type* rng_arr1 = sycl::malloc_shared<fp_type>(n_points, q);
    // Create engine object
    oneapi::mkl::rng::default_engine engine(q, seed);
    // Create distribution object
    oneapi::mkl::rng::gaussian<fp_type> distribution(mean, std_dev);
    // Perform generation
    oneapi::mkl::rng::generate(distribution, engine, n_points, rng_arr0);
    oneapi::mkl::rng::generate(distribution, engine, n_points, rng_arr1);
    q.wait_and_throw();
    // Launch T-test with expected mean
    res0 = t_test(q, rng_arr0, n_points, mean);
    // Launch T-test with two input arrays
    res1 = t_test(q, rng_arr0, n_points, rng_arr1, n_points);
    // Free allocated memory
    sycl::free(rng_arr0, q);
    sycl::free(rng_arr1, q);
  } catch (...) {
    // Some other exception detected
    std::cout << "Failure\n";
    std::terminate();
  }

  // Printing results
  std::cout << "T-test result with expected mean: " << res0 << "\n";
  std::cout << "T-test result with two input arrays: " << res1 << "\n\n";

  return 0;
}