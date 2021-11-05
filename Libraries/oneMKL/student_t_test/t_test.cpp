//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
 *
 *  Content:
 *       This file contains Student's T-test DPC++ implementation with
 *       buffer APIs.
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
std::int32_t t_test(sycl::queue &q, sycl::buffer<RealType, 1> &r,
                    std::int64_t n, RealType expected_mean) {
  std::int32_t res = -1;
  RealType sqrt_n_observations = sycl::sqrt(static_cast<RealType>(n));

  // Create buffers to be passed inside oneMKL stats functions
  sycl::buffer<RealType, 1> mean_buf(sycl::range{1});
  sycl::buffer<RealType, 1> variance_buf(sycl::range{1});
  // Perform computations of mean and variance
  auto dataset =
      oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
          1, n, r);
  oneapi::mkl::stats::mean(q, dataset, mean_buf);
  q.wait_and_throw();
  oneapi::mkl::stats::central_moment(q, mean_buf, dataset, variance_buf);
  q.wait_and_throw();
  // Create Host accessors and check the condition
  sycl::host_accessor mean_acc(mean_buf);
  sycl::host_accessor variance_acc(variance_buf);
  if ((sycl::abs(mean_acc[0] - expected_mean) * sqrt_n_observations /
       sycl::sqrt(variance_acc[0])) < static_cast<RealType>(threshold)) {
    res = 1;
  } else {
    res = 0;
  }
  return res;
}

// T-test function with two input arrays
// Returns: -1 if something went wrong, 1 - in case of NULL hypothesis should be
// accepted, 0 - in case of NULL hypothesis should be rejected
template <typename RealType>
std::int32_t t_test(sycl::queue &q, sycl::buffer<RealType, 1> &r1,
                    std::int64_t n1, sycl::buffer<RealType, 1> &r2,
                    std::int64_t n2) {
  std::int32_t res = -1;

  // Create buffers to be passed inside oneMKL stats functions
  sycl::buffer<RealType, 1> mean1_buf(sycl::range{1});
  sycl::buffer<RealType, 1> variance1_buf(sycl::range{1});
  sycl::buffer<RealType, 1> mean2_buf(sycl::range{1});
  sycl::buffer<RealType, 1> variance2_buf(sycl::range{1});
  // Perform computations of mean and variance
  auto dataset1 =
      oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
          1, n1, r1);
  auto dataset2 =
      oneapi::mkl::stats::make_dataset<oneapi::mkl::stats::layout::row_major>(
          1, n2, r2);
  oneapi::mkl::stats::mean(q, dataset1, mean1_buf);
  q.wait_and_throw();
  oneapi::mkl::stats::central_moment(q, mean1_buf, dataset1, variance1_buf);
  oneapi::mkl::stats::mean(q, dataset2, mean2_buf);
  q.wait_and_throw();
  oneapi::mkl::stats::central_moment(q, mean2_buf, dataset2, variance2_buf);
  q.wait_and_throw();
  // Create Host accessors and check the condition
  sycl::host_accessor mean1_acc{mean1_buf};
  sycl::host_accessor variance1_acc{variance1_buf};
  sycl::host_accessor mean2_acc{mean2_buf};
  sycl::host_accessor variance2_acc{variance2_buf};
  bool almost_equal = (variance1_acc[0] < 2 * variance2_acc[0]) ||
                      (variance2_acc[0] < 2 * variance1_acc[0]);
  if (almost_equal) {
    if ((sycl::abs(mean1_acc[0] - mean2_acc[0]) /
         sycl::sqrt((static_cast<RealType>(1.0) / static_cast<RealType>(n1) +
                     static_cast<RealType>(1.0) / static_cast<RealType>(n2)) *
                    ((n1 - 1) * (n1 - 1) * variance1_acc[0] +
                     (n2 - 1) * (n2 - 1) * variance2_acc[0]) /
                    (n1 + n2 - 2))) < static_cast<RealType>(threshold)) {
      res = 1;
    } else {
      res = 0;
    }
  } else {
    if ((sycl::abs(mean1_acc[0] - mean2_acc[0]) /
         sycl::sqrt((variance1_acc[0] + variance2_acc[0]))) <
        static_cast<RealType>(threshold)) {
      res = 1;
    } else {
      res = 0;
    }
  }
  return res;
}

int main(int argc, char **argv) {
  std::cout << "\nStudent's T-test Simulation\n";
  std::cout << "Buffer Api\n";
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
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception during generation:\n"
                  << e.what() << std::endl;
      }
    }
  };

  std::int32_t res0, res1;

  try {
    // Queue constructor passed exception handler
    sycl::queue q(sycl::default_selector{}, exception_handler);
    // Prepare buffers for random output
    sycl::buffer<fp_type, 1> rng_buf0(n_points);
    sycl::buffer<fp_type, 1> rng_buf1(n_points);
    // Create engine object
    oneapi::mkl::rng::default_engine engine(q, seed);
    // Create distribution object
    oneapi::mkl::rng::gaussian<fp_type> distribution(mean, std_dev);
    // Perform generation
    oneapi::mkl::rng::generate(distribution, engine, n_points, rng_buf0);
    oneapi::mkl::rng::generate(distribution, engine, n_points, rng_buf1);
    q.wait_and_throw();
    // Launch T-test with expected mean
    res0 = t_test(q, rng_buf0, n_points, mean);
    // Launch T-test with two input arrays
    res1 = t_test(q, rng_buf0, n_points, rng_buf1, n_points);
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