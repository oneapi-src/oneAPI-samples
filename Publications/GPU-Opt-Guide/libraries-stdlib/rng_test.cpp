//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// Snippet begin
#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/random>
#include <oneapi/mkl/rng.hpp>

int main(int argc, char **argv) {
  unsigned int N = (argc == 1) ? 20 : std::stoi(argv[1]);
  if (N < 20)
    N = 20;

  // Generate sequences of random numbers between [0.0, 1.0] using oneDPL and
  // oneMKL
  sycl::queue Q(sycl::gpu_selector_v);
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  auto test1 = sycl::malloc_shared<float>(N, Q.get_device(), Q.get_context());
  auto test2 = sycl::malloc_shared<float>(N, Q.get_device(), Q.get_context());

  std::uint32_t seed = (unsigned)time(NULL); // Get RNG seed value

  // oneDPL random number generator on GPU device
  clock_t start_time = clock(); // Start timer

  Q.parallel_for(N, [=](auto idx) {
     oneapi::dpl::minstd_rand rng_engine(seed, idx); // Initialize RNG engine
     oneapi::dpl::uniform_real_distribution<float>
         rng_distribution;                      // Set RNG distribution
     test1[idx] = rng_distribution(rng_engine); // Generate RNG sequence
   }).wait();

  clock_t end_time = clock(); // Stop timer
  std::cout << "oneDPL took " << float(end_time - start_time) / CLOCKS_PER_SEC
            << " seconds to generate " << N
            << " uniformly distributed random numbers." << std::endl;

  // oneMKL random number generator on GPU device
  start_time = clock(); // Start timer

  oneapi::mkl::rng::mcg31m1 engine(
      Q, seed); // Initialize RNG engine, set RNG distribution
  oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
      rng_distribution(0.0, 1.0);
  oneapi::mkl::rng::generate(rng_distribution, engine, N, test2)
      .wait(); // Generate RNG sequence

  end_time = clock(); // Stop timer
  std::cout << "oneMKL took " << float(end_time - start_time) / CLOCKS_PER_SEC
            << " seconds to generate " << N
            << " uniformly distributed random numbers." << std::endl;

  // Show first ten random numbers from each method
  std::cout << std::endl
            << "oneDPL"
            << "\t"
            << "oneMKL" << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << test1[i] << " " << test2[i] << std::endl;

  // Show last ten random numbers from each method
  std::cout << "..." << std::endl;
  for (size_t i = N - 10; i < N; i++)
    std::cout << test1[i] << " " << test2[i] << std::endl;

  // Cleanup
  sycl::free(test1, Q.get_context());
  sycl::free(test2, Q.get_context());
}
// Snippet end
