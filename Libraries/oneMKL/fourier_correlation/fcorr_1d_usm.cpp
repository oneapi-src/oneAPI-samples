//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the 1D Fourier correlation algorithm
//     using SYCL, oneMKL, and unified shared memory (USM).
//
// =============================================================

#include <mkl.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <oneapi/mkl/dft.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>
#include <oneapi/mkl/blas.hpp>

template <typename T>
static bool is_device_accessible(const T* x, const sycl::queue& Q) {
  sycl::usm::alloc alloc_type = sycl::get_pointer_type(x, Q.get_context());
  return (alloc_type == sycl::usm::alloc::shared
          || alloc_type == sycl::usm::alloc::device);
}

static sycl::event
naive_cross_correlation(sycl::queue& Q,
                        unsigned int N,
                        const float* u,
                        const float* v,
                        float* w,
                        const std::vector<sycl::event> deps = {}) {
  // u, v and w must be USM allocations of (at least) N device-accessible float
  // values (w must be writable)
  if (!is_device_accessible(u, Q) ||
      !is_device_accessible(v, Q) ||
      !is_device_accessible(w, Q)) {
    throw std::invalid_argument("Data arrays must be device-accessible");
  }
  auto ev = Q.parallel_for(sycl::range<1>{N}, deps, [=](sycl::id<1> item) {
    size_t s = item.get(0);
    w[s] = 0.0f;
    for (size_t j = 0; j < N; j++) {
      w[s] += u[j] * v[(j - s + N) % N];
    }
  });
  return ev;
}

int main(int argc, char** argv) {
  const unsigned int N = (argc > 1) ? std::stoi(argv[1]) : 32;
  // N >= 8 required for the arbitrary signals as defined herein
  if (N < 8 || N > INT_MAX)
    throw std::invalid_argument("The period of the signal, chosen as input of "
                                "the program, must be 8 or greater.");

  // Let s be an integer s.t. 0 <= s < N and let
  //       corr[s] = \sum_{j = 0}^{N-1} sig1[j] sig2[(j - s + N) mod N]
  // be the cross-correlation between two real periodic signals sig1 and sig2
  // of period N. This code shows how to calculate corr using Discrete Fourier
  // Transforms (DFTs).
  // 0 (resp. 1) is returned if naive and DFT-based calculations are (resp.
  // are not) within error tolerance of one another.
  int return_code = 0;

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector_v);
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  // Initialize signal and correlation arrays. The arrays must be large enough
  // to store the forward and backward domains' data, consisting of N real
  // values and (N/2 + 1) complex values, respectively (for the DFT-based
  // calculations). Note: 2 * (N / 2 + 1) > N for all N > 0, since
  //   2 * (N / 2 + 1) = N + 1 if N is odd
  //   2 * (N / 2 + 1) = N + 2 if N is even
  // so max(N, 2 * (N / 2 + 1)) == 2 * (N / 2 + 1)
  auto sig1 = sycl::malloc_shared<float>(2 * (N / 2 + 1), Q);
  auto sig2 = sycl::malloc_shared<float>(2 * (N / 2 + 1), Q);
  auto corr = sycl::malloc_shared<float>(2 * (N / 2 + 1), Q);
  // Array used for calculating corr without Discrete Fourier Transforms
  // for comparison purposes (calculations entirely done in forward domain):
  auto naive_corr = sycl::malloc_shared<float>(N, Q);

  // Initialize input signals with artificial "noise" data (random values of
  // magnitude much smaller than relevant signal data points)
  std::uint32_t seed = (unsigned)time(NULL);  // Get RNG seed value
  oneapi::mkl::rng::mcg31m1 engine(Q, seed);  // Initialize RNG engine
                                              // Set RNG distribution
  oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
      rng_distribution(-0.00005f, 0.00005f);

  auto evt1 = oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1);
  auto evt2 = oneapi::mkl::rng::generate(rng_distribution, engine, N, sig2);

  // Set the (relevant) signal data as shifted versions of one another
  auto evt = Q.single_task<>({evt1, evt2}, [=]() {
    sig1[N - N / 4 - 1] = 1.0f;
    sig1[N - N / 4]     = 1.0f;
    sig1[N - N / 4 + 1] = 1.0f;
    sig2[N / 4 - 1]     = 1.0f;
    sig2[N / 4]         = 1.0f;
    sig2[N / 4 + 1]     = 1.0f;
  });
  // Calculate L2 norms of both input signals before proceeding (for
  // normalization purposes and for the definition of error tolerance)
  float *norm_sig1 = sycl::malloc_shared<float>(1, Q);
  float *norm_sig2 = sycl::malloc_shared<float>(1, Q);
  evt1 = oneapi::mkl::blas::nrm2(Q, N, sig1, 1, norm_sig1, {evt});
  evt2 = oneapi::mkl::blas::nrm2(Q, N, sig2, 1, norm_sig2, {evt});
  // 1) Calculate the cross-correlation naively (for verification purposes);
  naive_cross_correlation(Q, N, sig1, sig2, naive_corr, {evt}).wait();
  // 2) Calculate the cross-correlation via Discrete Fourier Transforms (DFTs):
  //       corr = (1/N) * iDFT(DFT(sig1) * CONJ(DFT(sig2)))
  // Initialize DFT descriptor
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL> desc(N);
  // oneMKL DFT descriptors use unit scaling factors by default. Explicitly set
  // the non-default scaling factor for the backward ("inverse") DFT:
  desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0f / N);
  desc.commit(Q);
  // Compute in-place forward transforms of both signals:
  // sig1 <- DFT(sig1)
  evt1 = oneapi::mkl::dft::compute_forward(desc, sig1, {evt1});
  // sig2 <- DFT(sig2)
  evt2 = oneapi::mkl::dft::compute_forward(desc, sig2, {evt2});
  // Compute the element-wise multipication of (complex) coefficients in
  // backward domain:
  // corr <- sig1 * CONJ(sig2) [component-wise]
  evt = oneapi::mkl::vm::mulbyconj(
          Q, N / 2 + 1,
          reinterpret_cast<std::complex<float>*>(sig1),
          reinterpret_cast<std::complex<float>*>(sig2),
          reinterpret_cast<std::complex<float>*>(corr), {evt1, evt2});
  // Compute in-place (scaled) backward transform:
  // corr <- (1/N) * iDFT(corr)
  oneapi::mkl::dft::compute_backward(desc, corr, {evt}).wait();

  // Error bound for naive calculations:
  float max_err_threshold =
    2.0f * std::numeric_limits<float>::epsilon() * norm_sig1[0] * norm_sig2[0];
  // Adding an (empirical) error bound for the DFT-based calculation defined as
  //           epsilon * O(log(N)) * scaling_factor * nrm2(input data),
  // wherein (for the last DFT at play)
  // - scaling_factor = 1.0 / N;
  // - nrm2(input data) = norm_sig1[0] * norm_sig2[0] * N
  // - O(log(N)) ~ 2 * log(N) [arbitrary choice; implementation-dependent behavior]
  max_err_threshold +=
    2.0f * std::log(static_cast<float>(N))
         * std::numeric_limits<float>::epsilon() * norm_sig1[0] * norm_sig2[0];
  // Verify results by comparing DFT-based and naive calculations to each other,
  // and fetch optimal shift maximizing correlation (DFT-based calculation).
  float max_err = 0.0f;
  float max_corr = corr[0];
  int optimal_shift = 0;
  for (size_t s = 0; s < N; s++) {
    const float local_err = std::fabs(naive_corr[s] - corr[s]);
    if (local_err > max_err)
      max_err = local_err;
    if (max_err > max_err_threshold) {
      std::cerr << "An error was found when verifying the results." << std::endl;
      std::cerr << "For shift value s = " << s << ":" << std::endl;
      std::cerr << "\tNaive calculation results in " << naive_corr[s] << std::endl;
      std::cerr << "\tFourier-based calculation results in " << corr[s] << std::endl;
      std::cerr << "The error (" << max_err
                << ") exceeds the threshold value of "
                << max_err_threshold <<  std::endl;
      return_code = 1;
      break;
    }
    if (corr[s] > max_corr) {
      max_corr = corr[s];
      optimal_shift = s;
    }
  }
  // Conclude:
  if (return_code == 0) {
    // Get average and standard deviation of either signal for normalizing the
    // correlation "score"
    const float avg_sig1 = sig1[0] / N;
    const float avg_sig2 = sig2[0] / N;
    const float std_dev_sig1 =
          std::sqrt((norm_sig1[0] * norm_sig1[0] - N * avg_sig1 * avg_sig1) / N);
    const float std_dev_sig2 =
          std::sqrt((norm_sig2[0] * norm_sig2[0] - N * avg_sig2 * avg_sig2) / N);
    const float normalized_corr =
      (max_corr / N - avg_sig1 * avg_sig2) / (std_dev_sig1 * std_dev_sig2);
    std::cout << "Right-shift the second signal " << optimal_shift
              << " elements to get a maximum, normalized correlation score of "
              << normalized_corr
              << " (treating the signals as periodic)." << std::endl;
    std::cout << "Max difference between naive and Fourier-based calculations : "
              << max_err << " (verification threshold: " << max_err_threshold
              << ")." << std::endl;
  }

  // Cleanup
  sycl::free(sig1, Q);
  sycl::free(sig2, Q);
  sycl::free(corr, Q);
  sycl::free(naive_corr, Q);
  sycl::free(norm_sig1, Q);
  sycl::free(norm_sig2, Q);
  return return_code;
}
