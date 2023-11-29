/*
    Copyright 2023 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <oneapi/dpl/dynamic_selection>

namespace ds = oneapi::dpl::experimental;

constexpr int num_warmup_iterations = 10;

void verify(float nstream_time, size_t length, int iterations, float scalar, std::vector<float> &A)
{
  float ar(0);
  float br(2);
  float cr(2);

  for(int i = 0; i <= iterations; i++) {
    ar += br + scalar * cr;
  }

  ar *= length;
  float asum(0);

  for (size_t i = 0; i < length; i++) {
    asum += abs(A[i]);
  }

  float epsilon(1.e-8);

  if (abs(ar-asum)/asum > epsilon) {
    std::cout << "Failed Validation on output array" << std::endl
              << std::setprecision(16)
              << "Expected checksum: " << ar << std::endl
              << "Observed checksum: " << asum << std::endl
              << "ERROR: solution did not validate." << std::endl;
    exit(1);
  } else {
    float avgtime = nstream_time/(iterations-2);
    float nbytes = 4.0 * length * sizeof(float);
    std::cout << "\n Rate: larger better     (MB/s): " << (1.e-6*nbytes)/(1.e-9*avgtime)
              << "\n Avg time: lower better  (ns):   " << avgtime  << std::endl;
  }
}

template<typename T>
void invokeDS(int length, std::vector<sycl::queue> &universe){
  int iterations{1000};
  float scalar{3};

  std::vector<float> A(length, 0.0);
  std::vector<float> B(length, 2.0);
  std::vector<float> C(length, 2.0);

  const size_t bytes = length * sizeof(float);
  T policy(universe);

  // Start the timer
  auto begin = std::chrono::high_resolution_clock::now();

  for (int i = 0; i <= iterations; ++i) {
    if (i == num_warmup_iterations) {
      begin = std::chrono::high_resolution_clock::now();
    }

    ds::submit_and_wait(policy, [&](sycl::queue e)
    {
      if (i < num_warmup_iterations) {
        std::cout << e.get_device().get_info<sycl::info::device::name>() <<std::endl;
      }
      float *d_A = sycl::malloc_device<float>(length, e);
      float *d_B = sycl::malloc_device<float>(length, e);
      float *d_C = sycl::malloc_device<float>(length, e);

      e.memcpy(d_A, A.data(), bytes).wait();
      e.memcpy(d_B, B.data(), bytes).wait();
      e.memcpy(d_C, C.data(), bytes).wait();

      auto x = e.submit([&](sycl::handler &h) {
        h.parallel_for(length, [=](auto I) {
          d_A[I] += d_B[I] + scalar * d_C[I];
        });
      });

      x.wait();
      e.memcpy(A.data(), d_A, bytes).wait();

      sycl::free(d_C, e);
      sycl::free(d_B, e);
      sycl::free(d_A, e);

      //
      // Note: Dynamic Device Select requires a return event such as the following.
      //
      return x;
    });
  }

  auto end = std::chrono::high_resolution_clock::now();
  // Calculate time elapsed
  auto nstream_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
  verify(nstream_time, length, iterations, scalar, A);
}

void invokeSYCL(int length)
{
  int iterations{40};
  float scalar{3};

  std::vector<float> A(length, 0.0);
  std::vector<float> B(length, 2.0);
  std::vector<float> C(length, 2.0);

  const size_t bytes = length * sizeof(float);
  sycl::queue u(sycl::cpu_selector_v);

  // Start the timer
  auto begin = std::chrono::high_resolution_clock::now();

  for (int i = 0; i <= iterations; ++i) {
    float *d_A = sycl::malloc_device<float>(length, u);
    float *d_B = sycl::malloc_device<float>(length, u);
    float *d_C = sycl::malloc_device<float>(length, u);

    u.memcpy(d_A, A.data(), bytes).wait();
    u.memcpy(d_B, B.data(), bytes).wait();
    u.memcpy(d_C, C.data(), bytes).wait();

    auto x = u.submit([&](sycl::handler& h) {
      h.parallel_for<class vector_add>(length, [=](auto I) {
        d_A[I] += d_B[I] + scalar * d_C[I];
      });
    });

    x.wait();
    u.memcpy(A.data(), d_A, bytes).wait();

    sycl::free(d_C, u);
    sycl::free(d_B, u);
    sycl::free(d_A, u);
  }

  // End the timer
  auto end = std::chrono::high_resolution_clock::now();
  // Calculate time elapsed
  auto nstream_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
  verify(nstream_time, length, iterations, scalar, A);
}

void printUsage()
{
  std::cout << "Application requires arguments. Usage: 2_nstream_policies <vector length> <policy>" << std::endl
            << "Vector length must be an integer between 1 and 50000." << std::endl
            << "Policy:" << std::endl
            << "1 - Fixed Resource Policy (CPU)" << std::endl
            << "2 - Fixed Resource Policy (GPU)" << std::endl
            << "3 - Round Robin Policy" << std::endl
            << "4 - Dynamic Load Policy" << std::endl
            << "5 - Auto Tune Policy" << std::endl;
}

int main(int argc, char * argv[])
{
  int length{50}, policy{2};
  std::vector<sycl::queue> universe;

  if (argc >= 3) {
    int conv = std::atoi(argv[1]);

    if (conv <= 0 || conv > 50000) {
      std::cout << "Vector length must be an integer between 1 and 50000." << std::endl;
      return -1;
    }
    else {
      length = conv;
    }

    policy = std::atoi(argv[2]);
  }
  else {
    printUsage();
    return -1;
  }

  auto prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
  switch (policy) {
    case 1:
      std::cout << "Using Static Policy (CPU) to iterate on CPU device with vector length: " << length << std::endl;
      // Add CPUs to the universe of devices.
      universe.push_back(sycl::queue{sycl::cpu_selector_v});
      invokeDS<ds::fixed_resource_policy<ds::sycl_backend>>(length, universe);
      break;
    case 2:
      std::cout << "Using Static Policy (GPU) to iterate on GPU device with vector length: " << length << std::endl;
      // Add GPUs to the universe of devices.
      universe.push_back(sycl::queue{sycl::gpu_selector_v});
      invokeDS<ds::fixed_resource_policy<ds::sycl_backend>>(length, universe);
      break;
    case 3:
      std::cout << "Using Round Robin Policy to iterate across available devices with vector length: " << length << std::endl;
      // Add CPUs and GPUs to the universe of devices.
      universe.push_back(sycl::queue{sycl::cpu_selector_v});
      universe.push_back(sycl::queue{sycl::gpu_selector_v});
      invokeDS<ds::round_robin_policy<ds::sycl_backend>>(length, universe);
      break;
    case 4:
      std::cout << "Using Dynamic Load Policy to pick least loaded device: " << length << std::endl;
      universe.push_back(sycl::queue{sycl::gpu_selector_v});
      universe.push_back(sycl::queue{sycl::cpu_selector_v});
      invokeDS<ds::dynamic_load_policy<ds::sycl_backend>>(length, universe);
      break;
    case 5:
      std::cout << "Using Auto-Tune to pick the best performing device: " << length << std::endl;
      universe.push_back(sycl::queue{sycl::gpu_selector_v, prop_list});
      universe.push_back(sycl::queue{sycl::cpu_selector_v, prop_list});
      invokeDS<ds::auto_tune_policy<ds::sycl_backend>>(length, universe);
      break;
    default:
      std::cout << "Invalid policy." << std::endl;
      printUsage();
      return -1;
      break;
  }

  return 0;
}
