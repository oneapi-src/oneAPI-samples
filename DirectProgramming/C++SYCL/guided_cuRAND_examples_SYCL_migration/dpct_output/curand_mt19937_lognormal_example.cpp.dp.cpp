/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <dpct/rng_utils.hpp>

#include "curand_utils.h"

using data_type = float;

void run_on_device(const int &n, const data_type &mean, const data_type &stddev,
                   const unsigned long long &seed,
                   const curandOrdering_t &order, const curandRngType_t &rng,
                   const dpct::queue_ptr &stream,
                   std::shared_ptr<oneapi::mkl::rng::mt19937> &gen,
                   std::vector<data_type> &h_data) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  oneapi::mkl::rng::lognormal<float> distr_ct1(mean, stddev, 0.0, 1.0);
  data_type *d_data = nullptr;

  /* C data to device */
  /*
  DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((d_data = (data_type *)sycl::malloc_device(
                  sizeof(data_type) * h_data.size(), q_ct1),
              0));

  /* Create pseudo-random number generator */
  CURAND_CHECK(
      (gen = std::make_shared<oneapi::mkl::rng::mt19937>(q_ct1, seed), 0));

  /* Set cuRAND to stream */
  CURAND_CHECK(
      (gen = std::make_shared<oneapi::mkl::rng::mt19937>(*stream, seed), 0));

  /* Set ordering */
  /*
  DPCT1007:3: Migration of curandSetGeneratorOrdering is not supported.
  */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Set seed */
  /*
  DPCT1027:4: The call to curandSetPseudoRandomGeneratorSeed was replaced with 0
  because this call is redundant in SYCL.
  */
  CURAND_CHECK(0);

  /* Generate n floats on device */
  CURAND_CHECK((/*
               DPCT1003:5: Migrated API does not return error code. (*, 0) is
               inserted. You may need to rewrite this code.
               */
                oneapi::mkl::rng::generate(distr_ct1, *gen, h_data.size(),
                                           (float *)d_data),
                0));

  /* Copy data to host */
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK(
      (stream->memcpy(h_data.data(), d_data, sizeof(data_type) * h_data.size()),
       0));

  /* Sync stream */
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((stream->wait(), 0));

  /* Cleanup */
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((sycl::free(d_data, q_ct1), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void run_on_host(const int &n, const data_type &mean, const data_type &stddev,
                 const unsigned long long &seed, const curandOrdering_t &order,
                 const curandRngType_t &rng, const dpct::queue_ptr &stream,
                 std::shared_ptr<oneapi::mkl::rng::mt19937> &gen,
                 std::vector<data_type> &h_data) try {

  /* Create pseudo-random number generator */
  oneapi::mkl::rng::lognormal<float> distr_ct2(mean, stddev, 0.0, 1.0);
  CURAND_CHECK((gen = std::make_shared<oneapi::mkl::rng::mt19937>(
                    dpct::cpu_device().default_queue(), seed),
                0));

  /* Set cuRAND to stream */
  CURAND_CHECK(
      (gen = std::make_shared<oneapi::mkl::rng::mt19937>(*stream, seed), 0));

  /* Set ordering */
  /*
  DPCT1007:9: Migration of curandSetGeneratorOrdering is not supported.
  */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Set seed */
  /*
  DPCT1027:10: The call to curandSetPseudoRandomGeneratorSeed was replaced with
  0 because this call is redundant in SYCL.
  */
  CURAND_CHECK(0);

  /* Generate n floats on host */
  CURAND_CHECK((/*
               DPCT1003:11: Migrated API does not return error code. (*, 0) is
               inserted. You may need to rewrite this code.
               */
                oneapi::mkl::rng::generate(distr_ct2, *gen, h_data.size(),
                                           h_data.data()),
                0));

  /* Cleanup */
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CURAND_CHECK((gen.reset(), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();

  dpct::queue_ptr stream = &dpct::get_default_queue();
  std::shared_ptr<oneapi::mkl::rng::mt19937> gen = NULL;
  curandRngType_t rng = CURAND_RNG_PSEUDO_MT19937;
  curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;

  const int n = 12;

  const unsigned long long seed = 1234ULL;

  const data_type mean = 1.0f;
  const data_type stddev = 2.0f;

  /* Create stream */
  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  /*
  DPCT1025:14: The SYCL queue is created ignoring the flag and priority options.
  */
  CUDA_CHECK((stream = dev_ct1.create_queue(), 0));

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);

  run_on_host(n, mean, stddev, seed, order, rng, stream, gen, h_data);

  printf("Host\n");
  print_vector(h_data);
  printf("=====\n");

  run_on_device(n, mean, stddev, seed, order, rng, stream, gen, h_data);

  printf("Device\n");
  print_vector(h_data);
  printf("=====\n");

  /* Cleanup */
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((dev_ct1.destroy_queue(stream), 0));

  /*
  DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((dev_ct1.reset(), 0));

  return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
