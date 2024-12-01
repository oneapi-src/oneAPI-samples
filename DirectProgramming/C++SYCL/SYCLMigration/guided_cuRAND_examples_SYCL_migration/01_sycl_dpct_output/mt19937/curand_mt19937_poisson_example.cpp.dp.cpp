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
#include <dpct/rng_utils.hpp>

#include "curand_utils.h"

using data_type = unsigned int;

void run_on_device(const int &n, const unsigned long long &seed,
                   const double &lambda, const curandOrdering_t &order,
                   const dpct::rng::random_engine_type &rng,
                   const dpct::queue_ptr &stream, dpct::rng::host_rng_ptr &gen,
                   std::vector<data_type> &h_data) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  data_type *d_data = nullptr;

  /* C data to device */
  CUDA_CHECK(DPCT_CHECK_ERROR(d_data = (data_type *)sycl::malloc_device(
                                  sizeof(data_type) * h_data.size(), q_ct1)));

  /* Create pseudo-random number generator */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen = dpct::rng::create_host_rng(
                                    dpct::rng::random_engine_type::mt19937)));

  /* Set cuRAND to stream */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_queue(stream)));

  /* Set ordering */
  /*
  DPCT1007:2: Migration of curandSetGeneratorOrdering is not supported.
  */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Set seed */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_seed(seed)));

  /* Generate n floats on device */
  CURAND_CHECK(
      DPCT_CHECK_ERROR(gen->generate_poisson(d_data, h_data.size(), lambda)));

  /* Copy data to host */
  CUDA_CHECK(DPCT_CHECK_ERROR(stream->memcpy(
      h_data.data(), d_data, sizeof(data_type) * h_data.size())));

  /* Sync stream */
  CUDA_CHECK(DPCT_CHECK_ERROR(stream->wait()));

  /* Cleanup */
  CUDA_CHECK(DPCT_CHECK_ERROR(sycl::free(d_data, q_ct1)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void run_on_host(const int &n, const unsigned long long &seed,
                 const double &lambda, const curandOrdering_t &order,
                 const dpct::rng::random_engine_type &rng,
                 const dpct::queue_ptr &stream, dpct::rng::host_rng_ptr &gen,
                 std::vector<data_type> &h_data) try {

  /* Create pseudo-random number generator */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen = dpct::rng::create_host_rng(
                                    dpct::rng::random_engine_type::mt19937)));

  /* Set cuRAND to stream */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_queue(stream)));

  /* Set ordering */
  /*
  DPCT1007:3: Migration of curandSetGeneratorOrdering is not supported.
  */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Set seed */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_seed(seed)));

  /* Generate n floats on host */
  CURAND_CHECK(DPCT_CHECK_ERROR(
      gen->generate_poisson(h_data.data(), h_data.size(), lambda)));

  /* Cleanup */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen.reset()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();

  dpct::queue_ptr stream = &dpct::get_default_queue();
  dpct::rng::host_rng_ptr gen = NULL;
  dpct::rng::random_engine_type rng = dpct::rng::random_engine_type::mt19937;
  curandOrdering_t order = CURAND_ORDERING_PSEUDO_DEFAULT;

  const int n = 12;
  const double lambda = 10.0;

  const unsigned long long seed = 1234ULL;

  /* Create stream */
  /*
  DPCT1025:4: The SYCL queue is created ignoring the flag and priority options.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(stream = dev_ct1.create_queue()));

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);

  run_on_host(n, seed, lambda, order, rng, stream, gen, h_data);

  printf("Host\n");
  print_vector(h_data);
  printf("=====\n");

  run_on_device(n, seed, lambda, order, rng, stream, gen, h_data);

  printf("Device\n");
  print_vector(h_data);
  printf("=====\n");

  /* Cleanup */
  CUDA_CHECK(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(stream)));

  CUDA_CHECK(DPCT_CHECK_ERROR(dev_ct1.reset()));

  return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
