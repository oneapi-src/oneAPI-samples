/*
 * This program uses the host CURAND API to generate 100
 * quasirandom floats.
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

using data_type = float;

void run_on_device(const int &n, const data_type &mean, const data_type &stddev,
                   const unsigned long long &offset,
                   const unsigned int &num_dimensions,
                   const curandOrdering_t &order,
                   const dpct::rng::random_engine_type &rng,
                   const dpct::queue_ptr &stream, dpct::rng::host_rng_ptr &gen,
                   std::vector<data_type> &h_data) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  data_type *d_data = nullptr;

  /* C data to device */
  CUDA_CHECK(DPCT_CHECK_ERROR(d_data = (data_type *)sycl::malloc_device(
                                  sizeof(data_type) * h_data.size(), q_ct1)));

  /* Create quasi-random number generator */
  /*
  DPCT1032:8: A different random number generator is used. You may need to
  adjust the code.
  */
  CURAND_CHECK(DPCT_CHECK_ERROR(
      gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol)));

  /* Set cuRAND to stream */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_queue(stream)));

  /* Set offset */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->skip_ahead(offset)));

  /* Set Dimension */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_dimensions(num_dimensions)));

  /* Set ordering */
  /*
  DPCT1007:9: Migration of curandSetGeneratorOrdering is not supported.
  */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Generate n floats on device */
  CURAND_CHECK(DPCT_CHECK_ERROR(
      gen->generate_gaussian(d_data, h_data.size(), mean, stddev)));

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

void run_on_host(const int &n, const data_type &mean, const data_type &stddev,
                 const unsigned long long &offset,
                 const unsigned int &num_dimensions,
                 const curandOrdering_t &order,
                 const dpct::rng::random_engine_type &rng,
                 const dpct::queue_ptr &stream, dpct::rng::host_rng_ptr &gen,
                 std::vector<data_type> &h_data) try {

  /* Create quasi-random number generator */
  /*
  DPCT1032:10: A different random number generator is used. You may need to
  adjust the code.
  */
  CURAND_CHECK(DPCT_CHECK_ERROR(
      gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol)));

  /* Set cuRAND to stream */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_queue(stream)));

  /* Set offset */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->skip_ahead(offset)));

  /* Set Dimension */
  CURAND_CHECK(DPCT_CHECK_ERROR(gen->set_dimensions(num_dimensions)));

  /* Set ordering */
  /*
  DPCT1007:11: Migration of curandSetGeneratorOrdering is not supported.
  */
  CURAND_CHECK(curandSetGeneratorOrdering(gen, order));

  /* Generate n floats on host */
  CURAND_CHECK(DPCT_CHECK_ERROR(
      gen->generate_gaussian(h_data.data(), h_data.size(), mean, stddev)));

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
  /*
  DPCT1032:12: A different random number generator is used. You may need to
  adjust the code.
  */
  dpct::rng::random_engine_type rng = dpct::rng::random_engine_type::sobol;
  curandOrdering_t order = CURAND_ORDERING_QUASI_DEFAULT;

  const int n = 12;

  const unsigned long long offset = 0ULL;
  const unsigned int num_dimensions = 1;

  const data_type mean = 1.0f;
  const data_type stddev = 2.0f;

  /* Create stream */
  /*
  DPCT1025:13: The SYCL queue is created ignoring the flag and priority options.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(stream = dev_ct1.create_queue()));

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);

  run_on_host(n, mean, stddev, offset, num_dimensions, order, rng, stream, gen,
              h_data);

  printf("Host\n");
  print_vector(h_data);
  printf("=====\n");

  run_on_device(n, mean, stddev, offset, num_dimensions, order, rng, stream,
                gen, h_data);

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
