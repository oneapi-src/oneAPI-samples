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
  d_data = (data_type *)sycl::malloc_device(
                                  sizeof(data_type) * h_data.size(), q_ct1);

  /* Create quasi-random number generator */
  gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);

  /* Set cuRAND to stream */
  gen->set_queue(stream);

  /* Set offset */
  gen->skip_ahead(offset);

  /* Set Dimension */
  gen->set_dimensions(num_dimensions);

  /* Generate n floats on device */
  gen->generate_lognormal(d_data, h_data.size(), mean, stddev);

  /* Copy data to host */
  stream->memcpy(h_data.data(), d_data, sizeof(data_type) * h_data.size());

  /* Sync stream */
  stream->wait();

  /* Cleanup */
  sycl::free(d_data, q_ct1);
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
  gen = dpct::rng::create_host_rng(dpct::rng::random_engine_type::sobol);

  /* Set cuRAND to stream */
  gen->set_queue(stream);

  /* Set offset */
  gen->skip_ahead(offset);

  /* Set Dimension */
  gen->set_dimensions(num_dimensions);

  /* Generate n floats on host */
  gen->generate_lognormal(h_data.data(), h_data.size(), mean, stddev);

  /* Cleanup */
  gen.reset();
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
  dpct::rng::random_engine_type rng = dpct::rng::random_engine_type::sobol;
  curandOrdering_t order = CURAND_ORDERING_QUASI_DEFAULT;

  const int n = 12;

  const unsigned long long offset = 0ULL;
  const unsigned int num_dimensions = 1;

  const data_type mean = 1.0f;
  const data_type stddev = 2.0f;

  /* Create stream */
  stream = dev_ct1.create_queue();

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
  dev_ct1.destroy_queue(stream);

  dev_ct1.reset();

  return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
