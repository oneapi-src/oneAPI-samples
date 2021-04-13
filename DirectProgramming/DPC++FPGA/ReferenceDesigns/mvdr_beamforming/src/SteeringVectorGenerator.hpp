#ifndef __STEERING_VECTOR_GENERATOR_HPP__
#define __STEERING_VECTOR_GENERATOR_HPP__

#define _USE_MATH_DEFINES
#include <cmath>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "mvdr_complex.hpp"

using namespace sycl;

// SubmitSteeringVectorGeneratorKernel
// Accept the value sin(theta) for a defined number of steering vectors from a
// pipe.  Only read in new sin(theta) values when a value is available for
// every steering vector, as indicated by reads from a second pipe.
// For each sin(theta) value, generate a steering vector and write it to a
// output channel a single complex number at a time.
template <
    typename SteeringVectorGeneratorKernelName,  // Name to use for the Kernel

    size_t k_num_steering_vectors,  // number of steering vectors to generate
    size_t k_num_elements,          // number of elements in each vector

    typename SinThetaInPipe,          // sin(theta) input, updated by another
                                      // kernel with updates from the host.
                                      // Accept one float per read.
    typename SteeringVectorsOutPipe,  // Write the generated steering vectors.
                                      // Send one complex number per write.
    typename UpdateSteeringOutPipe    // Write to this pipe when a full set of
                                      // steering vectors have been sent.
                                      // Send one bool per write.
    >
event SubmitSteeringVectorGeneratorKernel(queue& q) {
  // Template parameter checking
  static_assert(k_num_steering_vectors > 0,
                "k_num_steering_vectors must be greater than 0");
  static_assert(std::numeric_limits<char>::max() > k_num_steering_vectors,
                "k_num_steering_vectors must fit in a char");
  static_assert(k_num_elements >= 0, "k_num_elements must be greater than 0");
  static_assert(std::numeric_limits<short>::max() > k_num_elements,
                "k_num_elements must fit in a short");

  auto e = q.submit([&](handler& h) {
    h.single_task<SteeringVectorGeneratorKernelName>([=] {
      while (1) {
        char vector_num;
        for (vector_num = 0; vector_num < (char)k_num_steering_vectors;
             vector_num++) {
          // fetch the value of sintheta from the pipe
          float sintheta;
          sintheta = SinThetaInPipe::read();

          // calculate the elements of the current steering vector and write
          // them to the SteeringVectorOutPipe
          // steering_vector[n] = e^-(i * pi * n * sin(theta))
          // e^-(i * t) = cos(t) - i*sin(t)
          for (short n = 0; n < (short)k_num_elements; n++) {
            float pi_n_sintheta = (float)M_PI * n * sintheta;

            ComplexType vector_element(sycl::cos(pi_n_sintheta),
                                       (-1) * sycl::sin(pi_n_sintheta));

            SteeringVectorsOutPipe::write(vector_element);
          }

        }  // end of for(...) iterating over all steering vectors

        // indicate to downstream kernels that a full set of steering vectors
        // are available
        // The fence here ensures that all writes to the SteeringVectorsOutPipe
        // above will occur before the write to the pipe below.  Without the
        // fence, the compiler could re-order these pipe writes.
        atomic_fence(sycl::ONEAPI::memory_order::acq_rel,
                     sycl::ONEAPI::memory_scope::device);
        UpdateSteeringOutPipe::write(true);

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitSteeringVectorGeneratorKernel()

#endif  // ifndef __STEERING_VECTOR_GENERATOR_HPP__
