#ifndef __CALC_WEIGHTS_HPP__
#define __CALC_WEIGHTS_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "mvdr_complex.hpp"

using namespace sycl;

// SubmitCalcWeightsKernel
// Accept y vector (= Rtranspose * R * Ccomplex_conj, calculated using QRD and
// forward/backward substitution) and C vector (steering vector) and calculate
// w (weights) vector = y / (Ctranspose * y)
template <
    typename CalcWeightsKernelName,  // Name to use for the Kernel

    size_t k_num_steering_vectors,  // number of steering vectors to accept
    size_t k_num_elements,          // number of elements in each vector

    typename YVectorsInPipe,         // Receive the Y vectors.
                                     // Receive one complex float per read.
    typename SteeringVectorsInPipe,  // Receive the steering (c) vectors.
                                     // Receive one complex float per read.

    typename WeightVectorsOutPipe  // Send Weight vectors.
                                   // Send one complex float per write.
    >
event SubmitCalcWeightsKernel(queue& q) {
  // Template parameter checking
  static_assert(k_num_steering_vectors > 0,
                "k_num_steering_vectors must be greater than 0");
  static_assert(std::numeric_limits<char>::max() > k_num_steering_vectors,
                "k_num_steering_vectors must fit in a char");
  static_assert(k_num_elements >= 0, "k_num_elements must be greater than 0");
  static_assert(std::numeric_limits<short>::max() > k_num_elements,
                "k_num_elements must fit in a short");

  auto e = q.submit([&](handler& h) {
    h.single_task<CalcWeightsKernelName>([=] {
      while (1) {
        char vector_num;
        for (vector_num = 0; vector_num < (char)k_num_steering_vectors;
             vector_num++) {
          ComplexType y_vector[k_num_elements];
          ComplexType c_vector[k_num_elements];

          // read in the y and c vectors
          for (short i = 0; i < (short)k_num_elements; i++) {
            y_vector[i] = YVectorsInPipe::read();
            c_vector[i] = SteeringVectorsInPipe::read();
          }

          // calculate Ct * y

          float re = 0;
          float im = 0;
          for (short i = 0; i < (short)k_num_elements; i++) {
            auto c = c_vector[i];
            auto y = y_vector[i];

            // floating point numbers use the msb to indicate sign, so flipping
            // that bit is equivalent to multiplying by -1
            // This bit-manipulation allows the compiler to infer a hardened
            // accumulator, which allows an II of 1 for this loop
            union {
              float f;
              uint i;
            } c_imag_neg;
            c_imag_neg.f = c.imag();
            c_imag_neg.i ^= 0x80000000;

            auto im_tmp = c.real() * y.imag() + c_imag_neg.f * y.real();
            auto re_tmp = c.real() * y.real() + c.imag() * y.imag();

            re += re_tmp;
            im += im_tmp;
          }
          ComplexType ctranspose_times_y(re, im);

          // calculate 1 / norm(Ctranspose * y)
          // Ct * y is a complex number, but it's norm is a real number (float)
          // much less computationally intensive to compute reciprocal of a
          // real number
          float recip_norm_ctranspose_times_y =
              half_precision::recip(ctranspose_times_y.mag_sqr());

          // calculate the weight vectors
          // We want to mulitply each element of y by 1 / Ctranspose * y.
          // To reduce comutational complexity, we multiply by conj(Ct * y) and
          // divide by norm(Ct * y), where norm(x) = x * conj(x).
          for (short i = 0; i < (short)k_num_elements; i++) {
            auto w = y_vector[i] * ctranspose_times_y.conj() *
                     recip_norm_ctranspose_times_y;
            WeightVectorsOutPipe::write(w);
          }

        }  // end of for( vector_num... )

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitCalcWeightsKernel()

#endif  // ifndef __CALC_WEIGHTS_HPP__
