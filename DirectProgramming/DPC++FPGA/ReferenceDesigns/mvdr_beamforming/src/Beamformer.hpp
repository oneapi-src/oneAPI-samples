#ifndef __BEAMFORMER_HPP__
#define __BEAMFORMER_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// utility classes
#include "ParallelCopyArray.hpp"
#include "Tuple.hpp"
#include "UnrolledLoop.hpp"

#include "mvdr_complex.hpp"

using namespace sycl;

// SubmitBeamformerKernel
// Accept data from an antenna array as a vector of complex numbers (Xrx) and
// apply a group of weight vectors to each sample.
template <typename BeamformerKernelName,  // Name to use for the Kernel

          size_t k_num_weight_vectors,  // number of weight vectors to apply to
                                        // each incoming sample vector
          size_t k_num_elements,        // number of elements in each vector
          size_t k_num_complex_per_xrx_read,  // Number of complex numbers
                                              // (contained in NTuple) per read
                                              // from the XrxVectorsInPipe.
          size_t k_unroll_factor,  // Factor by which to unroll the
                                   // calculation loop, so k_unroll_factor
                                   // operations will be performed in
                                   // parallel on each clock.

          typename XrxVectorsInPipe,     // Receive the Xrx vectors
                                         // Receive a NTuple containing
                                         // k_num_complex_per_xrx_read complex
                                         // floats per read
          typename WeightVectorsInPipe,  // Weight vectors
                                         // Accept one complex float per read.

          typename DataOutPipe  // For each Xrx input vector, send an
                                // output for each of the Weight vectors.
                                // Send one complex float per write.
          >
event SubmitBeamformerKernel(
    queue& q,
    short num_xrx_per_weights  // Number of xrx vectors to process with
                               // each set of Weight vectors.
) {
  // Template parameter checking
  static_assert(k_num_weight_vectors > 0,
                "k_num_weight_vectors must be greater than 0");
  static_assert(std::numeric_limits<char>::max() > k_num_weight_vectors,
                "k_num_weight_vectors must fit in a char");
  static_assert(k_num_elements >= 0, "k_num_elements must be greater than 0");
  static_assert(std::numeric_limits<short>::max() > k_num_elements,
                "k_num_elements must fit in a short");
  static_assert(k_num_complex_per_xrx_read > 0,
                "k_num_complex_per_xrx_read must be greater than 0");
  static_assert(
      k_num_elements % k_num_complex_per_xrx_read == 0,
      "k_num_elements must be evenly divisible by k_num_complex_per_xrx_read");
  static_assert(k_num_elements % k_unroll_factor == 0,
                "k_num_elements must be evenly divisible by k_unroll_factor");
  static_assert(k_unroll_factor >= k_num_complex_per_xrx_read,
                "k_unroll_factor must be >= k_num_complex_per_xrx_read");
  static_assert(
      k_unroll_factor % k_num_complex_per_xrx_read == 0,
      "k_unroll_factor must be evenly divisible by k_num_complex_per_xrx_read");

  // data coming from the Xrx pipe
  using XrxPipeType = NTuple<ComplexType, k_num_complex_per_xrx_read>;

  // this type represents the number of samples to be processed in parallel
  using CalcType = ParallelCopyArray<ComplexType, k_unroll_factor>;
  constexpr short kNumCalcTypePerVector = k_num_elements / k_unroll_factor;

  auto e = q.submit([&](handler& h) {
    h.single_task<BeamformerKernelName>([=] {
      while (1) {
        CalcType weight_vectors[k_num_weight_vectors][kNumCalcTypePerVector];

        // load the weight vectors to be used with the next set of Xrx vectors
        for (unsigned char vector_num = 0;
             vector_num < (unsigned char)k_num_weight_vectors; vector_num++) {
          // weights are loaded in reverse order
          for (short i = kNumCalcTypePerVector - 1; i >= 0; i--) {
            for (short j = (short)k_unroll_factor - 1; j >= 0; j--) {
              weight_vectors[vector_num][i][j] = WeightVectorsInPipe::read();
            }
          }
        }

        for (int xrx_vector_num = 0; xrx_vector_num < num_xrx_per_weights;
             xrx_vector_num++) {
          // force adequate private copies of this variable to not limit
          // throughput, extra private copies are cheap here
          // NO-FORMAT comments are for clang-format
          [[intel::private_copies(8)]]  // NO-FORMAT: Attribute
          CalcType xrx_vector[kNumCalcTypePerVector];
          XrxPipeType segment;

          // load a new Xrx vector
          constexpr short kReadsPerCalcType =
              k_unroll_factor / k_num_complex_per_xrx_read;
          for (short i = 0; i < kNumCalcTypePerVector * kReadsPerCalcType;
               i++) {
            segment = XrxVectorsInPipe::read();
            short index = i / kReadsPerCalcType;
            UnrolledLoop<k_num_complex_per_xrx_read>([&](auto k) {
              short subindex =
                  (i % kReadsPerCalcType) * (short)k_num_complex_per_xrx_read +
                  k;
              xrx_vector[index][subindex] = segment.template get<k>();
            });
          }

          [[intel::fpga_register]]  // NO-FORMAT: Attribute
          CalcType accum_vector;

          // don't let throughput be limited by result, adding extra private
          // copies is cheap as long as we stay below the depth of an M20K
          [[intel::private_copies(8)]]  // NO-FORMAT: Attribute
          ComplexType result[k_num_weight_vectors];

          // calculate an output vector for each weight vector
          for (unsigned char vector_num = 0;
               vector_num < (unsigned char)k_num_weight_vectors; vector_num++) {
            // zero the accumulators
            UnrolledLoop<k_unroll_factor>([&](auto i) { accum_vector[i] = 0; });

            // calculate the sum of products of the weight and xrx vectors
            // unroll by a factor of k_unroll_factor (so perform k_unroll_factor
            // operations in parallel)
            for (short i = 0; i < (kNumCalcTypePerVector); i++) {
              UnrolledLoop<k_unroll_factor>([&](auto j) {
                accum_vector[j] +=
                    xrx_vector[i][j] * weight_vectors[vector_num][i][j].conj();
              });
            }
            ComplexType accum_vector_sum = 0;
            UnrolledLoop<k_unroll_factor>(
                [&](auto i) { accum_vector_sum += accum_vector[i]; });

            result[vector_num] = accum_vector_sum;

          }  // end of for( vector_num... )

          for (unsigned char vector_num = 0;
               vector_num < (unsigned char)k_num_weight_vectors; vector_num++) {
            // send the result out
            DataOutPipe::write(result[vector_num]);
          }

        }  // end of for( xrx_vector_num... )

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitBeamformerKernel()

#endif  // ifndef __BEAMFORMER_HPP__
