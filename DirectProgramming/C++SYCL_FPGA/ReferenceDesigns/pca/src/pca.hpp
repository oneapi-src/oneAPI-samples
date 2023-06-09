#ifndef __PCA_HPP__
#define __PCA_HPP__

#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_covariance_matrix.hpp"
#include "streaming_eigen.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class InputMatrixFromDDRToLocalMem;
class CovarianceMatrixComputation;
class EigenValuesAndVectorsComputation;
class EigenVectorsFromLocalMemToDDR;
class EigenValuesFromLocalMemToDDR;
class RankDeficientFlagFromLocalMemToDDR;

class CMP;
class IMP;
class EValP;
class EVecP;
class RDFP;

/*
  Implementation of the Principal component analysis using multiple streaming
  kernels The input matrices are first transformed to their standardized
  covariance form before the Eigen values and Eigen vectors are computed. This
  function can be configured by datatype, input matrix size and works with
  square or rectangular matrices.
*/
template <unsigned k_samples_count,   // Number of samples in the input matrix
                                      // (a.k.a rows)
          unsigned k_features_count,  // Number of features in the input matrix
                                      // (a.k.a columns)
          unsigned k_raw_latency,     // RAW latency for triangular loop
                                      // optimization in the QR iteration
          int k_zero_threshold_1e,    // Threshold from which we consider a
                                    // floating point value to be 0 (e.g. -7 ->
                                    // 10e-7)
          typename T  // The datatype for the computation
          >
void PCAKernel(
    std::vector<T> &input_matrix,          // Input matrix to decompose
    std::vector<T> &eigen_values_matrix,   // Output matrix of Eigen values
    std::vector<T> &eigen_vectors_matrix,  // Output matrix of Eigen vectors
    std::vector<ac_int<1, false>>
        &rank_deficient_flag,  // Output a flag that is 1 if the input matrix
                               // is considered rank deficient
    sycl::queue &q,            // Device queue
    int matrix_count,          // Number of matrices to decompose
    int repetitions  // Number of repetitions, for performance evaluation
) {
  static_assert(k_samples_count % k_features_count == 0,
                "The feature count must be  a multiple of the samples count. "
                "This can be artificially achieved by increasing the number of "
                "samples with no data.");

  static_assert(k_samples_count > k_features_count,
                "The number of samples must be greater than the number of "
                "samples. Failing to do so, the standardized covariance matrix "
                "would be rank deficient. Processing such a matrix from the QR "
                "iteration kernel is not supported.");

  constexpr int kNumElementsPerDDRBurst = 8;
  constexpr int kInputMatrixSize = k_samples_count * k_features_count;
  constexpr int kEigenVectorsMatrixSize = k_features_count * k_features_count;
  constexpr int kEigenValuesVectorSize = k_features_count;

  using PipeType = fpga_tools::NTuple<T, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using InputMatrixPipe = sycl::ext::intel::pipe<IMP, PipeType, 3>;
  using CovarianceMatrixPipe = sycl::ext::intel::pipe<CMP, PipeType, 3>;
  using EigenValuesPipe = sycl::ext::intel::pipe<EValP, T, 3>;
  using EigenVectorsPipe = sycl::ext::intel::pipe<EVecP, PipeType, 3>;
  using RankDeficientFlagPipe =
      sycl::ext::intel::pipe<RDFP, ac_int<1, false>, 3>;

  T *input_matrix_device;
  T *eigen_vectors_device;
  T *eigen_values_device;
  ac_int<1, false> *rank_deficient_flag_device;

  if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
    std::cout << "Using device allocations" << std::endl;
    // Allocate FPGA DDR memory.
    input_matrix_device =
        sycl::malloc_device<T>(kInputMatrixSize * matrix_count, q);
    eigen_vectors_device =
        sycl::malloc_device<T>(kEigenVectorsMatrixSize * matrix_count, q);
    eigen_values_device =
        sycl::malloc_device<T>(kEigenValuesVectorSize * matrix_count, q);
    rank_deficient_flag_device =
        sycl::malloc_device<ac_int<1, false>>(matrix_count, q);
  } else if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    std::cout << "Using shared allocations" << std::endl;
    // No device allocations means that we are probably in an IP authoring flow
    input_matrix_device =
        sycl::malloc_shared<T>(kInputMatrixSize * matrix_count, q);
    eigen_vectors_device =
        sycl::malloc_shared<T>(kEigenVectorsMatrixSize * matrix_count, q);
    eigen_values_device =
        sycl::malloc_shared<T>(kEigenValuesVectorSize * matrix_count, q);
    rank_deficient_flag_device =
        sycl::malloc_shared<ac_int<1, false>>(matrix_count, q);
  } else {
    std::cerr << "USM device allocations or USM shared allocations must be "
                 "supported to run this sample."
              << std::endl;
    std::terminate();
  }

  // Check that the malloc succeeded.
  if (input_matrix_device == nullptr) {
    std::cerr << "Error when allocating the input matrix." << std::endl;
    std::terminate();
  }

  if (eigen_vectors_device == nullptr) {
    std::cerr << "Error when allocating the Eigen vectors matrix." << std::endl;
    std::terminate();
  }

  if (eigen_values_device == nullptr) {
    std::cerr << "Error when allocating the Eigen vectors matrix." << std::endl;
    std::terminate();
  }

  if (rank_deficient_flag_device == nullptr) {
    std::cerr << "Error when allocating the Eigen vectors matrix." << std::endl;
    std::terminate();
  }

  // Copy the input matrices from host DDR to FPGA DDR
  q.memcpy(input_matrix_device, input_matrix.data(),
           kInputMatrixSize * matrix_count * sizeof(T))
      .wait();

  // The covariance matrix is computed by blocks for k_features_count x
  // k_features_count Therefore we read the k_features_count x k_samples_count
  // by blocks of k_features_count x k_features_count. k_samples_count is
  // expected to be a multiple of k_features_count

  // Read the input matrix from FPGA DDR by blocks
  auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<InputMatrixFromDDRToLocalMem>([=
    ]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipeByBlocks<T, k_features_count, k_samples_count,
                                      kNumElementsPerDDRBurst, InputMatrixPipe>(
          input_matrix_device, matrix_count, repetitions);
    });
  });

  // Compute the covariance matrix
  q.single_task<CovarianceMatrixComputation>(
      fpga_linalg::StreamingCovarianceMatrix<
          T, k_samples_count, k_features_count, kNumElementsPerDDRBurst,
          InputMatrixPipe, CovarianceMatrixPipe>());

  // Compute the Eigen values and Eigen vectors
  q.single_task<EigenValuesAndVectorsComputation>(
      fpga_linalg::StreamingEigen<T, k_features_count, k_raw_latency,
                                  kNumElementsPerDDRBurst, k_zero_threshold_1e,
                                  CovarianceMatrixPipe, EigenValuesPipe,
                                  EigenVectorsPipe, RankDeficientFlagPipe>());

  // Write the Eigen values from local memory to FPGA DDR
  auto eigen_values_event = q.single_task<EigenValuesFromLocalMemToDDR>([=
  ]() [[intel::kernel_args_restrict]] {
    VectorReadPipeToDDR<T, k_features_count, EigenValuesPipe>(
        eigen_values_device, matrix_count, repetitions);
  });

  // Write the rank deficient flag from local memory to FPGA DDR
  auto rank_deficient_flag_event =
      q.single_task<RankDeficientFlagFromLocalMemToDDR>([=
  ]() [[intel::kernel_args_restrict]] {
        VectorReadPipeToDDR<ac_int<1, false>, 1, RankDeficientFlagPipe>(
            rank_deficient_flag_device, matrix_count, repetitions);
      });

  // Write the Eigen vectors from local memory to FPGA DDR
  auto eigen_vectors_event = q.single_task<EigenVectorsFromLocalMemToDDR>([=
  ]() [[intel::kernel_args_restrict]] {
    MatrixReadPipeToDDR<T, k_features_count, k_features_count,
                        kNumElementsPerDDRBurst, EigenVectorsPipe>(
        eigen_vectors_device, matrix_count, repetitions);
  });

  // Wait for the completion of the pipeline
  eigen_values_event.wait();
  eigen_vectors_event.wait();
  rank_deficient_flag_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = eigen_vectors_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;

  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * matrix_count / diff * 1e-3
            << "k matrices/s" << std::endl;

  // Copy the Eigen values and vectors from the FPGA DDR to the host memory
  q.memcpy(eigen_vectors_matrix.data(), eigen_vectors_device,
           kEigenVectorsMatrixSize * matrix_count * sizeof(T))
      .wait();
  q.memcpy(eigen_values_matrix.data(), eigen_values_device,
           kEigenValuesVectorSize * matrix_count * sizeof(T))
      .wait();
  q.memcpy(rank_deficient_flag.data(), rank_deficient_flag_device,
           matrix_count * sizeof(ac_int<1, false>))
      .wait();

  // Clean allocated FPGA memory
  free(input_matrix_device, q);
  free(eigen_vectors_device, q);
  free(eigen_values_device, q);
  free(rank_deficient_flag_device, q);
}

#endif /* __PCA_HPP__ */
