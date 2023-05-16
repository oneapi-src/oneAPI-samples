#ifndef __PCA_HPP__
#define __PCA_HPP__

#include <chrono>
#include <cstring>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "streaming_covariance_matrix.hpp"
#include "memory_transfers.hpp"
#include "streaming_eigen.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class InputMatrixFromDDRToLocalMem;
class CovarianceMatrixComputation;
class EigenValuesAndVectorsComputation;
class EigenVectorsFromLocalMemToDDR;
class EigenValuesFromLocalMemToDDR;

class CMP;
class IMP;
class EValP;
class EVecP;

/*
  Implementation of the QR decomposition using multiple streaming kernels
  Can be configured by datatype, matrix size and works with square or
  rectangular matrices.
*/
template <unsigned k_samples_count,   // Number of samples in the input matrix
          unsigned k_features_count,  // Number of features in the input matrix
          unsigned k_raw_latency,     // RAW latency for triangular loop
                                      // optimization
          bool k_use_rayleigh_shift,  // Use Rayleigh shift rather than
                                      // Wilkinson shift
          int k_zero_threshold_1e,    // Threshold from which we consider a
                                    // floating point value to be 0 (e.g. -4 ->
                                    // 10e-4)
          typename T  // The datatype for the computation
          >
void PCAsyclImpl(
    std::vector<T> &input_matrix,          // Input matrix to decompose
    std::vector<T> &eigen_values_matrix,   // Output matrix of Eigen values
    std::vector<T> &eigen_vectors_matrix,  // Output matrix of Eigen vectors
    sycl::queue &q,                        // Device queue
    int matrix_count,                      // Number of matrices to decompose
    int repetitions  // Number of repetitions, for performance evaluation
) {
  static_assert(k_samples_count % k_features_count == 0,
                "The feature count must be  a multiple of the samples count. "
                "This can be artificially achieved by increasing the number of "
                "samples with no data.");

  constexpr int kNumElementsPerDDRBurst = 8;
  constexpr int kInputMatrixSize =
      ((k_samples_count + k_features_count - 1) / k_features_count) *
      k_features_count * k_features_count;
  constexpr int kEigenVectorsMatrixSize = k_features_count * k_features_count;
  constexpr int kEigenValuesMatrixSize =
      k_features_count + 1;  // additional one for debug data

  using PipeType = fpga_tools::NTuple<T, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using InputMatrixPipe = sycl::ext::intel::pipe<IMP, PipeType, 3>;
  using CovarianceMatrixPipe = sycl::ext::intel::pipe<CMP, PipeType, 3>;
  using EigenValuesPipe = sycl::ext::intel::pipe<EValP, PipeType, 3>;
  using EigenVectorsPipe = sycl::ext::intel::pipe<EVecP, PipeType, 3>;

  // Allocate FPGA DDR memory.
  T *input_matrix_device =
      sycl::malloc_device<T>(kInputMatrixSize * matrix_count, q);
  T *eigen_vectors_device =
      sycl::malloc_device<T>(kEigenVectorsMatrixSize * matrix_count, q);
  T *eigen_values_device =
      sycl::malloc_device<T>(kEigenValuesMatrixSize * matrix_count, q);

  // Copy the input matrices from host DDR to FPGA DDR
  q.memcpy(input_matrix_device, input_matrix.data(),
           kInputMatrixSize * matrix_count * sizeof(T))
      .wait();

  // The covariance matrix is computed by blocks for k_features_count x
  // k_features_count Therefore we read the k_features_count x k_samples_count
  // by blocks of k_features_count x k_features_count. k_samples_count is
  // expected to be a multiple of k_features_count
  int matrix_blocks = matrix_count * (k_samples_count / k_features_count);

  // Read the input matrix from FPGA DDR by blocks
  auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<InputMatrixFromDDRToLocalMem>([=
    ]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<T, k_features_count, k_features_count,
                              kNumElementsPerDDRBurst, InputMatrixPipe>(
          input_matrix_device, matrix_blocks, repetitions);
    });
  });

  // Compute the covariance matrix
  q.single_task<CovarianceMatrixComputation>(
      fpga_linalg::StreamingCovarianceMatrix<
          T, k_samples_count, k_features_count, kNumElementsPerDDRBurst,
          InputMatrixPipe, CovarianceMatrixPipe>());

  // Compute the eigen values and eigen vectors
  q.single_task<EigenValuesAndVectorsComputation>(
      fpga_linalg::StreamingEigen<T, k_features_count, k_raw_latency,
                                  kNumElementsPerDDRBurst, k_zero_threshold_1e,
                                  CovarianceMatrixPipe, EigenValuesPipe,
                                  EigenVectorsPipe, k_use_rayleigh_shift>());


  // Write the Eigen values from local memory to FPGA DDR
  auto eigen_values_event = q.single_task<EigenValuesFromLocalMemToDDR>([=
  ]() [[intel::kernel_args_restrict]] {
    MatrixReadPipeToDDR<T, k_features_count + 1, 1, kNumElementsPerDDRBurst,
                        EigenValuesPipe>(eigen_values_device, matrix_count,
                                         repetitions);
  });

  // Write the Eigen vectors from local memory to FPGA DDR
  auto eigen_vectors_event = q.single_task<EigenVectorsFromLocalMemToDDR>([=
  ]() [[intel::kernel_args_restrict]] {
    MatrixReadPipeToDDR<T, k_features_count, k_features_count,
                        kNumElementsPerDDRBurst, EigenVectorsPipe>(
        eigen_vectors_device, matrix_count, repetitions);
  });

  // Wait for the completion of the pipeline
  eigen_vectors_event.wait();
  eigen_values_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = eigen_vectors_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * matrix_count / diff * 1e-3
            << "k matrices/s" << std::endl;

  // Copy the Eigen values and vectors from the FPGA DDR to the host memory
  q.memcpy(eigen_vectors_matrix.data(), eigen_vectors_device,
           kEigenVectorsMatrixSize * matrix_count * sizeof(T))
      .wait();
  q.memcpy(eigen_values_matrix.data(), eigen_values_device,
           kEigenValuesMatrixSize * matrix_count * sizeof(T))
      .wait();

  // Clean allocated FPGA memory
  free(input_matrix_device, q);
  free(eigen_values_device, q);
  free(eigen_vectors_device, q);
}

#endif /* __PCA_HPP__ */