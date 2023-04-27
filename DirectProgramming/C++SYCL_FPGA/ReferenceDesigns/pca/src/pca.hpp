#ifndef __QRD_HPP__
#define __QRD_HPP__

#include <chrono>
#include <cstring>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "CovMatrix.hpp"
#include "memory_transfers.hpp"
#include "streaming_eigen.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class QRDDDRToLocalMem;
class COV;
class EIGEN;
class QRDLocalMemToDDRQ;
class QRDLocalMemToDDRR;
class CPipe;
class APipe;
class CMatrixPipe;
class RQPipe;
class QQPipe;

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
          int k_zero_threshold_1e,     // Threshold from which we consider a
                                   // floating point value to be 0 (e.g. -4 -> 10e-4)
          typename T                  // The datatype for the computation
          >
void PCAsyclImpl(
    std::vector<T> &a_matrix,    // Input matrix to decompose
    std::vector<T> &eig_matrix,  // Output matrix Q
    std::vector<T> &qq_matrix,   // Output matrix R
    sycl::queue &q,              // Device queue
    int matrix_count,            // Number of matrices to decompose
    int repetitions  // Number of repetitions, for performance evaluation
) {
  constexpr int kNumElementsPerDDRBurst = 8;
  constexpr int kAMatrixSize =
      ((k_samples_count + k_features_count - 1) / k_features_count) *
      k_features_count * k_features_count;
  constexpr int kQQMatrixSize = k_features_count * k_features_count;
  constexpr int kEigMatrixSize =
      k_features_count + 1;  // additional one for debug data

  using PipeType = fpga_tools::NTuple<T, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using CMatrixPipe = sycl::ext::intel::pipe<CPipe, PipeType, 3>;
  using EigMatrixPipe = sycl::ext::intel::pipe<RQPipe, PipeType, 3>;
  using QQMatrixPipe = sycl::ext::intel::pipe<QQPipe, PipeType, 3>;

  // Allocate FPGA DDR memory.
  T *a_device = sycl::malloc_device<T>(kAMatrixSize * matrix_count, q);
  T *qq_device = sycl::malloc_device<T>(kQQMatrixSize * matrix_count, q);
  T *eig_device = sycl::malloc_device<T>(kEigMatrixSize * matrix_count, q);

  q.memcpy(a_device, a_matrix.data(), kAMatrixSize * matrix_count * sizeof(T))
      .wait();

  int matrixBlocks = matrix_count * ((k_samples_count + k_features_count - 1) /
                                     k_features_count);

  auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<QRDDDRToLocalMem>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<T, k_features_count, k_features_count,
                              kNumElementsPerDDRBurst, AMatrixPipe>(
          a_device, matrixBlocks, repetitions);
    });
  });

  q.single_task<COV>(
      fpga_linalg::StreamingMM<T, k_features_count, k_samples_count,
                               kNumElementsPerDDRBurst, kNumElementsPerDDRBurst,
                               AMatrixPipe, CMatrixPipe>());

  q.single_task<EIGEN>(
      fpga_linalg::StreamingEigen<T, k_features_count, k_raw_latency,
                                  kNumElementsPerDDRBurst, k_zero_threshold_1e,
                                  CMatrixPipe, EigMatrixPipe, QQMatrixPipe,
                                  k_use_rayleigh_shift>());

  auto eig_event = q.single_task<QRDLocalMemToDDRR>([=
  ]() [[intel::kernel_args_restrict]] {
    MatrixReadPipeToDDR<T, k_features_count + 1, 1, kNumElementsPerDDRBurst,
                        EigMatrixPipe>(eig_device, matrix_count, repetitions);
  });

  auto qq_event =
      q.single_task<QRDLocalMemToDDRQ>([=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, k_features_count, k_features_count,
                            kNumElementsPerDDRBurst, QQMatrixPipe>(
            qq_device, matrix_count, repetitions);
      });

  qq_event.wait();
  eig_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = qq_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * matrix_count / diff * 1e-3
            << "k matrices/s" << std::endl;

  // Copy the Q and R matrices result from the FPGA DDR to the host memory
  q.memcpy(qq_matrix.data(), qq_device,
           kQQMatrixSize * matrix_count * sizeof(T))
      .wait();
  q.memcpy(eig_matrix.data(), eig_device,
           kEigMatrixSize * matrix_count * sizeof(T))
      .wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(eig_device, q);
  free(qq_device, q);
}

#endif /* __QRD_HPP__ */
