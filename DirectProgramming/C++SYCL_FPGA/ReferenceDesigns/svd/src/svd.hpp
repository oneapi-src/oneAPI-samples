#ifndef __SVD_HPP__
#define __SVD_HPP__

#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>
#include "tuple.hpp"

// Headers shared with other oneAPI FPGA samples
#include "exception_handler.hpp"
#include "streaming_eigen.hpp"
#include "streaming_qrd.hpp"

// Headers specific to SVD
#include "memory_transfers.hpp"
#include "non_std_covariance.hpp"
#include "post_process.hpp"


// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)

// kernels
class IDInputMatrixFromDDRToLocalMem;
class IDInputTransMatrixFromDDRToLocalMem;
class IDCovarianceMatrixComputation;
class IDEigenValuesAndVectorsComputation;
class IDInputMatrixFromDDRToLocalMemPostProcess;
class IDUSVFromEigens;
class IDOrthogonalizingU;
class IDRankDeficientFlagFromLocalMemToDDR;
class IDUMatrixFromLocalMemToDDR;
class IDRMatrixPipeTerminate;
class IDSMatrixFromLocalMemToDDR;
class IDVMatrixFromLocalMemToDDR;

// pipes
class IDCovarianceMatrixPipe;
class IDInputMatrixPipe;
class IDInputMatrixPipe2;
class IDEigenValuesPipe;
class IDEigenVectorsPipe;
class IDRankDeficientFlagPipe;
class IDUTempMatrixPipe;
class IDUMatrixPipe;
class IDRMatrixPipe;
class IDSMatrixPipe;
class IDVMatrixPipe;

template <unsigned rows,            // row size of input matrix
          unsigned cols,            // col size of input matrix
          unsigned fixed_iteration, // fixed number of QR iteration
          unsigned raw_latency,   // RAW latency for triangular loop
                                    // optimization in the QR iteration
          int zero_threshold_1e,  // Threshold from which we consider a
          typename T                // The datatype for the computation
          >
double SingularValueDecomposition(
    std::vector<T> &a_matrix,  // input matrix A (col maj)
    std::vector<T> &u_matrix,  // left singular vectors, U (col maj)
    std::vector<T> &s_matrix,  // list of singular values, S
    std::vector<T> &v_matrix,  // right singular vectors, V (row maj)
    std::vector<ac_int<1, false>>
        &rank_deficient_flag,  // Output a flag that is 1 if the input matrix
                               // is considered rank deficient
    sycl::queue &q, int matrix_count=1, int repetitions=1) {
  constexpr int kNumElementsPerDDRBurst = 8;            // total of 256 bit
  constexpr int kInputMatrixSize = rows * cols;         // row x col
  constexpr int kEigenVectorsMatrixSize = cols * cols;  // col x col
  constexpr int kUMatrixSize = rows * rows;
  constexpr int kRMatrixSize = rows * (rows + 1) / 2;

  // pipes
  using PipeType = fpga_tools::NTuple<T, kNumElementsPerDDRBurst>;

  using InputMatrixPipe = sycl::ext::intel::pipe<IDInputMatrixPipe, PipeType, 3>;
  using InputMatrixPipe2 = sycl::ext::intel::pipe<IDInputMatrixPipe2, PipeType, 3>;
  using CovarianceMatrixPipe = sycl::ext::intel::pipe<IDCovarianceMatrixPipe, PipeType, 3>;
  using EigenValuesPipe = sycl::ext::intel::pipe<IDEigenValuesPipe, T, 3>;
  using EigenVectorsPipe = sycl::ext::intel::pipe<IDEigenVectorsPipe, PipeType, 3>;
  using RankDeficientFlagPipe =
      sycl::ext::intel::pipe<IDRankDeficientFlagPipe, ac_int<1, false>, 3>;
  using UTempMatrixPipe = sycl::ext::intel::pipe<IDUTempMatrixPipe, PipeType, 3>;
  using UMatrixPipe = sycl::ext::intel::pipe<IDUMatrixPipe, PipeType, 3>;
  using RMatrixPipe =
      sycl::ext::intel::pipe<IDRMatrixPipe, T, kNumElementsPerDDRBurst * 4>;
  using SMatrixPipe = sycl::ext::intel::pipe<IDSMatrixPipe, PipeType, 3>;
  using VMatrixPipe = sycl::ext::intel::pipe<IDVMatrixPipe, PipeType, 3>;

  // USM allocations
  T *input_matrix_device;
  ac_int<1, false> *rank_deficient_flag_device;
  T *u_matrix_device;
  T *s_matrix_device;
  T *v_matrix_device;

  if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
    std::cout << "Using device allocations" << std::endl;
    // Allocate FPGA DDR memory.
    input_matrix_device = sycl::malloc_device<T>(kInputMatrixSize * matrix_count, q);
    rank_deficient_flag_device = sycl::malloc_device<ac_int<1, false>>(1 * matrix_count, q);
    u_matrix_device = sycl::malloc_device<T>(kUMatrixSize * matrix_count, q);
    s_matrix_device = sycl::malloc_device<T>(kInputMatrixSize * matrix_count, q);
    v_matrix_device = sycl::malloc_device<T>(kEigenVectorsMatrixSize * matrix_count, q);
  } else if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    std::cout << "Using shared allocations" << std::endl;
    // No device allocation support, using shared allocation instead 
    input_matrix_device = sycl::malloc_shared<T>(kInputMatrixSize * matrix_count, q);
    rank_deficient_flag_device = sycl::malloc_shared<ac_int<1, false>>(1 * matrix_count, q);
    u_matrix_device = sycl::malloc_shared<T>(kUMatrixSize * matrix_count, q);
    s_matrix_device = sycl::malloc_shared<T>(kInputMatrixSize * matrix_count, q);
    v_matrix_device = sycl::malloc_shared<T>(kEigenVectorsMatrixSize * matrix_count, q);
  } else {
    std::cerr << "USM device allocations or USM shared allocations must be "
                 "supported to run this sample."
              << std::endl;
    std::terminate();
  }

  // Check that the malloc succeeded.
  if (nullptr == input_matrix_device) {
    std::cerr << "Error when allocating the input matrix." << std::endl;
    std::terminate();
  }
  if (nullptr == rank_deficient_flag_device) {
    std::cerr << "Error when allocating the rank deficient flag." << std::endl;
    std::terminate();
  }
  if (nullptr == u_matrix_device) {
    std::cerr << "Error when allocating the U matrix." << std::endl;
    std::terminate();
  }
  if (nullptr == s_matrix_device) {
    std::cerr << "Error when allocating the S matrix." << std::endl;
    std::terminate();
  }
  if (nullptr == v_matrix_device) {
    std::cerr << "Error when allocating the V matrix." << std::endl;
    std::terminate();
  }

  // copy input matrix to memory that the device can access
  q.memcpy(input_matrix_device, a_matrix.data(),
           matrix_count * kInputMatrixSize * sizeof(float))
      .wait();

  // read input matrix from DDR
  sycl::event ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<IDInputMatrixFromDDRToLocalMem>(
        [=]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRTo2PipesByBlocks<
              T, cols, rows, kNumElementsPerDDRBurst, InputMatrixPipe, InputMatrixPipe2>(
              input_matrix_device, matrix_count, repetitions);
        });
  });

  // Compute the covariance matrix
  q.single_task<IDCovarianceMatrixComputation>(
      fpga_linalg::StreamingNStdCovarianceMatrix<
          T, rows, cols, kNumElementsPerDDRBurst, InputMatrixPipe,
          CovarianceMatrixPipe>());

  // Compute the Eigen values and Eigen vectors
  q.single_task<IDEigenValuesAndVectorsComputation>(
      fpga_linalg::StreamingEigen<T, cols, fixed_iteration,
                                  kNumElementsPerDDRBurst, zero_threshold_1e,
                                  CovarianceMatrixPipe, EigenValuesPipe,
                                  EigenVectorsPipe, RankDeficientFlagPipe>());

  // Post process. Using eigen values and eigen vector to produce U, S, V matrix
  q.single_task<IDUSVFromEigens>(
      USVFromEigens <T, false, rows, cols, kNumElementsPerDDRBurst,
                            InputMatrixPipe2, EigenValuesPipe, EigenVectorsPipe,
                            UTempMatrixPipe, SMatrixPipe, VMatrixPipe>{});

  // Orthogonalizing the U matrix by running it through QRD
  q.single_task<IDOrthogonalizingU>(
      fpga_linalg::StreamingQRD<T, false, rows, rows, raw_latency,
                                kNumElementsPerDDRBurst, UTempMatrixPipe,
                                UMatrixPipe, RMatrixPipe>());

  // Write the rank deficient flag from local memory to FPGA DDR
  sycl::event rank_deficient_flag_event =
      q.single_task<IDRankDeficientFlagFromLocalMemToDDR>(
          [=]() [[intel::kernel_args_restrict]] {
            VectorReadPipeToDDR<ac_int<1, false>, 1, RankDeficientFlagPipe>(
                rank_deficient_flag_device, matrix_count, repetitions);
          });

  // terminating R matrix pipe from the orthogonalizing kernel
  // This kernel only read from the pipe 
  // (and throw away the data since its not useful here)
  q.single_task<IDRMatrixPipeTerminate>(
      [=]() [[intel::kernel_args_restrict]] {
        T r_sum = 0.0;
        for (int mat_idx = 0; mat_idx < matrix_count; mat_idx ++) {
          for (int rep = 0; rep < repetitions; rep++) {
            for (int r_idx = 0; r_idx < kRMatrixSize; r_idx++) {
              r_sum += RMatrixPipe::read();
            }  // end of r_idx
          }    // end of rep
        }
      });

  // collecting U matrix from pipe into DDR
  sycl::event u_matrix_event = q.single_task<IDUMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, rows, rows, kNumElementsPerDDRBurst,
                            UMatrixPipe>(u_matrix_device, matrix_count, repetitions);
      });

  // collecting s matrix from pipe into DDR
  sycl::event s_matrix_event = q.single_task<IDSMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, rows, cols, kNumElementsPerDDRBurst,
                            SMatrixPipe>(s_matrix_device, matrix_count, repetitions);
      });

  // collecting V matrix from pipe into DDR
  sycl::event v_matrix_event = q.single_task<IDVMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, cols, cols, kNumElementsPerDDRBurst,
                            VMatrixPipe>(v_matrix_device, matrix_count, repetitions);
      });

  // Wait for output memory access kernels to finish
  u_matrix_event.wait();
  s_matrix_event.wait();
  v_matrix_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = u_matrix_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;

  // Copy outputs from the FPGA DDR

  q.memcpy(rank_deficient_flag.data(), rank_deficient_flag_device,
           matrix_count * sizeof(ac_int<1, false>))
      .wait();
  q.memcpy(u_matrix.data(), u_matrix_device, kUMatrixSize * matrix_count * sizeof(float))
      .wait();
  q.memcpy(s_matrix.data(), s_matrix_device,
           kInputMatrixSize * matrix_count * sizeof(float))
      .wait();
  q.memcpy(v_matrix.data(), v_matrix_device,
           kEigenVectorsMatrixSize * matrix_count * sizeof(float))
      .wait();

  // free all the usm buffers
  sycl::free(input_matrix_device, q);
  sycl::free(rank_deficient_flag_device, q);
  sycl::free(u_matrix_device, q);
  sycl::free(s_matrix_device, q);
  sycl::free(v_matrix_device, q);

  return diff;
}

#endif  // __SVD_HPP__