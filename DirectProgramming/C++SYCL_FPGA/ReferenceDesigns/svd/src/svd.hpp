#ifndef __SVD_HPP__
#define __SVD_HPP__

#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "exception_handler.hpp"
#include "memory_transfers.hpp"
#include "non_std_convariance.hpp"
#include "post_process.hpp"
#include "streaming_eigen.hpp"
#include "streaming_qrd.hpp"
#include "svd_helper.hpp"
#include "tuple.hpp"

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
class IDRMatrixFromLocalMemToDDR;
class IDSMatrixFromLocalMemToDDR;
class IDVMatrixFromLocalMemToDDR;

// pipes
class CMP;
class IMP;
class IMP2;
class ITMP;
class IDP;
class EValP;
class EVecP;
class RDFP;
class OUTMP;
class OUMP;
class ORMP;
class OSMP;
class OVMP;

template <unsigned rows,            // row size of input matrix
          unsigned cols,            // col size of input matrix
          unsigned k_fixed_iteration, // fixed number of QR iteration
          unsigned k_raw_latency,   // RAW latency for triangular loop
                                    // optimization in the QR iteration
          int k_zero_threshold_1e,  // Threshold from which we consider a
          typename T                // The datatype for the computation
          >
double singularValueDecomposition(
    std::vector<T> &a_matrix,  // input matrix A (col maj)
    std::vector<T> &u_matrix,  // left singular vectors, U (col maj)
    std::vector<T> &s_matrix,  // list of singular values, S
    std::vector<T> &v_matrix,  // right singular vectors, V (row maj)
    std::vector<ac_int<1, false>>
        &rank_deficient_flag,  // Output a flag that is 1 if the input matrix
                               // is considered rank deficient
    sycl::queue &q, int repetitions) {
  constexpr int kNumElementsPerDDRBurst = 8;            // total of 256 bit
  constexpr int kInputMatrixSize = rows * cols;         // row x col
  constexpr int kEigenVectorsMatrixSize = cols * cols;  // col x col
  constexpr int kUMatrixSize = rows * rows;
  constexpr int kRMatrixSize = rows * (rows + 1) / 2;

  // pipes
  using PipeType = fpga_tools::NTuple<T, kNumElementsPerDDRBurst>;

  using InputMatrixPipe = sycl::ext::intel::pipe<IMP, PipeType, 16>;
  using InputMatrixPipe2 = sycl::ext::intel::pipe<IMP2, PipeType, 16>;
  using CovarianceMatrixPipe = sycl::ext::intel::pipe<CMP, PipeType, 16>;
  using EigenValuesPipe = sycl::ext::intel::pipe<EValP, T, 3>;
  using EigenVectorsPipe = sycl::ext::intel::pipe<EVecP, PipeType, 3>;
  using RankDeficientFlagPipe =
      sycl::ext::intel::pipe<RDFP, ac_int<1, false>, 3>;
  using UTempMatrixPipe = sycl::ext::intel::pipe<OUTMP, PipeType, 3>;
  using UMatrixPipe = sycl::ext::intel::pipe<OUMP, PipeType, 3>;
  using RMatrixPipe =
      sycl::ext::intel::pipe<ORMP, T, kNumElementsPerDDRBurst * 4>;
  using SMatrixPipe = sycl::ext::intel::pipe<OSMP, PipeType, 3>;
  using VMatrixPipe = sycl::ext::intel::pipe<OVMP, PipeType, 3>;

  // USM allocations
  T *input_matrix_device;
  ac_int<1, false> *rank_deficient_flag_device;
  T *u_matrix_device;
  T *s_matrix_device;
  T *v_matrix_device;
  T *r_device;

  if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
    std::cout << "Using device allocations" << std::endl;
    // Allocate FPGA DDR memory.
    input_matrix_device = sycl::malloc_device<T>(kInputMatrixSize, q);
    rank_deficient_flag_device = sycl::malloc_device<ac_int<1, false>>(1, q);
    u_matrix_device = sycl::malloc_device<T>(kUMatrixSize, q);
    s_matrix_device = sycl::malloc_device<T>(kInputMatrixSize, q);
    v_matrix_device = sycl::malloc_device<T>(kEigenVectorsMatrixSize, q);
    r_device = sycl::malloc_device<T>(kRMatrixSize, q);
  } else if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    std::cout << "Using shared allocations" << std::endl;
    // No device allocations means that we are probably in a SYCL HLS flow
    input_matrix_device = sycl::malloc_shared<T>(kInputMatrixSize, q);
    rank_deficient_flag_device = sycl::malloc_shared<ac_int<1, false>>(1, q);
    u_matrix_device = sycl::malloc_shared<T>(kUMatrixSize, q);
    s_matrix_device = sycl::malloc_shared<T>(kInputMatrixSize, q);
    v_matrix_device = sycl::malloc_shared<T>(kEigenVectorsMatrixSize, q);
    r_device = sycl::malloc_shared<T>(kRMatrixSize, q);
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
  if (rank_deficient_flag_device == nullptr) {
    std::cerr << "Error when allocating the rank deficient flag." << std::endl;
    std::terminate();
  }
  if (u_matrix_device == nullptr) {
    std::cerr << "Error when allocating the U matrix." << std::endl;
    std::terminate();
  }
  if (s_matrix_device == nullptr) {
    std::cerr << "Error when allocating the S matrix." << std::endl;
    std::terminate();
  }
  if (v_matrix_device == nullptr) {
    std::cerr << "Error when allocating the V matrix." << std::endl;
    std::terminate();
  }
  if (r_device == nullptr) {
    std::cerr << "Error when allocating the R matrix." << std::endl;
    std::terminate();
  }

  // copy input matrix to device
  q.memcpy(input_matrix_device, a_matrix.data(),
           kInputMatrixSize * sizeof(float))
      .wait();

  // read input matrix from DDR
  auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<IDInputMatrixFromDDRToLocalMem>(
        [=]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRToPipeByBlocks<
              T, cols, rows, kNumElementsPerDDRBurst, InputMatrixPipe>(
              input_matrix_device, 1, repetitions);
        });
  });

  // Compute the covariance matrix
  q.single_task<IDCovarianceMatrixComputation>(
      fpga_linalg::StreamingNStdCovarianceMatrix<
          T, rows, cols, kNumElementsPerDDRBurst, InputMatrixPipe,
          CovarianceMatrixPipe>());

  // Compute the Eigen values and Eigen vectors
  q.single_task<IDEigenValuesAndVectorsComputation>(
      fpga_linalg::StreamingEigen<T, cols, k_fixed_iteration,
                                  kNumElementsPerDDRBurst, k_zero_threshold_1e,
                                  CovarianceMatrixPipe, EigenValuesPipe,
                                  EigenVectorsPipe, RankDeficientFlagPipe>());

  // read input matrix from DDR (again)
  auto ddr_write_event_post = q.submit([&](sycl::handler &h) {
    h.single_task<IDInputMatrixFromDDRToLocalMemPostProcess>(
        [=]() [[intel::kernel_args_restrict]] {
          MatrixReadFromDDRToPipeByBlocks<
              T, cols, rows, kNumElementsPerDDRBurst, InputMatrixPipe2>(
              input_matrix_device, 1, repetitions);
        });
  });

  // Post process. Using eigen values and eigen vector to produce U, S, V matrix
  q.single_task<IDUSVFromEigens>(
      fpga_svd::PostProcess<T, false, rows, cols, kNumElementsPerDDRBurst,
                            InputMatrixPipe2, EigenValuesPipe, EigenVectorsPipe,
                            UTempMatrixPipe, SMatrixPipe, VMatrixPipe>{});

  // Orthogonalizing the U matrix by running it through QRD
  q.single_task<IDOrthogonalizingU>(
      fpga_linalg::StreamingQRD<T, false, rows, rows, k_raw_latency,
                                kNumElementsPerDDRBurst, UTempMatrixPipe,
                                UMatrixPipe, RMatrixPipe>());

  // Write the rank deficient flag from local memory to FPGA DDR
  auto rank_deficient_flag_event =
      q.single_task<IDRankDeficientFlagFromLocalMemToDDR>(
          [=]() [[intel::kernel_args_restrict]] {
            VectorReadPipeToDDR<ac_int<1, false>, 1, RankDeficientFlagPipe>(
                rank_deficient_flag_device, 1, repetitions);
          });

  // collect R matrix from the orthogonalizing kernel
  q.single_task<IDRMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        for (int rep = 0; rep < repetitions; rep++) {
          for (int r_idx = 0; r_idx < kRMatrixSize; r_idx++) {
            r_device[r_idx] = RMatrixPipe::read();
          }  // end of r_idx
        }    // end of rep
      });

  // collecting U matrix from pipe into DDR
  auto u_matrix_event = q.single_task<IDUMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, rows, rows, kNumElementsPerDDRBurst,
                            UMatrixPipe>(u_matrix_device, 1, repetitions);
      });

  // collecting s matrix from pipe into DDR
  auto s_matrix_event = q.single_task<IDSMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, rows, cols, kNumElementsPerDDRBurst,
                            SMatrixPipe>(s_matrix_device, 1, repetitions);
      });

  // collecting V matrix from pipe into DDR
  auto v_matrix_event = q.single_task<IDVMatrixFromLocalMemToDDR>(
      [=]() [[intel::kernel_args_restrict]] {
        MatrixReadPipeToDDR<T, cols, cols, kNumElementsPerDDRBurst,
                            VMatrixPipe>(v_matrix_device, 1, repetitions);
      });

  // Wait for all 3 outputs are collected
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
           1 * sizeof(ac_int<1, false>))
      .wait();
  q.memcpy(u_matrix.data(), u_matrix_device, kUMatrixSize * 1 * sizeof(float))
      .wait();
  q.memcpy(s_matrix.data(), s_matrix_device,
           kInputMatrixSize * 1 * sizeof(float))
      .wait();
  q.memcpy(v_matrix.data(), v_matrix_device,
           kEigenVectorsMatrixSize * 1 * sizeof(float))
      .wait();

  // free all the usm buffers
  sycl::free(input_matrix_device, q);
  sycl::free(rank_deficient_flag_device, q);
  sycl::free(u_matrix_device, q);
  sycl::free(s_matrix_device, q);
  sycl::free(v_matrix_device, q);
  sycl::free(r_device, q);

  return diff;
}

#endif  // __SVD_HPP__