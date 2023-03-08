#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include <chrono>
#include <iostream>

#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "matmul_common.hpp"
#include "memory_transfers.hpp"
#include "streaming_matmul.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class FeederA;
class FeederB;
class Matmul;
class Drain;
class APipe;
class BPipe;
class CPipe;

/**
 * Implementation of the matrix multiplication using multiple streaming kernels.
 * Parameterized by datatype, matrix size, and tile size. Exercises the kernels
 * by running multiple repetitions for a set of matrices.
 *
 * Function arguments:
 *  q: device queue
 *  a_matrix: input matrix pointer (given in column-major)
 *  b_matrix: input matrix pointer (given in row-major, i.e., transposed)
 *  c_matrix: output matrix pointer (will be stored in column-major)
 *  repetitions: number of repetitions of the computation to execute
 *
 */
template <typename TT,          // Datatype of the elements of the matrix
          int k_rows_a,         // Rows of matrix A
          int k_common,         // Columns of matrix A / rows of matrix B
          int k_cols_b,         // Columns of matrix B
          int k_tile_a,         // Tile size for matrix A
          int k_tile_b,         // Tile size for matrix B
          int k_num_matrices>   // Number of pairs of matrices to multiply
void MatmulImpl(sycl::queue &q, // Device queue
                std::vector<TT> &a_matrix, // Input matrix A
                std::vector<TT> &b_matrix, // Input matrix B
                std::vector<TT> &c_matrix, // Output matrix C = A * B
                int repetitions            // Number of repetitions
) {
  // Number of elements per DDR access
  // NOTE: optimized for single-precision floating-point matrices
  constexpr int kElemsPerDDRAccess = 8;

  // Matrix sizes
  constexpr int kMatsizeA = k_rows_a * k_common;
  constexpr int kMatsizeB = k_cols_b * k_common;
  constexpr int kMatsizeC = k_rows_a * k_cols_b;

  // Allocate FPGA DDR memory
#if defined(IS_BSP)
  TT *a = sycl::malloc_device<TT>(kMatsizeA * k_num_matrices, q);
  TT *b = sycl::malloc_device<TT>(kMatsizeB * k_num_matrices, q);
  TT *c = sycl::malloc_device<TT>(kMatsizeC * k_num_matrices, q);
#else
  // malloc_device are not supported when targetting an FPGA part/family
  TT *a = sycl::malloc_shared<TT>(kMatsizeA * k_num_matrices, q);
  TT *b = sycl::malloc_shared<TT>(kMatsizeB * k_num_matrices, q);
  TT *c = sycl::malloc_shared<TT>(kMatsizeC * k_num_matrices, q);
#endif

  // Copy matrices over
  q.memcpy(a, a_matrix.data(), kMatsizeA * k_num_matrices * sizeof(TT)).wait();
  q.memcpy(b, b_matrix.data(), kMatsizeB * k_num_matrices * sizeof(TT)).wait();

  using PipeDataA = FeederAData<fpga_tools::NTuple<TT, TILE_A>>;
  using PipeDataB = fpga_tools::NTuple<TT, TILE_B>;
  using PipeDataC = fpga_tools::NTuple<TT, TILE_A>;

  // Pipes to communicate the matrices between kernels
  using PipeA = sycl::ext::intel::pipe<APipe, PipeDataA, 64>;
  using PipeB = sycl::ext::intel::pipe<BPipe, PipeDataB, 64>;
  using PipeC = sycl::ext::intel::pipe<CPipe, PipeDataC, 64>;

  // Producer kernel for matrix A
  auto feeder_a_event = q.single_task<FeederA>([=
  ]() [[intel::kernel_args_restrict]] { // NO-FORMAT: Attribute
    MatrixReadFromDDRToPipeA<TT, k_rows_a, k_common, k_cols_b, k_tile_a,
                             k_tile_b, kElemsPerDDRAccess, k_num_matrices,
                             PipeA>(a, repetitions);
  });

  // Producer kernel for matrix B
  q.single_task<FeederB>([=
  ]() [[intel::kernel_args_restrict]] { // NO-FORMAT: Attribute
    MatrixReadFromDDRToPipeB<TT, k_rows_a, k_common, k_cols_b, k_tile_a,
                             k_tile_b, kElemsPerDDRAccess, k_num_matrices,
                             PipeB>(b, repetitions);
  });

  // Matrix multiply kernel
  q.single_task<Matmul>(
      fpga_linalg::StreamingMatmul<TT, k_common, k_tile_a, k_tile_b, PipeA,
                                   PipeB, PipeC>{});

  // Consumer kernel for matrix C
  auto drain_event = q.single_task<Drain>([=
  ]() [[intel::kernel_args_restrict]] { // NO-FORMAT: Attribute
    MatrixReadPipeToDDR<TT, k_rows_a, k_cols_b, k_tile_a, k_tile_b,
                        kElemsPerDDRAccess, k_num_matrices, PipeC>(c,
                                                                   repetitions);
  });

  drain_event.wait();

  // Compute the total time the execution lasted
  auto start_time = feeder_a_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = drain_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * k_num_matrices / diff * 1e-3
            << "k matrices/s" << std::endl;

  // Copy result matrix back
  q.memcpy(c_matrix.data(), c, kMatsizeC * k_num_matrices * sizeof(TT)).wait();

  // Free USM
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
}

#endif /* __MATMUL_HPP__ */