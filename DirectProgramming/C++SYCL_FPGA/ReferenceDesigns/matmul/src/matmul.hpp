#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include <iostream>

#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "memory_transfers.hpp"
// Included from DirectProgramming/C++SYCL_FPGA/include/
#include "streaming_matmul.hpp"

#if not defined(IS_BSP)
using sycl::ext::intel::experimental::property::usm::buffer_location;
#endif

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class FeederA;
class FeederB;
class Matmul;
class Drain;
class APipe;
class BPipe;
class CPipe;
class DonePipe;

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
          int rows_a,           // Rows of matrix A
          int common,           // Columns of matrix A / rows of matrix B
          int cols_b,           // Columns of matrix B
          int tile_a,           // Tile size for matrix A
          int tile_b,           // Tile size for matrix B
          int num_matrices>     // Number of pairs of matrices to multiply
void MatmulImpl(sycl::queue &q,            // Device queue
                std::vector<TT> &a_matrix, // Input matrix A
                std::vector<TT> &b_matrix, // Input matrix B
                std::vector<TT> &c_matrix, // Output matrix C = A * B
                int repetitions            // Number of repetitions
) {
  // Number of elements per DDR access
  // NOTE: optimized for single-precision floating-point matrices
  constexpr int kElemsPerDDRAccess = 8;

  // Matrix sizes
  constexpr int kMatsizeA = rows_a * common;
  constexpr int kMatsizeB = cols_b * common;
  constexpr int kMatsizeC = rows_a * cols_b;

  // Buffer locations for mmhost interfaces
  constexpr int kBL1 = 0;
  constexpr int kBL2 = 1;
  constexpr int kBL3 = 2;

  // Allocate FPGA DDR memory
#if defined(IS_BSP)
  TT *a = sycl::malloc_device<TT>(kMatsizeA * num_matrices, q);
  TT *b = sycl::malloc_device<TT>(kMatsizeB * num_matrices, q);
  TT *c = sycl::malloc_device<TT>(kMatsizeC * num_matrices, q);
#else
  // malloc_device are not supported when targetting an FPGA part/family
  TT *a = sycl::malloc_shared<TT>(kMatsizeA * num_matrices, q,
                                  sycl::property_list{buffer_location(kBL1)});
  TT *b = sycl::malloc_shared<TT>(kMatsizeB * num_matrices, q,
                                  sycl::property_list{buffer_location(kBL2)});
  TT *c = sycl::malloc_shared<TT>(kMatsizeC * num_matrices, q,
                                  sycl::property_list{buffer_location(kBL3)});
#endif

  // Copy matrices over
  q.memcpy(a, a_matrix.data(), kMatsizeA * num_matrices * sizeof(TT)).wait();
  q.memcpy(b, b_matrix.data(), kMatsizeB * num_matrices * sizeof(TT)).wait();

  using PipeDataA = fpga_tools::NTuple<TT, tile_a>;
  using PipeDataB = fpga_tools::NTuple<TT, tile_b>;
  using PipeDataC = fpga_tools::NTuple<TT, tile_a>;

  // Pipes to communicate the matrices between kernels
  using PipeA = sycl::ext::intel::pipe<APipe, PipeDataA, 64>;
  using PipeB = sycl::ext::intel::pipe<BPipe, PipeDataB, 64>;
  using PipeC = sycl::ext::intel::pipe<CPipe, PipeDataC, 64>;
  using PipeDone = sycl::ext::intel::pipe<DonePipe, bool, 64>;

  // Producer kernel for matrix A
  auto feeder_a_event = q.single_task<FeederA>(
      MatrixReadFromDDRToPipeA<TT, kBL1, rows_a, common, cols_b, tile_a, tile_b,
                               kElemsPerDDRAccess, num_matrices, PipeA,
                               PipeDone>{a, repetitions});

  // Producer kernel for matrix B
  auto feeder_b_event = q.single_task<FeederB>(
      MatrixReadFromDDRToPipeB<TT, kBL2, rows_a, common, cols_b, tile_a, tile_b,
                               kElemsPerDDRAccess, num_matrices, PipeB>{
          b, repetitions});

  // Matrix multiply kernel
  q.single_task<Matmul>(
      fpga_linalg::StreamingMatmul<TT, common, tile_a, tile_b, PipeA, PipeB,
                                   PipeC, PipeDone>{});

  // Consumer kernel for matrix C
  auto drain_event = q.single_task<Drain>(
      MatrixReadPipeToDDR<TT, kBL3, rows_a, cols_b, tile_a, tile_b,
                          kElemsPerDDRAccess, num_matrices, PipeC>{
          c, repetitions});

  feeder_a_event.wait();
  feeder_b_event.wait();
  drain_event.wait();

  // Compute the total time the execution lasted
  auto start_time = feeder_a_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = drain_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * num_matrices / diff * 1e-3
            << "k matrices/s" << std::endl;

  // Copy result matrix back
  q.memcpy(c_matrix.data(), c, kMatsizeC * num_matrices * sizeof(TT)).wait();

  // Free USM
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
}

#endif /* __MATMUL_HPP__ */