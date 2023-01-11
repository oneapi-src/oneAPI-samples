#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include <chrono>
#include <iostream>

#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "memory_transfers.hpp"
#include "streaming_matmul.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class FEEDERA;
class FEEDERB;
class MATMUL;
class DRAIN;
class PIPEA;
class PIPEB;
class PIPEC;

/**
 * Feeder A kernel.
 *
 * Reads all matrix A's tile-by-tile from global memory in burst of pipe_size
 * and writes result to a pipe, pipe_size elements at a time. Matrices must be
 * provided in column-major order.
 *
 * Repeats this operation "repetitions" times.
 *
 * Must coordinate with the other feeder kernel to support matrix tiling, by
 * repeating each tile accordingly.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int rows_A,       // Rows of matrix A
          int common,       // Columns of matrix A / rows of matrix B
          int cols_B,       // Columns of matrix B
          int tile_A,       // Tile size for matrix A
          int tile_B,       // Tile size for matrix B
          int pipe_size,    // Number of elements per DDR burst access
          typename pipe_A>  // Input matrix pipe for A
void FeederA(T *A, int repetitions, int num_matrices) {
  feeder_A<T, rows_A, common, cols_B, tile_A, tile_B, pipe_size, pipe_A>(
      A, repetitions, num_matrices);
}

/**
 * Feeder B kernel.
 *
 * Reads all matrix B's tile-by-tile from global memory in burst of pipe_size
 * and writes result to a pipe, pipe_size elements at a time. Matrices must be
 * provided in row-major order (or, equivalently, given as the transpose).
 *
 * Repeats this operation "repetitions" times.
 *
 * Must coordinate with the other feeder kernel to support matrix tiling, by
 * repeating each tile accordingly.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int rows_A,       // Rows of matrix A
          int common,       // Columns of matrix A / rows of matrix B
          int cols_B,       // Columns of matrix B
          int tile_A,       // Tile size for matrix A
          int tile_B,       // Tile size for matrix B
          int pipe_size,    // Number of elements per DDR burst access
          typename pipe_B>  // Input matrix pipe for B
void FeederB(T *B, int repetitions, int num_matrices) {
  feeder_B<T, rows_A, common, cols_B, tile_A, tile_B, pipe_size, pipe_B>(
      B, repetitions, num_matrices);
}

/**
 * Matrix multiply kernel.
 *
 * Repeatedly reads matrix tiles of A and B from input pipes and computes A * B
 * using a systolic array of PEs. Writes result matrix tile of C to output pipe.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int common,       // Columns of matrix A / rows of matrix B
          int tile_common,  // Tile size for common side
          int tile_A,       // Tile size for matrix A
          int tile_B,       // Tile size for matrix B
          int pipe_size,    // Number of elements read/write per pipe operation
          typename pipe_A,  // Input matrix pipe for A
          typename pipe_B,  // Input matrix pipe for B
          typename pipe_C>  // Output matrix pipe for C
class StreamingMatmul {
 public:
  void operator()() const {
    streaming_matmul<T, common, tile_common, tile_A, tile_B, pipe_size, pipe_A,
                     pipe_B, pipe_C>();
  }
};

/**
 * Drain kernel.
 *
 * Reads all matrix C's tile-by-tile from the pipe, pipe_size elements at a
 * time, and writes to global memory in burst of pipe_size. Matrices are stored
 * in column-major order.
 *
 * Repeats this operation "repetitions" times.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int rows_A,       // Rows of matrix A
          int cols_B,       // Columns of matrix B
          int tile_A,       // Tile size for matrix A
          int tile_B,       // Tile size for matrix B
          int pipe_size,    // Number of elements per DDR burst access
          typename pipe_C>  // Output matrix pipe for C
void Drain(T *C, int repetitions, int num_matrices) {
  drain<T, rows_A, cols_B, tile_A, tile_B, pipe_size, pipe_C>(C, repetitions,
                                                              num_matrices);
}

/**
 * Implementation of the matrix multiplication using multiple streaming kernels.
 * Parameterized by datatype, matrix size, and tile size. Exercises the kernels
 * by running multiple repetitions for a set of matrices.
 *
 * Function arguments:
 *  q: device queue
 *  A_matrix: input matrix pointer (given in column-major)
 *  B_matrix: input matrix pointer (given in row-major, i.e., transposed)
 *  C_matrix: output matrix pointer (will be stored in column-major)
 *  repetitions: number of repetitions of the computation to execute
 *  num_matrices: number of pairs of matrices to multiply
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int rows_A,       // Rows of matrix A
          int common,       // Columns of matrix A / rows of matrix B
          int cols_B,       // Columns of matrix B
          int tile_A,       // Tile size for matrix A
          int tile_B,       // Tile size for matrix B
          int tile_common>  // Tile size for common side
void MATMULImpl(sycl::queue &q, T *A_matrix, T *B_matrix, T *C_matrix,
                int repetitions, int num_matrices) {
  // Number of elements per DDR burst access and number of elements read/written
  // on every pipe operation
  constexpr int kPipeSize = 8;

  // Matrix sizes
  constexpr size_t kMatsizeA = rows_A * common;
  constexpr size_t kMatsizeB = cols_B * common;
  constexpr size_t kMatsizeC = rows_A * cols_B;

  // Pipes to communicate the matrices between kernels
  using pipe_A =
      sycl::ext::intel::pipe<PIPEA, fpga_tools::NTuple<T, kPipeSize>, 64>;
  using pipe_B =
      sycl::ext::intel::pipe<PIPEB, fpga_tools::NTuple<T, kPipeSize>, 64>;
  using pipe_C =
      sycl::ext::intel::pipe<PIPEC, fpga_tools::NTuple<T, kPipeSize>, 64>;

  // Allocate FPGA DDR memory
  T *A = sycl::malloc_device<T>(kMatsizeA * num_matrices, q);
  T *B = sycl::malloc_device<T>(kMatsizeB * num_matrices, q);
  T *C = sycl::malloc_device<T>(kMatsizeC * num_matrices, q);

  q.memcpy(A, A_matrix, kMatsizeA * num_matrices * sizeof(T)).wait();
  q.memcpy(B, B_matrix, kMatsizeB * num_matrices * sizeof(T)).wait();

  // Producer kernel for matrix B
  auto feederA_event = q.single_task<FEEDERA>([=
  ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
    FeederA<T, rows_A, common, cols_B, tile_A, tile_B, kPipeSize, pipe_A>(
        A, repetitions, num_matrices);
  });

  // Producer kernel for matrix B
  q.single_task<FEEDERB>([=
  ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
    FeederB<T, rows_A, common, cols_B, tile_A, tile_B, kPipeSize, pipe_B>(
        B, repetitions, num_matrices);
  });

  // Matrix multiply kernel
  q.single_task<MATMUL>(StreamingMatmul<T, common, tile_common, tile_A, tile_B,
                                        kPipeSize, pipe_A, pipe_B, pipe_C>{});

  // Consumer kernel for matrix C
  auto drain_event = q.single_task<DRAIN>([=
  ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
    Drain<T, rows_A, cols_B, tile_A, tile_B, kPipeSize, pipe_C>(C, repetitions,
                                                                num_matrices);
  });

  drain_event.wait();

  // Compute the total time the execution lasted
  auto start_time = feederA_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = drain_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  q.throw_asynchronous();
  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * num_matrices / diff * 1e-3
            << "k matrices/s" << std::endl;

  q.memcpy(C_matrix, C, kMatsizeC * num_matrices * sizeof(T)).wait();

  // Free allocated FPGA memory
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
}

#endif /* __MATMUL_HPP__ */