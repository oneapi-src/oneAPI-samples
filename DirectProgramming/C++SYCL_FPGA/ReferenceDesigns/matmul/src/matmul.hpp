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
class FeederA;
class FeederB;
class Matmul;
class Drain;
class APipe;
class BPipe;
class CPipe;

/**
 * Feeder A kernel.
 *
 * Reads all matrix A's tile-by-tile from global memory in burst of k_pipe_size
 * and writes result to a pipe, k_pipe_size elements at a time. Matrices must be
 * provided in column-major order.
 *
 * Repeats this operation "repetitions" times.
 *
 * Must coordinate with the other feeder kernel to support matrix tiling, by
 * repeating each tile accordingly.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int k_rows_a,     // Rows of matrix A
          int k_common,     // Columns of matrix A / rows of matrix B
          int k_cols_b,     // Columns of matrix B
          int k_tile_a,     // Tile size for matrix A
          int k_tile_b,     // Tile size for matrix B
          int k_pipe_size,  // Number of elements per DDR burst access
          typename PipeA>   // Input matrix pipe for A
void FeederAKernel(T *a, int repetitions, int num_matrices) {
  MatrixReadFromDDRToPipeA<T, k_rows_a, k_common, k_cols_b, k_tile_a, k_tile_b,
      k_pipe_size, PipeA>(a, repetitions, num_matrices);
}

/**
 * Feeder B kernel.
 *
 * Reads all matrix B's tile-by-tile from global memory in burst of k_pipe_size
 * and writes result to a pipe, k_pipe_size elements at a time. Matrices must be
 * provided in row-major order (or, equivalently, given as the transpose).
 *
 * Repeats this operation "repetitions" times.
 *
 * Must coordinate with the other feeder kernel to support matrix tiling, by
 * repeating each tile accordingly.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int k_rows_a,     // Rows of matrix A
          int k_common,     // Columns of matrix A / rows of matrix B
          int k_cols_b,     // Columns of matrix B
          int k_tile_a,     // Tile size for matrix A
          int k_tile_b,     // Tile size for matrix B
          int k_pipe_size,  // Number of elements per DDR burst access
          typename PipeB>   // Input matrix pipe for B
void FeederBKernel(T *b, int repetitions, int num_matrices) {
  MatrixReadFromDDRToPipeB<T, k_rows_a, k_common, k_cols_b, k_tile_a, k_tile_b,
      k_pipe_size, PipeB>(b, repetitions, num_matrices);
}

/**
 * Matrix multiply kernel.
 *
 * Repeatedly reads matrix tiles of A and B from input pipes and computes A * B
 * using a systolic array of PEs. Writes result matrix tile of C to output pipe.
 *
 */
template <typename T,         // Datatype of the elements of the matrix
          int k_common,       // Columns of matrix A / rows of matrix B
          int k_tile_common,  // Tile size for common side
          int k_tile_a,       // Tile size for matrix A
          int k_tile_b,       // Tile size for matrix B
          int k_pipe_size,    // Number of elements per pipe operation
          typename PipeA,     // Input matrix pipe for A
          typename PipeB,     // Input matrix pipe for B
          typename PipeC>     // Output matrix pipe for C
class MatmulKernel {
 public:
  void operator()() const {
    StreamingMatmul<T, k_common, k_tile_common, k_tile_a, k_tile_b, k_pipe_size,
        PipeA, PipeB, PipeC>();
  }
};

/**
 * Drain kernel.
 *
 * Reads all matrix C's tile-by-tile from the pipe, k_pipe_size elements at a
 * time, and writes to global memory in burst of k_pipe_size. Matrices are
 * stored in column-major order.
 *
 * Repeats this operation "repetitions" times.
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int k_rows_a,     // Rows of matrix A
          int k_cols_b,     // Columns of matrix B
          int k_tile_a,     // Tile size for matrix A
          int k_tile_b,     // Tile size for matrix B
          int k_pipe_size,  // Number of elements per DDR burst access
          typename PipeC>   // Output matrix pipe for C
void DrainKernel(T *c, int repetitions, int num_matrices) {
  MatrixReadPipeToDDR<T, k_rows_a, k_cols_b, k_tile_a, k_tile_b, k_pipe_size,
      PipeC>(c, repetitions, num_matrices);
}

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
 *  num_matrices: number of pairs of matrices to multiply
 *
 */
template <typename T,       // Datatype of the elements of the matrix
          int k_rows_a,       // Rows of matrix A
          int k_common,       // Columns of matrix A / rows of matrix B
          int k_cols_b,       // Columns of matrix B
          int k_tile_a,       // Tile size for matrix A
          int k_tile_b,       // Tile size for matrix B
          int k_tile_common>  // Tile size for k_common side
void MatmulImpl(sycl::queue &q, T *a_matrix, T *b_matrix, T *c_matrix,
                int repetitions, int num_matrices) {

  // Number of elements per DDR burst access and number of elements read/written
  // on every pipe operation (NOTE: tuned for single-precision floating-point)
  constexpr int kPipeSize = 8;

  // Matrix sizes
  constexpr size_t kMatsizeA = k_rows_a * k_common;
  constexpr size_t kMatsizeB = k_cols_b * k_common;
  constexpr size_t kMatsizeC = k_rows_a * k_cols_b;

  // Pipes to communicate the matrices between kernels
  using PipeType = fpga_tools::NTuple<T, kPipeSize>;
  using PipeA = sycl::ext::intel::pipe<APipe, PipeType, 64>;
  using PipeB = sycl::ext::intel::pipe<BPipe, PipeType, 64>;
  using PipeC = sycl::ext::intel::pipe<CPipe, PipeType, 64>;

  // Allocate FPGA DDR memory
  T *a = sycl::malloc_device<T>(kMatsizeA * num_matrices, q);
  T *b = sycl::malloc_device<T>(kMatsizeB * num_matrices, q);
  T *c = sycl::malloc_device<T>(kMatsizeC * num_matrices, q);

  q.memcpy(a, a_matrix, kMatsizeA * num_matrices * sizeof(T)).wait();
  q.memcpy(b, b_matrix, kMatsizeB * num_matrices * sizeof(T)).wait();

  // Producer kernel for matrix A
  auto feeder_a_event = q.single_task<FeederA>([=
  ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
    FeederAKernel<T, k_rows_a, k_common, k_cols_b, k_tile_a, k_tile_b,
        kPipeSize, PipeA>(a, repetitions, num_matrices);
  });

  // Producer kernel for matrix B
  q.single_task<FeederB>([=
  ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
    FeederBKernel<T, k_rows_a, k_common, k_cols_b, k_tile_a, k_tile_b,
        kPipeSize, PipeB>(b, repetitions, num_matrices);
  });

  // Matrix multiply kernel
  q.single_task<Matmul>(MatmulKernel<T, k_common, k_tile_common, k_tile_a,
      k_tile_b, kPipeSize, PipeA, PipeB, PipeC>{});

  // Consumer kernel for matrix C
  auto drain_event = q.single_task<Drain>([=
  ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
    DrainKernel<T, k_rows_a, k_cols_b, k_tile_a, k_tile_b, kPipeSize,
        PipeC>(c, repetitions, num_matrices);
  });

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

  q.memcpy(c_matrix, c, kMatsizeC * num_matrices * sizeof(T)).wait();

  // Free allocated FPGA memory
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
}

#endif /* __Matmul_HPP__ */