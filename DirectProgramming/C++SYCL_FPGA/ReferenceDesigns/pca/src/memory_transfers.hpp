#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

/*
  Read matrix_count matrices of type TT from DDR by bursts of num_elem_per_bank
  elements, and write the matrices to the "MatrixPipe" pipe num_elem_per_bank by
  num_elem_per_bank elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int num_elem_per_bank,  // Number of TT elements per DDR burst access
          typename MatrixPipe     // Output matrix pipe
          >
void MatrixReadFromDDRToPipeByBlocks(
    TT* matrix_ptr,    // Input matrix pointer
    int matrix_count,  // Number of matrix to read from DDR
    int repetitions    // Number of time to write the same matrix to the pipe
) {
  static_assert(columns % rows == 0,
                "In order to be able to send the matrix by blocs, the number "
                "of rows must be a multiple of the number of columns");

  constexpr int kMatrixSize = rows * columns;
  constexpr int kBlockCount = columns / rows;

  // Repeatedly read matrix_count matrices from DDR and sends them to the pipe
  for (int repetition = 0; repetition < repetitions; repetition++) {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      for (int block_index = 0; block_index < kBlockCount; block_index++) {
        for (int row = 0; row < rows; row++) {
          for (int column = 0; column < rows; column += num_elem_per_bank) {
            // Read num_elem_per_bank elements per burst
            fpga_tools::NTuple<TT, num_elem_per_bank> ddr_read;
            fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
              if (column + k < rows) {
                ddr_read.template get<k>() =
                    matrix_ptr[matrix_index * kMatrixSize + block_index * rows +
                               row * columns + column + k];
              }
            });

            MatrixPipe::write(ddr_read);
          }  // end of column
        }    // end of row
      }      // end of block_index
    }        // end of matrix_index
  }          // end of repetition
}

/*
  Write matrix_count matrices of type TT from a pipe, num_elem_per_bank by
  num_elem_per_bank and write them to DDR by bursts of num_elem_per_bank
  elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int num_elem_per_bank,  // Number of TT elements per DDR burst access
          typename MatrixPipe     // Input matrix
          >
void MatrixReadPipeToDDR(
    TT* matrix_ptr,    // Output matrix pointer
    int matrix_count,  // Number of matrix to write to DDR
    int repetitions    // Number of time to read the same matrix to the pipe
) {
  // We may perform an incomplete memory write if the number of elements per row
  // is not a multiple of the DDR burst size
  constexpr bool kIncompleteBurst = rows % num_elem_per_bank != 0;
  constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
  // Number of DDR burst of num_elem_per_bank required to write a full column
  constexpr int kLoopIterationsPerColumn =
      rows / num_elem_per_bank + kExtraIteration;
  // Number of DDR burst of num_elem_per_bank to write all the matrices
  constexpr int kLoopIterations = kLoopIterationsPerColumn * columns;
  // Size of a full matrix
  constexpr int kMatrixSize = rows * columns;

  // Repeatedly read matrix_count matrices from the pipe and write them to DDR
  for (int repetition = 0; repetition < repetitions; repetition++) {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      // Keep track of the current element index in the output matrix
      // Only useful in the case of kIncompleteBurst
      int write_idx = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep]]                   // NO-FORMAT: Attribute
      for (int i = 0; i < kLoopIterations; i++) {
        fpga_tools::NTuple<TT, num_elem_per_bank> pipe_read =
            MatrixPipe::read();

        bool last_burst_of_col;
        if constexpr (kIncompleteBurst) {
          // Check if we are writing the last DDR burst of the current column
          last_burst_of_col =
              (i % kLoopIterationsPerColumn) == kLoopIterationsPerColumn - 1;
        }

        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
          if constexpr (kIncompleteBurst) {
            // Check if the current write index is beyond the end of the current
            // matrix column
            bool out_of_bounds =
                last_burst_of_col && (k > ((rows - 1) % num_elem_per_bank));

            // Only perform the DDR writes that are relevant (and don't access a
            // memory address that may be beyond the buffer last address)
            if (!out_of_bounds) {
              matrix_ptr[matrix_index * kMatrixSize + write_idx + k] =
                  pipe_read.template get<k>();
            }
          } else {
            matrix_ptr[matrix_index * kMatrixSize + i * num_elem_per_bank + k] =
                pipe_read.template get<k>();
          }
        });

        if constexpr (kIncompleteBurst) {
          // Update the current element index in the write buffer according
          // to the write size of the current iteration
          write_idx +=
              last_burst_of_col ? rows % num_elem_per_bank : num_elem_per_bank;
        }
      }  // end of i
    }    // end of matrix_index
  }      // end of repetition
}

/*
  Write vector_count vectors of type TT from a pipe, one element at the time and
  write them to DDR. Repeat this operations "repetitions" times.
*/
template <typename TT,         // Datatype of the elements of the matrix
          int size,            // Number of rows of the matrix
          typename VectorPipe  // Input matrix
          >
void VectorReadPipeToDDR(
    TT* vector_ptr,    // Output matrix pointer
    int vector_count,  // Number of vectors to write to DDR
    int repetitions    // Number of time to read the same matrix to the pipe
) {
#if defined(IS_BSP)
  // When targeting a BSP, we instruct the compiler that this pointer
  // lives on the device.
  // Knowing this, the compiler won't generate hardware to
  // potentially get data from the host.
  sycl::device_ptr<TT> vector_ptr_located(vector_ptr);
#else
  // Device pointers are not supported when targeting an FPGA
  // family/part
  TT* vector_ptr_located(vector_ptr);
#endif

  // Repeat vector_count complete R matrix pipe reads
  // for as many repetitions as needed
  for (int repetition = 0; repetition < repetitions; repetition++) {
    [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
    for (int vector_index = 0; vector_index < vector_count; vector_index++) {
      for (int k = 0; k < size; k++) {
        vector_ptr_located[vector_index * size + k] = VectorPipe::read();
      }  // end of k
    }    // end of vector_index
  }      // end of repetition
}

#endif /* __MEMORY_TRANSFERS_HPP__ */