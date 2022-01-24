#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "utils.hpp"
#include "metaprogramming_math.hpp"
#include "unrolled_loop.hpp"

/*
  Read "matrix_count" matrices of type TT from DDR by bursts of 
  num_elem_per_bank elements, and write the matrices to the "matrixPipe" pipe
  num_elem_per_bank by num_elem_per_bank.
  This implementation is used for matrices that have a number of rows that is a
  multiple of the number of elements per DDR burst read (num_elem_per_bank).
  Another version of this function is written below and will be selected at
  compile time if the row count is not a multiple num_elem_per_bank.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          int matrix_count,      // Number of matrices to read from the buffer
          typename matrixPipe    // Output matrix pipe
          >
void MatrixReadFromDDRToPipe(
    TT* matrix_ptr,  // Input matrix buffer
    typename std::enable_if_t<(rows % num_elem_per_bank) == 0>* = 0) {
  // Number of DDR burst reads of num_elem_per_bank elements required to read a
  // full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank;
  // Number of DDR burst reads of num_elem_per_bank to read all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrix_count;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();

  sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

  [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
  for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
    PipeTable<num_elem_per_bank, TT> ddr_read;
    // Perform the DDR burst read of num_elem_per_bank elements
    UnrolledLoop<num_elem_per_bank>([&](auto k) {
      ddr_read.elem[k] = matrix_ptr_device[(int)(li)*num_elem_per_bank + k];
    });

    matrixPipe::write(ddr_read);
  }  // end of li
}

/*
  Read "matrix_count" matrices of type TT from DDR by bursts of 
  num_elem_per_bank elements, and write the matrices to the "matrixPipe" pipe
  num_elem_per_bank by num_elem_per_bank.
  This implementation is used for matrices that have a number of rows that is
  not a multiple of the number of elements per DDR burst read
  (num_elem_per_bank).
  Another version of this function is written above and will be selected at
  compile time if the row count is a multiple num_elem_per_bank.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          int matrix_count,      // Number of matrices to read from the buffer
          typename matrixPipe    // Output matrix pipe
          >
void MatrixReadFromDDRToPipe(
    TT* matrix_ptr,  // Input matrix buffer
    typename std::enable_if_t<(rows % num_elem_per_bank) != 0>* = 0) {
  // Number of DDR burst reads of num_elem_per_bank elements required to read a
  // full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + 1;
  // Number of DDR burst reads of num_elem_per_bank to read all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrix_count;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();

  sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

  // Keep track of the current element index in the read buffer
  int load_index = 0;

  [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
  for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
    // Check if we are reading the last DDR burst of the current column
    bool last_burst_of_col = 
                            (li % kLoopIterPerColumn) == kLoopIterPerColumn - 1;

    PipeTable<num_elem_per_bank, TT> ddr_read;

#pragma unroll
    for (int k = 0; k < num_elem_per_bank; k++) {
      // Check if the current read index is beyond the end of the current
      // matrix column
      bool out_of_bounds = last_burst_of_col &&
                  ((k % num_elem_per_bank) > ((rows - 1) % num_elem_per_bank));

      // Only perform the DDR reads that are relevant (and don't access a
      // memory address that may be beyond the buffer last address)
      if (!out_of_bounds) {
        ddr_read.elem[k] = matrix_ptr_device[load_index + k];
      }
    }

    // Update the current element index in the read buffer according
    // to the read size of the current iteration
    load_index += last_burst_of_col ? rows % num_elem_per_bank :
                                      num_elem_per_bank;

    // Send the pipe read data over the pipe
    matrixPipe::write(ddr_read);

  }  // end of li
}

/*
  Read "matrix_count" matrices of type TT from a pipe, num_elem_per_bank by
  num_elem_per_bank and  write them to DDR by bursts of num_elem_per_bank 
  elements.
  This implementation is used for matrices that have a number of rows that is
  a multiple of the number of elements per DDR burst write (num_elem_per_bank).
  Another version of this function is written below and will be selected at
  compile time if the row count is not a multiple num_elem_per_bank.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          int matrix_count,      // Number of matrices to write to the buffer
          typename matrixPipe    // Input matrix
          >
void MatrixReadPipeToDDR(
    TT* matrix_ptr,
    typename std::enable_if_t<(rows % num_elem_per_bank) == 0>* = 0) {
  // Number of DDR burst of num_elem_per_bank required to write a full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank;
  // Number of DDR burst of num_elem_per_bank to write all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrix_count;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();

  sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

  [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
  for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
    PipeTable<num_elem_per_bank, TT> pipe_read = matrixPipe::read();

// Write the banks[0] to DDR
#pragma unroll
    for (int k = 0; k < num_elem_per_bank; k++) {
      *(matrix_ptr_device + static_cast<int>(li * num_elem_per_bank + k)) = 
                                                              pipe_read.elem[k];
    }

  }  // end of li
}

/*
  Read "matrix_count" matrices of type TT from a pipe, num_elem_per_bank by
  num_elem_per_bank and write them to DDR by bursts of num_elem_per_bank 
  elements.
  This implementation is used for matrices that have a number of rows that is
  not a multiple of the number of elements per DDR burst read 
  (num_elem_per_bank).
  Another version of this function is written above and will be selected at
  compile time if the row count is a multiple num_elem_per_bank.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          int matrix_count,      // Number of matrices to write to the buffer
          typename matrixPipe    // Input matrix
          >
void MatrixReadPipeToDDR(
    TT* matrix_ptr,
    typename std::enable_if_t<(rows % num_elem_per_bank) != 0>* = 0) {
  // Number of DDR burst of num_elem_per_bank required to write a full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + 1;
  // Number of DDR burst of num_elem_per_bank to write all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrix_count;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();

  sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

  // Keep track of the current element index in the write buffer
  int write_idx = 0;

  [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
  for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
    PipeTable<num_elem_per_bank, TT> pipe_read = matrixPipe::read();

    // Check if we are writing the last DDR burst of the current column
    bool last_burst_of_col = li % kLoopIterPerColumn == kLoopIterPerColumn - 1;

#pragma unroll
    for (int k = 0; k < num_elem_per_bank; k++) {
      // Check if the current write index is beyond the end of the current
      // matrix column
      bool out_of_bounds = last_burst_of_col && 
                            (k > ((rows - 1) % num_elem_per_bank));

      // Only perform the DDR writes that are relevant (and don't access a
      // memory address that may be beyond the buffer last address)
      if (!out_of_bounds) {
        matrix_ptr_device[write_idx + k] = pipe_read.elem[k];
      }
    }

    // Update the current element index in the write buffer according
    // to the write size of the current iteration
    write_idx += last_burst_of_col ? rows % num_elem_per_bank : 
                                     num_elem_per_bank;
  }  // end of li
}

/*
  Read "vector_count" vectors of type TT from a pipe, one element at the time
  and write them to DDR by bursts of num_elem_per_bank elements.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int size,              // Number of elements in the vector
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          int vector_count,      // Number of vectors to read from the buffer
          typename vectorPipe    // Input vector pipe
          >
void VectorReadPipeToDDR(TT* vector_ptr  // Output vector buffer
                        ) {
  // Number of DDR burst of num_elem_per_bank required to write one vector
  constexpr int kLoopIter = (size / num_elem_per_bank);

  sycl::device_ptr<TT> vector_ptr_device(vector_ptr);

  for(int vector_number = 0; vector_number < vector_count; vector_number++){
    [[intel::private_copies(4)]] // NO-FORMAT: Attribute
    TT r_result[size];

    for(int vector_elem = 0; vector_elem < size; vector_elem++){
      r_result[vector_elem] = vectorPipe::read();
    }

    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    for (int li = 0; li < kLoopIter; li++) {

// Write a burst of num_elem_per_bank elements to DDR
#pragma unroll
      for (int k = 0; k < num_elem_per_bank; k++) {
        *(vector_ptr_device + li * num_elem_per_bank + k + vector_number * size) =
            r_result[li * num_elem_per_bank + k];
      }
    }  // end of li
  }

}

#endif /* __MEMORY_TRANSFERS_HPP__ */