#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "tuple.hpp"
#include "constexpr_math.hpp"
#include "unrolled_loop.hpp"

/*
  Read matrix_count matrices of type TT from DDR by bursts of num_elem_per_bank
  elements, and write the matrices to the "MatrixPipe" pipe num_elem_per_bank by
  num_elem_per_bank elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          typename MatrixPipe    // Output matrix pipe
          >
void MatrixReadFromDDRToPipe(
    TT* matrix_ptr,  // Input matrix pointer
    int matrix_count,// Number of matrix to read from DDR
    int repetitions  // Number of time to write the same matrix to the pipe
    ) {

  // We may perform an incomplete memory read if the number of elements per row
  // is not a multiple of the DDR burst size
  constexpr bool kIncompleteBurst = rows%num_elem_per_bank != 0;
  constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
  // Number of DDR burst reads of num_elem_per_bank elements required to read a
  // full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + kExtraIteration;
  // Number of DDR burst reads of num_elem_per_bank to read all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();
  // Size of a full matrix
  constexpr int kMatrixSize = rows * columns;

  sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

  // Repeatedly read matrix_count matrices from DDR and sends them to the pipe
  for (int repetition = 0; repetition < repetitions; repetition++){

    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++){

      // Keep track of the current element index in the matrix
      // Only useful in the case of kIncompleteBurst
      int load_index = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        bool last_burst_of_col;
        if constexpr (kIncompleteBurst){
          // Check if we are reading the last DDR burst of the current column
          last_burst_of_col = (li % kLoopIterPerColumn)
                              == kLoopIterPerColumn - 1;
        }

        fpga_tools::NTuple<TT, num_elem_per_bank> ddr_read;

        // Perform the DDR burst read of num_elem_per_bank elements
        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {

          if constexpr (kIncompleteBurst){
            // Check if the current read index is beyond the end of the current
            // matrix column
            bool out_of_bounds = last_burst_of_col &&
                  ((k % num_elem_per_bank) > ((rows - 1) % num_elem_per_bank));

            // Only perform the DDR reads that are relevant (and don't access a
            // memory address that may be beyond the matrix last address)
            if (!out_of_bounds) {
              ddr_read.template get<k>() = matrix_ptr_device
                                  [matrix_index * kMatrixSize + load_index + k];
            }
          }
          else{
            ddr_read.template get<k>() = matrix_ptr_device
                [matrix_index * kMatrixSize + (int)(li)*num_elem_per_bank + k];

          }
        });

        if constexpr (kIncompleteBurst){
          // Update the current element index in the input matrix according
          // to the read size of the current iteration
          load_index += last_burst_of_col ? rows % num_elem_per_bank :
                                            num_elem_per_bank;
        }

        MatrixPipe::write(ddr_read);
      }  // end of li

    } // end of matrix_index
  } // end of repetition
}

/*
  Write matrix_count matrices of type TT from a pipe, num_elem_per_bank by
  num_elem_per_bank and write them to DDR by bursts of num_elem_per_bank
  elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          typename MatrixPipe    // Input matrix
          >
void MatrixReadPipeToDDR(
    TT* matrix_ptr,  // Output matrix pointer
    int matrix_count,// Number of matrix to write to DDR
    int repetitions  // Number of time to read the same matrix to the pipe
    ) {

  // We may perform an incomplete memory write if the number of elements per row
  // is not a multiple of the DDR burst size
  constexpr bool kIncompleteBurst = rows%num_elem_per_bank != 0;
  constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
  // Number of DDR burst of num_elem_per_bank required to write a full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + kExtraIteration;
  // Number of DDR burst of num_elem_per_bank to write all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();
  // Size of a full matrix
  constexpr int kMatrixSize = rows * columns;

  sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

  // Repeatedly read matrix_count matrices from the pipe and write them to DDR
  for (int repetition = 0; repetition < repetitions; repetition++){

    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++){
      // Keep track of the current element index in the output matrix
      // Only useful in the case of kIncompleteBurst
      int write_idx = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<TT, num_elem_per_bank> pipe_read =
                                                            MatrixPipe::read();

        bool last_burst_of_col;
        if constexpr (kIncompleteBurst){
          // Check if we are writing the last DDR burst of the current column
          last_burst_of_col =
                            (li % kLoopIterPerColumn) == kLoopIterPerColumn - 1;
        }

        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
          if constexpr (kIncompleteBurst){
            // Check if the current write index is beyond the end of the current
            // matrix column
            bool out_of_bounds = last_burst_of_col &&
                                  (k > ((rows - 1) % num_elem_per_bank));

            // Only perform the DDR writes that are relevant (and don't access a
            // memory address that may be beyond the buffer last address)
            if (!out_of_bounds) {
              matrix_ptr_device[matrix_index * kMatrixSize + write_idx + k] =
                                                    pipe_read.template get<k>();
            }
          }
          else{
            matrix_ptr_device[matrix_index * kMatrixSize
              + int(li) * num_elem_per_bank + k] = pipe_read.template get<k>();
          }

        });

        if constexpr (kIncompleteBurst){
          // Update the current element index in the write buffer according
          // to the write size of the current iteration
          write_idx += last_burst_of_col ? rows % num_elem_per_bank :
                                           num_elem_per_bank;
        }
      }  // end of li
    } // end of matrix_index
  } // end of repetition
}

#endif /* __MEMORY_TRANSFERS_HPP__ */