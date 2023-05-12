#ifndef __STREAMING_COVARIANCE_MATRIX_HPP__
#define __STREAMING_COVARIANCE_MATRIX_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {
// This functor computes the rows x rows covariance matrix of a rows x columns
// input matrix

// The general algorithm is the following:
// 1. compute the mean of each row
// 2. go through each element of the matrix and subtract the mean of its row to
// it
// 3. compute the variance of each row as the sum of all its elements squared
// divided by the size of the row
// 4. Divide each row elements by there associated variance
// 5. compute the covariance matrix C = modified input matrix times the
// transpose of modified input matrix

// In cases where the number of columns is very large, following these steps one
// by one would make us go through the matrix multiple times, and perform a
// large matrix multiplication. Therefore, this code implements a block
// covariance matrix computation.

// A preliminary step of computing the standardized covariance matrix is to
// compute the standardized input matrix STD
// To compute STD, we need to first compute the variance of each row VAR[i].
// VAR[i] = SUM_{j=0, N-1}{(A[i][j] - mean[i])^2}/N
// STD is then defined as:
// STD[i][j] = (A[i][j] - mean[i]) / sqrt(VAR[i])

// The standardized covariance matrix is then defined as:
// STDCOV[i][j] = SUM_{k=0, N-1}{STD[i][k] * STD[j][k]}

// This computation would be hard to implement in hardware as is, so we can
// rewrite these equations in a way that fits FPGAs better. First, lets focus on
// the variance computation: VAR[i] = SUM_{j=0, N-1}{(A[i][j] - mean[i])^2}/N We
// can expand the square operation: VAR[i] = SUM_{j=0, N-1}{A[i][j]^2
// -2*A[i][j]*mean[i] + mean[i]^2)}/N We can then break this single sum into 3:
// VAR[i] = (SUM_{j=0, N-1}{A[i][j]^2} -2*mean[i]*SUM_{j=0, N-1}{A[i][j]} +
// SUM_{j=0, N-1}{mean[i]^2)})/N Note that SUM_{j=0, N-1}{mean[i]^2)} is
// actually equal to N*(mean[i])^2 And that SUM_{j=0, N-1}{A[i][j]} is actually
// equal to mean[i]*N By substituting these into the formula, we get: VAR[i] =
// (SUM_{j=0, N-1}{A[i][j]^2} -2*N*(mean[i]^2) + N*(mean[i])^2))/N Which is
// equal to: VAR[i] = (SUM_{j=0, N-1}{A[i][j]^2} + N*(mean[i]^2))/N

// So the STD matrix can be expressed as:
// STD[i][j] = (A[i][j] - mean[i]) / sqrt((SUM_{j=0, N-1}{A[i][j]^2} +
// N*(mean[i]^2))/N)

// We can continue by now looking at the STDCOV equation:
// STDCOV[i][j] = SUM_{k=0, N-1}{STD[i][k] * STD[j][k]}
// Which after substitution becomes:
// STDCOV[i][j] = SUM_{k=0, N-1}{((A[i][k] - mean[i]) / sqrt(VAR[i]) * ((A[j][k]
// - mean[j]) / sqrt(VAR[j]))} This can be simplified down to: STDCOV[i][j] =
// 1/(sqrt(var[i]*var[j])) * SUM_{k=0, N-1}{(A[i][k] - mean[i]) * (A[j][k] -
// mean[j])} We can expand the product in the sum to get: STDCOV[i][j] =
// 1/(sqrt(var[i]*var[j])) * (
//                       SUM_{k=0, N-1}{A[i][k] * A[j][k]}
//                     - SUM_{k=0, N-1}{mean[j] * A[i][k]}
//                     - SUM_{k=0, N-1}{mean[i] * A[j][k]}
//                     + SUM_{k=0, N-1}{mean[i] * mean[j]} )
// Which can be simplified to:
// STDCOV[i][j] = 1/(sqrt(var[i]*var[j])) * (
//                       SUM_{k=0, N-1}{A[i][k] * A[j][k]}
//                     - mean[j] * SUM_{k=0, N-1}{A[i][k]}
//                     - mean[i] * SUM_{k=0, N-1}{A[j][k]}
//                     + N * mean[i] * mean[j] )
// We noticed earlier that SUM_{k=0, N-1}{A[i][k]} is

// Let consider A_new the temporary matrix containing:
// A_new[i][j] = (A[i][j] - mean[i])/variance[i];
// Then the computation of the covariance matrix is defined as:
// C[i][j] = Dot(A_new[i][] , A_new[j][])
// We can expand the computation with the value of A_new:
// C[i][j] = Dot((A[i][] - mean[i])/variance[i] , (A[j][] -
// mean[j])/variance[j]) C[i][j] = (1/(variance[i]*variance[j])) * Dot((A[i][] -
// mean[i]), (A[j][] - mean[j])) C[i][j] = (1/(variance[i]*variance[j])) *
// (Dot(A[i][], A[j][]) - N* mean[i]*mean[j])

// mean[i] = sum(A[i][])/N
// var[i] = sqrt((Dot(A[i][], A[j][]) - N* mean[i]*mean[j])/N)

template <typename T,          // The datatype for the computation
          unsigned rows,       // Number of rows in the A matrices
          unsigned columns,    // Number of columns in the A matrices
          unsigned pipe_size,  // Number of elements read/write per pipe
                               // operation, the matrix is received through the
                               // pipe
          // by blocks of size columns*columns.
          typename InputPipe,  // A matrix input pipe, receive pipe_size
                               // elements from the pipe with each read
          typename OutputPipe  // Q matrix output pipe, send pipe_size
                               // elements to the pipe with each write
          >
struct StreamingCovarianceMatrix {
  void operator()() const {
    // static_assert(rows % columns == 0,
    //             "The feature count must be  a multiple of the samples count.
    //             " "This can be artificially achieved by increasing the number
    //             of " "samples with no data.");

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<T, rows>;

    // Number of matrix blocks to read from the pipe
    constexpr int block_count = rows / columns;

    // Matrix to hold the partial bloc results of At*A
    T t_matrix[columns][columns];

    // Array to keep the means of all the A matrix columns
    T means[columns];

    // Break memories up to store 4 complex numbers (32 bytes) per bank
    constexpr short kBankwidth = pipe_size * sizeof(T);
    constexpr unsigned short kNumBanks = rows / pipe_size;

    // When specifying numbanks for a memory, it must be a power of 2.
    // Unused banks will be automatically optimized away.
    constexpr short kNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

    // Copy a matrix from the pipe to a local memory
    // Number of pipe reads of pipe_size required to read a full column
    constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
    constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
    // Number of pipe reads of pipe_size to read all the matrices
    constexpr int kLoopIter = kLoopIterPerColumn * columns;

    while (1) {
      for (int block = 0; block < block_count; block++) {
        // Read the first matrix block into the a_load local memory

        // Three copies of the block matrix, so that each matrix has a single
        // load and a single store.
        // a_load is the initial matrix received from the pipe
        // a_compute is used and modified during calculations
        // q_result is a copy of a_compute and is used to send the final output

        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
        column_tuple a_load[columns];

        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (int li = 0; li < kLoopIter; li++) {
          fpga_tools::NTuple<T, pipe_size> pipe_read = InputPipe::read();

          int write_idx = li % kLoopIterPerColumn;
          int a_col_index = li / kLoopIterPerColumn;

          fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
            fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
              if (write_idx == k) {
                if constexpr (k * pipe_size + t < columns) {
                  a_load[a_col_index].template get<k * pipe_size + t>() =
                      pipe_read.template get<t>();
                }
              }

              // Delay data signals to create a vine-based data distribution
              // to lower signal fanout.
              pipe_read.template get<t>() =
                  sycl::ext::intel::fpga_reg(pipe_read.template get<t>());
            });

            write_idx = sycl::ext::intel::fpga_reg(write_idx);
          });
        }

        // Compute the block T matrix
        for (int row = 0; row < columns; row++) {
          // We are going to reuse the same column of the matrix multiple
          // iterations in a row, so we keep it locally
          column_tuple current_base_column;
          column_tuple next_base_column;

          for (int column = 0; column < columns; column++) {
            // Load the current column of the bloc
            column_tuple current_column = a_load[column];

            // Keep the current column in the local cache for future reuse
            if (column == 0) {
              if (column == row) {
                current_base_column = current_column;
              } else {
                current_base_column = next_base_column;
              }
            } else if (column == (row + 1)) {
              next_base_column = current_column;
            }

            T dot_product = 0;
            T mean = 0;
            fpga_tools::UnrolledLoop<columns>([&](auto t) {
              dot_product += current_column.template get<t>() *
                             current_base_column.template get<t>();
              mean += (current_column.template get<t>() / columns);
            });

            // Update the partial results matrix
            t_matrix[row][column] =
                block == 0 ? dot_product : dot_product + t_matrix[row][column];
            means[column] =
                (block == 0) || (row != 0) ? mean : mean + means[column];
          }  // end for:column
        }    // end for:row
      }      // end of for: block

      // t_matrix now contains the full matrix product of the transpose of A
      // times A. mean now contains the mean of all the columns of A We now need
      // to composed all of these results to get the covariance matrix
      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      T cov_matrix[columns][columns];

      for (int row = 0; row < columns; row++) {
        for (int column = 0; column < columns; column++) {
          T numerator =
              t_matrix[row][column] - columns * means[row] * means[column];

          T denominator = std::sqrt(t_matrix[row][row] -
                                    columns * means[row] * means[row]) *
                          std::sqrt(t_matrix[column][column] -
                                    columns * means[column] * means[column]);
          cov_matrix[row][column] = numerator / denominator;
        }  // end for:column
      }    // end for:row

      // Write the standardized covariance matrix to the output pipe
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int li = 0; li < kLoopIter; li++) {
        int column_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<T, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < columns) {
              pipe_write.template get<k>() =
                  get[t]
                      ? cov_matrix[li / kLoopIterPerColumn][t * pipe_size + k]
                      : sycl::ext::intel::fpga_reg(
                            pipe_write.template get<k>());
            }
          });
        });
        OutputPipe::write(pipe_write);
      }

      // storing in a internal matrix

      //   // NO-FORMAT: Attribute
      // double MatrixC[rows][rows], MatrixCW[rows][rows];
      // // row_tuple  Avg_G;
      // double Avg[rows];
      // pipe_tuple pipe_read;
      // double digValM[rows];

      // for(int blk = 0; blk < kColBlocks; blk++){

      //   // loading block data onchip memory
      //   // samples for a feature comes sequentially
      //   row_tuple MatrixA[rows];
      //   for(ac_int<kLoopIterBitSize, false> itr = 0; itr < kLoopItr; itr++){
      //     ac_int<kRowBitSize, false> i_ll = itr / kRowBlocks;
      //     ac_int<kRowBitSize, false> j_ll = itr % kRowBlocks;

      //     pipe_read = InputPipe::read();
      //     row_tuple rowblk;
      //     fpga_tools::UnrolledLoop<kRowBlocks>([&](auto k) {
      //       fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
      //         if(k == j_ll){
      //           if constexpr (k*pipe_size+t < rows){
      //             rowblk.template get<k*pipe_size+t> () = pipe_read.template
      //             get<t>();
      //           }
      //         }
      //       });
      //     });

      //     MatrixA[i_ll] = rowblk;

      //   }

      //   // computing the covariance matrix block wise and accumulating
      //   T row1[rows], row2[rows], row_temp[rows];
      //   for(ac_int<kRowBitSize, false> i_ll = 0; i_ll < rows; i_ll++){

      //     fpga_tools::UnrolledLoop<rows>([&](auto t) {
      //       row1[t] = row_temp[t];
      //     });

      //     if(blk == 0){
      //       Avg[i_ll] = 0;
      //     }

      //     T avg = 0;
      //     [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      //     // [[intel::ivdep(rowSumL, rows)]]
      //     for(int j_ll = 0; j_ll < rows; j_ll++){

      //       fpga_tools::UnrolledLoop<rows>([&](auto t) {
      //         row2[t] = MatrixA[j_ll].template get <t>();
      //         if(j_ll == i_ll + 1){
      //           row_temp[t] = row2[t];
      //         }
      //         if(i_ll == 0 && j_ll == 0){
      //           row1[t] = row2[t];
      //         }
      //       });

      //       T rowSum = 0;
      //       fpga_tools::UnrolledLoop<rows>([&](auto t) {
      //         T row1Elem = row1[t];
      //         rowSum += row1Elem * row2[t];
      //       });

      //       avg += row1[j_ll];

      //       T sum_a = rowSum;
      //       double sum_b = blk == 0 ? 0 : MatrixC[i_ll][j_ll];
      //       double sum = sum_a + sum_b;

      //       MatrixC[i_ll][j_ll] = sum;
      //       MatrixCW[i_ll][j_ll] = sum;

      //       if (i_ll == j_ll){
      //         digValM[i_ll] = sum;
      //       }

      //     } // end of j_ll

      //     Avg[i_ll] += avg/columns;
      //   } // end of i_ll

      // } // end of blk

      // // adjusting based on variance and mean
      // // C[i][j] = (1.0f/(variance[i]*variance[j])) * (Dot(A[i][], A[j][]) -
      // N* mean[i]*mean[j])
      // // mean[i] = sum(Dot(A[i][])/N
      // // var[i] = sqrt((Dot(A[i][], A[j][]) - N* mean[i]*mean[j])/N)
      // pipe_tuple pipe_write;
      // double avg1, avg2, avg_temp;
      // double digVal1, digVal2, dig_temp;
      // for(ac_int<kRowBitSize, false> i_ll = 0; i_ll < rows; i_ll++){
      //   for(ac_int<kRowBitSize, false> j_ll = 0; j_ll < rows; j_ll++){
      //     T loadVal;
      //     row_tuple loadRow;
      //     fpga_tools::UnrolledLoop<rows>([&](auto t) {
      //       loadRow.template get<t>() = MatrixCW[i_ll][t];
      //       if(j_ll == t){
      //         loadVal = loadRow.template get<t>();
      //       }
      //     });

      //     //---------------------------
      //     digVal2 = digValM[j_ll];
      //     avg2 = Avg[j_ll];
      //     if(j_ll == i_ll + 1){
      //       dig_temp = digVal2;
      //       avg_temp = avg2;
      //     }

      //     if(i_ll == 0 && j_ll == 0){
      //       digVal1 = digVal2;
      //       avg1 = avg2;
      //     } else if(j_ll == 0){
      //       digVal1 = dig_temp;
      //       avg1 = avg_temp;
      //     }

      //     T cov_i_i = digVal1 - columns * avg1 * avg1;
      //     T cov_j_j = digVal2 - columns * avg2 * avg2;

      //     T cov_i_j_tmp = loadVal - columns * avg1 * avg2;
      //     T cov_i_j = cov_i_j_tmp/sqrt(cov_i_i*cov_j_j);

      //     fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
      //       if(t == j_ll % pipe_size){
      //         pipe_write.template get<t> () = cov_i_j;
      //       }
      //     });

      //     if(j_ll % pipe_size == pipe_size -1 || j_ll == rows-1){
      //       OutputPipe::write(pipe_write);
      //     }

      //   } // end of j_ll
      // } // endo of i_ll

    }  // end of while

  };  // end of operator()
};    // end of struct{}

}  // namespace fpga_linalg

#endif /* __STREAMING_COVARIANCE_MATRIX_HPP__ */
