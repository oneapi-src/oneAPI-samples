#ifndef __STREAMING_CHOLESKY_INVERSION_HPP__
#define __STREAMING_CHOLESKY_INVERSION_HPP__

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif
#define PRINTF(format, ...)                                          \
  {                                                                  \
    static const CL_CONSTANT char _format[] = format;                \
    sycl::ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/*
  Cholesky decomposition - Computes L such that A=LL* where:
  - A is the input matrix (hermitian, positive definite)
  - L is a lower triangular matrix
  - L* is the conjugate transpose of L

  This function implements a modified version of the Cholesky–Banachiewicz
  algorithm.
  Pseudo code:

  int row_size = 0;
  for (column = 0; column <= row_size; column++) {
    for (row = column; row < rows; row++) {
      float sum = 0;
      for (k = 0; k < column; k++)
        sum += L[row][k] * L[column][k];

      if (row == column)
        L[row][column] = sqrt(A[row][row] - sum);
      else
        L[row][column] = (A[row][column] - sum) / L[column][column];
    }
  }

  The input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<X>
          int rows,         // Number of rows==columns in the A matrices
          int raw_latency,  // Read after write latency (in iterations) of
                            // the triangular loop of this function.
                            // This value depends on the FPGA target, the
                            // datatype, the target frequency, etc.
                            // This value will have to be tuned for optimal
                            // performance. Refer to the Triangular Loop
                            // design pattern tutorial.
                            // In general, find a high value for which the
                            // compiler is able to achieve an II of 1 and
                            // go down from there.
          int pipe_size,    // Number of elements read/write per pipe operation
                            // to read the input matrix
          typename LIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename IOut     // L matrix output pipe, send one elements to the
                            // pipe with each write.
                            // Only lower-left elements of L are
                            // sent in row order, starting with row 0.
          >
struct StreamingCholeskyInversion {
  void operator()() const {
    // Functional assertions
    static_assert(rows >= 4,
                  "Only matrices of size 4x4 and over are supported");
    static_assert(pipe_size >= 1,
                  "The pipe must be able to contain at least one element");

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    constexpr int kColumns = rows;

    // L matrix read from pipe
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
    TT l_matrix[rows][kColumns];

    // L inverse matrix for the compute
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
    TT li_matrix_compute[rows][kColumns];

    // L inverse matrix
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
    TT li_matrix[rows][kColumns];

    // Final inverse matrix (only the triangular elements)
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
    TT i_matrix[kColumns * (kColumns + 1) / 2];

    // Compute Cholesky-based inversions as long as L input matrices are given
    while (1) {
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col <= row; col++) {
          TT element = LIn::read();

          if constexpr (is_complex) {
            l_matrix[row][col] = element.conj();
          } else {
            l_matrix[row][col] = element;
          }
        }
      }

      // PRINTF("L matrix\n");
      // for (int row = 0; row < rows; row++) {
      //   for (int col = 0; col < rows; col++) {
      //     li_matrix_compute[row][col] = 0;
      //     PRINTF("%f ", l_matrix[row][col]);
      //     // PRINTF("%f %fi  ", l_matrix[row][col].r(),
      //     // l_matrix[row][col].i());
      //   }
      //   PRINTF("\n");
      // }

      /*
        Compute the inverse of L

        The inverse of L is computed using the following algorithm:

        LInverse = 0 // matrix initialized to 0
        for col=1:n
          for row=1:col-n
            // Because Id[row][col] = R[row:] * LInverse[:col], we have:
            // LInverse[row][col] = (Id[row][col] - L[row:] * LInverse[:col])
                                                                  /L[col][col]
            for k=1:n
              dp = L[col][k] * LInverse[row][k]

            LInverse[row][col] = (Id[row][col] - dp)/R[col][col]
      */

      // Count the total number of loop iterations, using the triangular loop
      // optimization (refer to the triangular loop optimization tutorial)
      constexpr int kNormalIterations = kColumns * (kColumns + 1) / 2;
      constexpr int kExtraIterations =
          raw_latency > rows
              ? (rows - 1) * (raw_latency - rows) + (rows - 2) * (rows - 1) / 2
              : (raw_latency - 2) * (raw_latency - 2 + 1) / 2;
      constexpr int kTotalIterations = kNormalIterations + kExtraIterations;

      constexpr int kInitIterations = 0;

      // All the loop control variables with all the requirements to apply
      // some shannonization (refer to the shannonization tutorial)

      /*
            for(int diag_number =  0; diag_number < kColumns; diag_number++){
              int diag_size = std::max(kColumns - diag_number, raw_latency);
              if(diag_number == kColumns-1){
                diag_size = 1;
              }
              int col = diag_number;
              for(int row = 0; row < diag_size; row++, col++){
                if(row<rows && col<kColumns){
                  //compute
                }
              }
            }
      */

      // All the loop control variables with all the requirements to apply
      // some shannonization (refer to the shannonization tutorial)

      int diagonal_number = 0;
      int next_diagonal_number = 1;
      int diagonal_size = (kColumns > raw_latency ? kColumns : raw_latency) - 1;
      int col = diagonal_number;
      int row = 0;

      [[intel::ivdep(raw_latency)]]  // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations + kInitIterations; it++) {
        // PRINTF("it: %d [%d][%d]\n", it, row, col);

        // Only perform work when in not dummy iterations
        if (row < rows & col < kColumns) {
          TT current_sum = row == col ? TT{1} : TT{0};
          TT div_val;

          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {

            // auto lhs = l_matrix[col][k];
            auto li_loaded = l_matrix[col][k];

            TT lhs;
            if(k > col){
              lhs = TT{0};
            }
            else{
              lhs = li_loaded;
            }

            auto li_compute_load = li_matrix_compute[row][k];
            TT rhs;
            if ((k >= row) && (k < col)) {
              rhs = li_compute_load;
            } else {
              rhs = TT{0};
            }
            // auto rhs = (k >= row) && (k < col) ? li_compute_load : TT{0};
            // auto rhs = li_compute_load;

            if (k == col) {
              div_val = lhs;
            }

            current_sum -= lhs * rhs;
          });

          TT result = current_sum / div_val;

          // Write the result to both the working copy and the final matrix
          // This is done to only have matrices with a single read and a
          // single write.
          li_matrix_compute[row][col] = result;

          li_matrix[row][col] = result;
        }

        if (row == diagonal_size) {
          diagonal_number = next_diagonal_number;
          diagonal_size =
              std::max(kColumns - next_diagonal_number, raw_latency) - 1;
          col = next_diagonal_number;
          row = 0;
          next_diagonal_number++;
        } else {
          row++;
          col++;
        }
      }

      // PRINTF("L inverse matrix\n");
      // for (int row = 0; row < rows; row++) {
      //   for (int col = 0; col < rows; col++) {
      //     PRINTF("%f ", li_matrix[row][col]);
      //   }
      //   PRINTF("\n");
      // }

      // PRINTF("L inverse matrix transpose\n");
      // for (int row = 0; row < rows; row++) {
      //   for (int col = 0; col < rows; col++) {
      //     PRINTF("%f ", li_matrix_transpose[row][col]);
      //   }
      //   PRINTF("\n");
      // }

      int idx = 0;
      // Compute inv(A) = inv(L)*trans(inv(L))
      for (int col = 0; col < rows; col++) {
        TT col_of_transpose_matrix[rows];
        int row_index;

        // for (int row = col; row < rows + col; row++) {
        for (int row = col; row < rows; row++) {
          if(row >= rows){
            row_index = row - rows;
          }
          else{
            row_index = row;
          }
          TT elem{0};
          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
            auto li_load = li_matrix[row_index][k];

            if (row_index == col) {
              col_of_transpose_matrix[k] = li_load;
            }

            auto lhs = k < row_index ? TT{0} : li_load;
            auto rhs = k < col ? TT{0} : col_of_transpose_matrix[k];
            if constexpr (is_complex) {
              elem += lhs * rhs.conj();
            } else {
              elem += lhs * rhs;
            }
          });
          i_matrix[idx] = elem;
          idx++;
        }
      }

      // PRINTF("Inverse matrix\n");
      // for (int row = 0; row < rows; row++) {
      //   for (int col = 0; col < rows; col++) {
      //     if(row >)
      //     PRINTF("%f ", i_matrix[row*kColumns + col]);
      //   }
      //   PRINTF("\n");
      // }

      int i_idx_pipe_write = 0;
      for(int idx = 0; idx < kNormalIterations; idx++){
        IOut::write(i_matrix[i_idx_pipe_write]);
        i_idx_pipe_write++;
      }


    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_CHOLESKY_INVERSION_HPP__ */