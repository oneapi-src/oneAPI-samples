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

    // L inverse matrix transposed
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
    TT li_matrix_transpose[kColumns][rows];

    // final inverse matrix
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
    TT i_matrix[rows][kColumns];

    // Compute Cholesky-based inversions as long as L input matrices are given
    while (1) {
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < rows; col++) {
          TT element;
          if (col > row) {
            element = {0};
          } else {
            element = LIn::read();
          }
          if constexpr (is_complex) {
            l_matrix[row][col] = element.conj();
          }
          else{
            l_matrix[row][col] = element;
          }
        }
      }

      // PRINTF("L matrix\n");
      // for (int col = 0; col < rows; col++) {
      //   for (int row = 0; row < rows; row++) {
      //     li_matrix_compute[row][col] = 0;
      //     PRINTF("%f %fi  ", l_matrix[row][col].r(), l_matrix[row][col].i());
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
      constexpr bool kThereAreExtraInitIterations = rows - 1 < raw_latency - 1;
      constexpr int kInitExtraIterations =
          kThereAreExtraInitIterations ? raw_latency - 1 : rows - 1;
      constexpr int kInitIterations =
          (rows - 2) * (rows - 1) / 2 + kInitExtraIterations;

      // All the loop control variables with all the requirements to apply
      // some shannonization (refer to the shannonization tutorial)
      int row = rows - 1;
      int row_plus_1 = rows;
      int col = 0;
      int col_plus_1 = 1;
      int row_limit = rows - 1;
      int diag_size = 1;
      int next_diag_size = 2;
      int diag_iteration = 0;
      int diag_iteration_plus_1 = 1;
      int start_row = rows - 2;
      int start_row_plus_1 = rows - 1;
      int start_col = 0;
      int start_col_plus_1 = 1;
      int next_row_limit = rows - 1;

      [[intel::ivdep(raw_latency)]]  // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations + kInitIterations; it++) {
        // Only perform work when in not dummy iterations
        if (row < rows & col < kColumns) {
          TT current_sum = row == col ? TT{1} : TT{0};
          TT div_val;

          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
            auto lhs = l_matrix[col][k];
            auto rhs =
                (k >= col) || (col < row) ? TT{0} : li_matrix_compute[row][k];

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
          li_matrix_transpose[col][row] = result;
        }

        // Update loop indexes
        if (row == row_limit) {
          row_limit = next_row_limit;
          if constexpr (kThereAreExtraInitIterations) {
            next_row_limit = diag_iteration + 2 >= rows - 2
                                 ? sycl::max(next_diag_size, raw_latency - 1)
                                 : rows - 1;
          } else {
            next_row_limit =
                diag_iteration + 2 >= rows
                    ? sycl::max(next_diag_size - 2, raw_latency - 1)
                    : rows - 1;
          }

          diag_size = next_diag_size;
          int to_sum = diag_iteration >= rows - 2 ? -1 : 1;
          next_diag_size = diag_size + to_sum;

          row = start_row;
          row_plus_1 = start_row_plus_1;
          col = start_col;
          col_plus_1 = start_col_plus_1;
          int start = diag_iteration + 1 - rows + 2;
          start_col = start < 0 ? 0 : start;
          start_col_plus_1 = start_col + 1;
          start_row = start >= 0 ? 0 : -start;
          start_row_plus_1 = start_row + 1;

          diag_iteration = diag_iteration_plus_1;
          diag_iteration_plus_1 = diag_iteration_plus_1 + 1;
        } else {
          row = row_plus_1;
          row_plus_1++;
          col = col_plus_1;
          col_plus_1++;
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

      // Compute inv(A) = trans(inv(L))*inv(L)
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < rows; col++) {
          TT elem{0};
          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
            if constexpr (is_complex) {
              elem += li_matrix[row][k] * li_matrix_transpose[k][col].conj();
            }
            else{
              elem += li_matrix[row][k] * li_matrix_transpose[k][col];
            }
          });
          i_matrix[row][col] = elem;
        }
      }

      // PRINTF("Inverse matrix\n");
      // for (int row = 0; row < rows; row++) {
      //   for (int col = 0; col < rows; col++) {
      //     PRINTF("%f ", i_matrix[row][col]);
      //   }
      //   PRINTF("\n");
      // }


      // Copy the inverse matrix from the local memory to the output pipe
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * kColumns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
                                  fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      // Copy the inverse matrix result to the output pipe
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int column_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<TT, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < rows) {
              pipe_write.template get<k>() =
                  get[t]
                      ? i_matrix[li / kLoopIterPerColumn][t * pipe_size + k]
                    : sycl::ext::intel::fpga_reg(pipe_write.template get<k>());
            }
          });
        });

        IOut::write(pipe_write);
      }

    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_CHOLESKY_INVERSION_HPP__ */