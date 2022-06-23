#ifndef __STREAMING_QRI_HPP__
#define __STREAMING_QRI_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/*
  QRI (QR inversion) - Given two matrices Q and R from the QR decomposition
  of a matrix A such that A=QR, this function computes the inverse of A.
  - Input matrix Q (unitary/orthogonal)
  - Input matrix R (upper triangular)
  - Output matrix I, the inverse of A such that A=QR

  Then input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<T>
          int rows,         // Number of rows in the input matrices
          int columns,      // Number of columns in the input matrices
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
          int pipe_size,    // Number of elements read/write per pipe
                            // operation
          typename QIn,     // Q input pipe, receive a full column with each
                            // read.
          typename RIn,     // R input pipe. Receive one element per read.
                            // Only upper-right elements of R are sent.
                            // Sent in row order, starting with row 0.
          typename IOut     // Inverse matrix output pipe.
                            // The output is written column by column
          >
struct StreamingQRI {
  void operator()() const {
    // Functional limitations
    static_assert(rows == columns,
                  "only square matrices with rows==columns are supported");
    static_assert(columns >= 4,
                  "only matrices of size 4x4 and upper are supported");

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Continue to compute as long as matrices are given as inputs
    while (1) {
      // Q matrix read from pipe
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT q_matrix[rows][columns];

      // Transpose of Q matrix
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT qt_matrix[rows][columns];

      // Transpose of R matrix
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT rt_matrix[rows][columns];

      // Inverse of R matrix
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT ri_matrix[rows][columns];

      [[intel::private_copies(2)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT ri_matrix_compute[rows][columns];

      // Inverse matrix of A=QR
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT i_matrix[rows][columns];

      // Transpose the R matrix
      [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          rt_matrix[col][row] = col < row ? TT{0} : RIn::read();
        }
      }

      // Copy a Q matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full
      // column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
          fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read = QIn::read();

        int write_idx = li % kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < rows) {
                q_matrix[li / kLoopIterPerColumn][k * pipe_size + t] =
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

      /*
        Compute the inverse of R

        The inverse of R is computed using the following algorithm:

        RInverse = 0 // matrix initialized to 0
        for col=1:n
          for row=1:col-n
            // Because Id[row][col] = R[row:] * RInverse[:col], we have:
            // RInverse[row][col] = (Id[row][col] - R[row:] * RInverse[:col])
                                                                  /R[col][col]
            for k=1:n
              dp = R[col][k] * ri_matrix[row][k]

            RInverse[row][col] = (Id[row][col] - dp)/R[col][col]
      */

      // Count the total number of loop iterations, using the triangular loop
      // optimization (refer to the triangular loop optimization tutorial)
      constexpr int kNormalIterations = columns * (columns + 1) / 2;
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
        if (row < rows & col < columns) {
          qt_matrix[row][col] = q_matrix[col][row];

          TT current_sum = row == col ? TT{1} : TT{0};
          TT div_val;

          fpga_tools::UnrolledLoop<columns>([&](auto k) {
            auto lhs = rt_matrix[col][k];
            auto rhs =
                (k >= col) || (col < row) ? TT{0} : ri_matrix_compute[row][k];
            if (k == col) {
              div_val = lhs;
            }

            current_sum -= lhs * rhs;
          });

          TT result = current_sum / div_val;

          // Write the result to both the working copy and the final matrix
          // This is done to only have matrices with a single read and a
          // single write.
          ri_matrix_compute[row][col] = result;
          ri_matrix[row][col] = result;
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

      // Multiply the inverse of R by the transposition of Q
      [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          TT dot_product = {0.0};
          fpga_tools::UnrolledLoop<rows>([&](auto k) {
            if constexpr (is_complex) {
              dot_product += ri_matrix[row][k] * qt_matrix[col][k].conj();
            } else {
              dot_product += ri_matrix[row][k] * qt_matrix[col][k];
            }
          });
          i_matrix[row][col] = dot_product;
        }  // end of col
      }    // end of row

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
                  get[t] ? i_matrix[li / kLoopIterPerColumn][t * pipe_size + k]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });

        IOut::write(pipe_write);
      }
    }  // end of while (1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_QRI_HPP__ */
