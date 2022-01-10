#pragma once

#include "utils.hpp"
#include "unrolled_loop.hpp"


/*
  QRI (QR inversion) - Given two matrices Q and R from the QR decomposition
  of a matrix A such that A=QR, this function computes the inverse of A.
  - Input matrix Q (unitary/orthogonal)
  - Input matrix R (upper triangular)
  - Output matrix I, the inverse of A such that A=QR

  Then input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,        // The datatype for the computation
          bool is_complex,   // True if T is ac_complex<T>
          int rows,          // Number of rows in the input matrices
          int columns,       // Number of columns in the input matrices
          int raw_latency,   // Read after write latency (in iterations) of
                             // the triangular loop of this function.
                             // This value depends on the FPGA target, the
                             // datatype, the target frequency, etc.
                             // This value will have to be tuned for optimal
                             // performance. Refer to the Triangular Loop
                             // design pattern tutorial.
                             // In general, find a high value for which the
                             // compiler is able to achieve an II of 1 and
                             // go down from there.
          int matrix_count,  // Number of matrices to read from the input
                             // pipes sequentially
          int pipe_size,     // Number of elements read/write per pipe
                             // operation
          typename QIn,      // Q input pipe, receive a full column with each
                             // read.
          typename RIn,      // R input pipe. Receive one element per read.
                             // Only upper-right elements of R are sent.
                             // Sent in row order, starting with row 0.
          typename IOut      // Inverse matrix output pipe.
                             // The output is written column by column
          >
struct StreamingQRI {
  void operator()() const {
    // Functional limitations
    static_assert(rows == columns,
                  "only square matrices with rows==columns are supported");
    static_assert((columns <= 512) && (columns >= 4),
                  "only matrices of size 4x4 to 512x512 are supported");

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Iterate over the number of matrices to decompose per function call
    for (int matrix_iter = 0; matrix_iter < matrix_count; matrix_iter++) {
      // Q matrix read from pipe
      TT q_matrix[rows][columns];
      // Transpose of Q matrix
      TT qt_matrix[rows][columns];
      // R matrix read from pipe
      TT r_matrix[rows][columns];
      // Transpose of R matrix
      TT rt_matrix[rows][columns];
      // Inverse of R matrix
      TT ri_matrix[rows][columns];
      // Inverse matrix of A=QR
      TT i_matrix[rows][columns];

      // Copy a R matrix from the pipe to a local memory
      int read_counter = 0;
      int next_read_counter = 1;
      PipeTable<pipe_size, TT> read;
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int i = 0; i < rows; i++) {
        // "Shift register" that will contain a full row of R after
        // columns iterations.
        // Each pipe read writes to r_row[columns-1] and at each loop iteration
        // each r_row[x] will be assigned r_row[x+1]
        // This ensures that the fanout is kept to a minimum
        TT r_row[columns];
        bool cond;
        bool next_cond = 0 >= i;
        int to_sum = -i + 2;
        for (int j = 0; j < columns; j++) {
          cond = next_cond;
          next_cond = j + to_sum > 0;
          // For shannonization
          int potential_next_read_counter = read_counter + 2;

// Perform the register shifting of the banks
#pragma unroll
          for (int col = 0; col < columns - 1; col++) {
            r_row[col] = r_row[col + 1];
          }

          if (cond && (read_counter == 0)) {
            read = RIn::read();
          }
          // Read a new value from the pipe if the current row element
          // belongs to the upper-right part of R. Otherwise write 0.
          if (cond) {
            r_row[columns - 1] = read.elem[read_counter];
            read_counter = next_read_counter % pipe_size;
            next_read_counter = potential_next_read_counter;
          } else {
            r_row[columns - 1] = TT{0.0};
          }
        }

        // Copy the entire row to the R matrix
        UnrolledLoop<columns>([&](auto k) { r_matrix[i][k] = r_row[k]; });
      }

      
      // Copy a Q matrix from the pipe to a local memory
      // Number of DDR burst reads of pipe_size required to read a full
      // column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of DDR burst reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        PipeTable<pipe_size, TT> pipe_read = QIn::read();

        int write_idx = li % kLoopIterPerColumn;

        UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < rows) {
                q_matrix[li / kLoopIterPerColumn][k * pipe_size + t] =
                    pipe_read.elem[t];
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipe_read.elem[t] = sycl::ext::intel::fpga_reg(pipe_read.elem[t]);
          });

          write_idx = sycl::ext::intel::fpga_reg(write_idx);
        });
      }

      
      // Transpose the R matrix
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          rt_matrix[row][col] = r_matrix[col][row];
        }
      }

      
      // Transpose the Q matrix (to get Q as non transposed)
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          qt_matrix[row][col] = q_matrix[col][row];
        }
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
      // Initialise ri_matrix with 0
      for (int i = 0; i < rows; i++) {
        UnrolledLoop<columns>([&](auto k) { ri_matrix[i][k] = {0}; });
      }

      // Count the total number of loop iterations, using the triangular loop
      // optimization (refer to the triangular loop optimization tutorial)
      constexpr int kNormalIterations = rows * (rows + 1) / 2;
      constexpr int kExtraIterations =
          raw_latency > rows
              ? (raw_latency - 2) * (raw_latency - 2 + 1) / 2 -
                    (raw_latency - rows - 1) * (raw_latency - rows) / 2
              : (raw_latency - 2) * (raw_latency - 2 + 1) / 2;
      constexpr int kTotalIterations = kNormalIterations + kExtraIterations;

      // All the loop control variables with all the requirements to apply
      // some shannonization (refer to the shannonization tutorial)
      int row = 0;
      int col = 0;
      int cp1 = 1;
      int iter = 0;
      int ip1 = 1;
      int ip2 = 2;
      int diag_size = columns;
      int diag_size_m1 = columns - 1;
      int cp1_limit =
          raw_latency - columns - columns > 0 ? raw_latency - columns : columns;
      int next_cp1_limit = raw_latency - columns - 1 - columns > 0
                             ? raw_latency - columns - 1
                             : columns;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep(raw_latency)]]       // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations; it++) {
        // Only compute during the non dummy iterations
        if ((row < rows) & (col < columns)) {
          // Compute the dot product of R[row:] * RInverse[:col]
          TT dot_product = {0};

          // While reading R, keep the R[col][col] value for the follow up
          // division
          TT div_val;

          UnrolledLoop<columns>([&](auto k) {
            auto lhs = rt_matrix[col][k];
            auto rhs = ri_matrix[row][k];

            if (k == col) {
              div_val = lhs;
            }

            dot_product += lhs * rhs;
          });

          // Find the value of the identity matrix at these coordinates
          TT idMatrixValue = row == col ? TT{1} : TT{0};
          // Compute the value of the inverse of R
          ri_matrix[row][col] = (idMatrixValue - dot_product) / div_val;
        }

        // Update loop indexes
        if (cp1 >= cp1_limit) {
          col = ip1;
          cp1 = ip2;
          iter = ip1;
          row = 0;
          diag_size = diag_size_m1;
          cp1_limit = next_cp1_limit;
        } else {
          col = cp1;
          cp1 = col + 1;
          row = row + 1;
          ip1 = iter + 1;
          ip2 = iter + 2;
          next_cp1_limit = sycl::max(raw_latency - (diag_size - 1), columns);
          diag_size_m1 = diag_size - 1;
        }
      }

      
      // Multiply the inverse of R by the transposition of Q
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          TT dot_product = {0.0};
          UnrolledLoop<rows>([&](auto k) {
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
        UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        PipeTable<pipe_size, TT> pipe_write;
        UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < rows) {
              pipe_write.elem[k] =
                  get[t]
                      ? i_matrix[li / kLoopIterPerColumn][t * pipe_size + k]
                      : sycl::ext::intel::fpga_reg(pipe_write.elem[k]);
            }
          });
        });

        IOut::write(pipe_write);
      }
    }  // end of matrix_iter
  }    // end of operator
};     // end of struct