#pragma once

#include "Utils.hpp"

/*
  QRI (QR inversion) - Given two matrices Q and R from the QR decomposition
  of a matrix A such that A=QR, this function computes the inverse of A.
  - Input matrix Q (unitary/orthogonal)
  - Input matrix R (upper triangular)
  - Output matrix I, the inverse of A such that A=QR

  Then input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,        // The datatype for the computation
          bool isComplex,    // True if T is ac_complex<T>
          int rows,          // Number of rows in the input matrices
          int columns,       // Number of columns in the input matrices
          int RAWLatency,    // Read after write latency (in iterations) of
                             // the triangular loop of this function.
                             // This value depends on the FPGA target, the
                             // datatype, the target frequency, etc.
                             // This value will have to be tuned for optimal
                             // performance. Refer to the Triangular Loop
                             // design pattern tutorial.
                             // In general, find a high value for which the
                             // compiler is able to achieve an II of 1 and
                             // go down from there.
          int matrixCount,   // Number of matrices to read from the input
                             // pipes sequentially
          int pipeElemSize,  // Number of elements read/write per pipe
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
    // of isComplex
    typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

    // Iterate over the number of matrices to decompose per function call
    for (int matrixIter = 0; matrixIter < matrixCount; matrixIter++) {
      // Q matrix read from pipe
      TT QMatrix[rows][columns];
      // Transpose of Q matrix
      TT QTMatrix[rows][columns];
      // R matrix read from pipe
      TT RMatrix[rows][columns];
      // Transpose of R matrix
      TT RTMatrix[rows][columns];
      // Inverse of R matrix
      TT RIMatrix[rows][columns];
      // Inverse matrix of A=QR
      TT IMatrix[rows][columns];

      /*
        ======================================================================
        Copy a R matrix from the pipe to a local memory
        ======================================================================
      */
      int readCounter = 0;
      int nextReadCounter = 1;
      pipeTable<pipeElemSize, TT> read;
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int i = 0; i < rows; i++) {
        // "Shift register" that will contain a full row of R after
        // columns iterations.
        // Each pipe read writes to RRow[columns-1] and at each loop iteration
        // each RRow[x] will be assigned RRow[x+1]
        // This ensures that the fanout is kept to a minimum
        TT RRow[columns];
        bool cond;
        bool nextCond = 0 >= i;
        int tosum = -i + 2;
        for (int j = 0; j < columns; j++) {
          cond = nextCond;
          nextCond = j + tosum > 0;
          // For shannonization
          int potentialNextReadCounter = readCounter + 2;

// Perform the register shifting of the banks
#pragma unroll
          for (int col = 0; col < columns - 1; col++) {
            RRow[col] = RRow[col + 1];
          }

          if (cond && (readCounter == 0)) {
            read = RIn::read();
          }
          // Read a new value from the pipe if the current row element
          // belongs to the upper-right part of R. Otherwise write 0.
          if (cond) {
            RRow[columns - 1] = read.elem[readCounter];
            readCounter = nextReadCounter % pipeElemSize;
            nextReadCounter = potentialNextReadCounter;
          } else {
            RRow[columns - 1] = TT{0.0};
          }
        }

        // Copy the entire row to the R matrix
        UnrolledLoop<columns>([&](auto k) { RMatrix[i][k] = RRow[k]; });
      }

      /*
        ======================================================================
        Copy a Q matrix from the pipe to a local memory
        ======================================================================
      */

      // Number of DDR burst reads of pipeElemSize required to read a full
      // column
      constexpr int kExtraIteration = (rows % pipeElemSize) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipeElemSize + kExtraIteration;
      // Number of DDR burst reads of pipeElemSize to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        pipeTable<pipeElemSize, TT> pipeRead = QIn::read();

        int writeIdx = li % kLoopIterPerColumn;

        UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          UnrolledLoop<pipeElemSize>([&](auto t) {
            if (writeIdx == k) {
              if constexpr (k * pipeElemSize + t < rows) {
                QMatrix[li / kLoopIterPerColumn][k * pipeElemSize + t] =
                    pipeRead.elem[t];
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipeRead.elem[t] = sycl::ext::intel::fpga_reg(pipeRead.elem[t]);
          });

          writeIdx = sycl::ext::intel::fpga_reg(writeIdx);
        });
      }

      /*
        ======================================================================
        Transpose the R matrix
        ======================================================================
      */
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          RTMatrix[row][col] = RMatrix[col][row];
        }
      }

      /*
        ======================================================================
        Transpose the Q matrix (to get Q as non transposed)
        ======================================================================
      */
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          QTMatrix[row][col] = QMatrix[col][row];
        }
      }

      /*
        ======================================================================
        Compute the inverse of R
        ======================================================================

        The inverse of R is computed using the following algorithm:

        RInverse = 0 // matrix initialized to 0
        for col=1:n
          for row=1:col-n
            // Because Id[row][col] = R[row:] * RInverse[:col], we have:
            // RInverse[row][col] = (Id[row][col] - R[row:] * RInverse[:col])
                                                                  /R[col][col]
            for k=1:n
              dp = R[col][k] * RIMatrix[row][k]

            RInverse[row][col] = (Id[row][col] - dp)/R[col][col]
      */

      // Initialise RIMatrix with 0
      for (int i = 0; i < rows; i++) {
        UnrolledLoop<columns>([&](auto k) { RIMatrix[i][k] = {0}; });
      }

      // Count the total number of loop iterations, using the triangular loop
      // optimization (refer to the triangular loop optimization tutorial)
      constexpr int kNormalIterations = rows * (rows + 1) / 2;
      constexpr int kExtraIterations =
          RAWLatency > rows
              ? (RAWLatency - 2) * (RAWLatency - 2 + 1) / 2 -
                    (RAWLatency - rows - 1) * (RAWLatency - rows) / 2
              : (RAWLatency - 2) * (RAWLatency - 2 + 1) / 2;
      constexpr int kTotalIterations = kNormalIterations + kExtraIterations;

      // All the loop control variables with all the requirements to apply
      // some shannonization (refer to the shannonization tutorial)
      int row = 0;
      int col = 0;
      int cp1 = 1;
      int iter = 0;
      int ip1 = 1;
      int ip2 = 2;
      int diagSize = columns;
      int diagSizem1 = columns - 1;
      int cp1Limit =
          RAWLatency - columns - columns > 0 ? RAWLatency - columns : columns;
      int nextcp1Limit = RAWLatency - columns - 1 - columns > 0
                             ? RAWLatency - columns - 1
                             : columns;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep(RAWLatency)]]       // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations; it++) {
        // Only compute during the non dummy iterations
        if ((row < rows) & (col < columns)) {
          // Compute the dot product of R[row:] * RInverse[:col]
          TT dotProduct = {0};

          // While reading R, keep the R[col][col] value for the follow up
          // division
          TT div_val;

          UnrolledLoop<columns>([&](auto k) {
            auto lhs = RTMatrix[col][k];
            auto rhs = RIMatrix[row][k];

            if (k == col) {
              div_val = lhs;
            }

            dotProduct += lhs * rhs;
          });

          // Find the value of the identity matrix at these coordinates
          TT idMatrixValue = row == col ? TT{1} : TT{0};
          // Compute the value of the inverse of R
          RIMatrix[row][col] = (idMatrixValue - dotProduct) / div_val;
        }

        // Update loop indexes
        if (cp1 >= cp1Limit) {
          col = ip1;
          cp1 = ip2;
          iter = ip1;
          row = 0;
          diagSize = diagSizem1;
          cp1Limit = nextcp1Limit;
        } else {
          col = cp1;
          cp1 = col + 1;
          row = row + 1;
          ip1 = iter + 1;
          ip2 = iter + 2;
          nextcp1Limit = sycl::max(RAWLatency - (diagSize - 1), columns);
          diagSizem1 = diagSize - 1;
        }
      }

      /*
        ======================================================================
        Multiply the inverse of R by the transposition of Q
        ======================================================================
      */
      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
          TT dotProduct = {0.0};
          UnrolledLoop<rows>([&](auto k) {
            if constexpr (isComplex) {
              dotProduct += RIMatrix[row][k] * QTMatrix[col][k].conj();
            } else {
              dotProduct += RIMatrix[row][k] * QTMatrix[col][k];
            }
          });
          IMatrix[row][col] = dotProduct;
        }  // end of col
      }    // end of row

      /*
        ======================================================================
        Copy the inverse matrix result to the output pipe
        ======================================================================
      */
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int columnIter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = columnIter == k;
          columnIter = sycl::ext::intel::fpga_reg(columnIter);
        });

        pipeTable<pipeElemSize, TT> pipeWrite;
        UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          UnrolledLoop<pipeElemSize>([&](auto k) {
            if constexpr (t * pipeElemSize + k < rows) {
              pipeWrite.elem[k] =
                  get[t]
                      ? IMatrix[li / kLoopIterPerColumn][t * pipeElemSize + k]
                      : sycl::ext::intel::fpga_reg(pipeWrite.elem[k]);
            }
          });
        });

        IOut::write(pipeWrite);
      }
    }  // end of matrixIter
  }    // end of operator
};     // end of struct