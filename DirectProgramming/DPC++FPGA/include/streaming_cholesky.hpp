#ifndef __STREAMING_CHOLESKY_HPP__
#define __STREAMING_CHOLESKY_HPP__

#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"

/*
  Cholesky decomposition - Computes L such that A=LL* where:
  - A is the input matrix (hermitian, positive definite)
  - L is a lower triangular matrix
  - L* is the conjugate transpose of L

  This function implements the Cholesky–Banachiewicz algorithm.
  Pseudo code:
  for (i = 0; i < dimensionSize; i++) {
    for (j = 0; j <= i; j++) {
        float sum = 0;
        for (k = 0; k < j; k++)
            sum += L[i][k] * L[j][k];

        if (i == j)
            L[i][j] = sqrt(A[i][i] - sum);
        else
            L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum));
    }
  }

  The input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,        // The datatype for the computation
          bool is_complex,   // True if T is ac_complex<X>
          int rows,          // Number of rows in the A matrices
          int columns,       // Number of columns in the A matrices
                             // , must be <= rows
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
          int pipe_size,     // Number of elements read/write per pipe operation
          typename AIn,      // A matrix input pipe, receive pipe_size
                             // elements from the pipe with each read
          typename LOut      // L matrix output pipe, send pipe_size
                             // elements to the pipe with each write.
                             // Only lower-left elements of L are
                             // sent in row order, starting with row 0.
          >
struct StreamingCholesky {
  void operator()() const {
    // Functional limitations
    static_assert(rows >= columns,
                  "only rectangular matrices with rows>=columns are supported");
    // static_assert(columns >= 4,
    //               "only matrices of size 4x4 and over are supported");

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<TT, rows>;

    // Number of upper-right elements in the R output matrix
    constexpr int kLMatrixSize = columns * (columns + 1) / 2;

    // Compute QRDs as long as matrices are given as inputs
    while(1) {
      // Three copies of the full matrix, so that each matrix has a single
      // load and a single store.
      // a_load is the initial matrix received from the pipe
      // a_compute is used and modified during calculations

      // Break memories up to store 4 complex numbers (32 bytes) per bank
      constexpr short kBankwidth = pipe_size * sizeof(TT);
      constexpr unsigned short kNumBanks = rows / pipe_size;

      // When specifying numbanks for a memory, it must be a power of 2.
      // Unused banks will be automatically optimized away.
      constexpr short kNumBanksNextPow2 =
                              fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT a_load[rows][columns], a_compute[rows][columns];
      // column_tuple a_load[columns], a_compute[columns];

      // Contains the values of the upper-right part of R in a row by row
      // fashion, starting by row 0
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      // TT l_result[kLMatrixSize];
      TT l_result[rows][columns];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
                                  fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read = AIn::read();

        int write_idx = li % kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < rows) {
                // a_load[li / kLoopIterPerColumn].template get<k * pipe_size 
                //                           + t>() = pipe_read.template get<t>();
                a_load[li / kLoopIterPerColumn][k * pipe_size + t] = 
                                                    pipe_read.template get<t>();
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipe_read. template get<t>() =
                      sycl::ext::intel::fpga_reg(pipe_read. template get<t>());
          });

          write_idx = sycl::ext::intel::fpga_reg(write_idx);
        });
      }

      for (int row = 0; row < rows; row++) {
        for (int column = 0; column <= columns; column++) {
          l_result[row][column] = {0};
        }
      }

      for (int row = 0; row < rows; row++) {
        for (int column = 0; column <= row; column++) {
          TT sum = 0;
          for (int k = 0; k < column; k++)
            if constexpr (is_complex) {
              sum += l_result[row][k] * l_result[column][k].conj();
            }
            else {
              sum += l_result[row][k] * l_result[column][k];
            }

          if (row == column) {
            if constexpr (is_complex) {
              l_result[row][column] = {sqrt(a_load[row][row].r() - sum.r()), 0};
            }
            else{
              l_result[row][column] = sqrt(a_load[row][row] - sum);
            }
          }
          else {
            l_result[row][column] = (1.0 / l_result[column][column] * (a_load[row][column] - sum));
          }
        }
      }


      for (int row = 0; row < rows; row++) {
        for (int column = 0; column <= columns; column++) {
          if(column <= row){
            if constexpr (is_complex) {
              LOut::write(l_result[row][column].conj());
            }
            else{
              LOut::write(l_result[row][column]);
            }
          }
        }
      }

      // [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      // for (int r_idx = 0; r_idx < kLMatrixSize; r_idx++) {
      // }

    }  // end of while(1)
  }    // end of operator
};     // end of struct

#endif /* __STREAMING_CHOLESKY_HPP__ */