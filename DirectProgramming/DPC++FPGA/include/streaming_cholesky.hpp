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
template <typename T,        // The datatype for the computation
          bool is_complex,   // True if T is ac_complex<X>
          int rows,          // Number of rows==columns in the A matrices
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
                             // to read the input matrix
          typename AIn,      // A matrix input pipe, receive pipe_size
                             // elements from the pipe with each read
          typename LOut      // L matrix output pipe, send one elements to the
                             // pipe with each write.
                             // Only lower-left elements of L are
                             // sent in row order, starting with row 0.
          >
struct StreamingCholesky {
  void operator()() const {

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    constexpr int kColumns = rows;

    // Number of lower-left elements in the L output matrix
    constexpr int kLMatrixSize = kColumns * (kColumns + 1) / 2;

    // Compute Cholesky decompositions as long as matrices are given as inputs
    while(1) {

      // Break memories up to store pipe_size elements per bank
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
      TT a_load[rows][kColumns];

      // Two copies of L to be able to load two complete rows per iteration
      // Multiple private copies to be able to overlap multiple loop
      // iterations
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      TT l_result_compute[rows][kColumns];
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      TT l_result_compute_copy[rows][kColumns];

      [[intel::private_copies(2)]]            // NO-FORMAT: Attribute
      TT l_result[kLMatrixSize];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * kColumns;
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
              if constexpr (k * pipe_size + t < kColumns) {
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


      // Computation of the number of iterations required for the triangular
      // loop. Refer to the triangular_loop tutorial for details on how
      // to compute these.
      constexpr int kRegularIterations = kColumns * (kColumns + 1) / 2;
      constexpr int kExtraIterations = (raw_latency - 1) * raw_latency / 2;
      constexpr int kExtraIterationsToRemove = kColumns >= raw_latency ? 0 :
                    (raw_latency - kColumns) * (raw_latency - kColumns + 1) /2;
      constexpr int kTotalIterations = kRegularIterations + kExtraIterations
                                       - kExtraIterationsToRemove;

      // Compute the L matrix
      int column = 0;
      int row = 0;
      TT div_term{0};
      [[intel::ivdep(raw_latency)]]
      for(int iteration = 0; iteration < kTotalIterations; iteration++){

        // Only do usefull work for meaningfull iterations
        if(column <= row){
          // Perform the dot product of the elementsof the two rows indexed by 
          // row and column from element 0 to column
          TT sum = 0;
          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
            TT to_add;
            TT mul_lhs = k < column ? l_result_compute[row][k] : TT{0};
            TT mul_rhs = l_result_compute_copy[column][k];

            if constexpr (is_complex) {
              to_add = mul_lhs * mul_rhs.conj();
            }
            else {
              to_add = mul_lhs * mul_rhs;
            }
            sum += to_add;
          });

          TT diff = a_load[row][column] - sum;

          TT to_store;
          if (row == column) {
            // Perform the reciprocal sqrt rather than the sqrt because:
            // - it has a shorter latency and will reduce the RAW latency
            //   of the loop
            // - the result of the sqrt is used as a divisor which is also 
            //   a long operation, so replacing x/sqrt by x*rsqrt will save
            //   latency
            // - the diagonal elements will need to be inverted later, but we 
            //   can do that while outside this loop when we transfer the L 
            //   matrix to the pipe
            if constexpr (is_complex) {
              div_term = {sycl::rsqrt(diff.r()), 0};
            }
            else{
              div_term = sycl::rsqrt(diff);
            }
            to_store = div_term;
          }
          else {
            to_store = diff * div_term;
          }

          // Store the results to two working copies of L to be able to read 
          // two complete rows at each iteration
          l_result_compute[row][column] = to_store;
          l_result_compute_copy[row][column] = to_store;
          // Store the result to the output matrix
          l_result[row*(row+1)/2+column] = to_store;
        }

        // Update loop indexes
        if(row == (rows - 1)) {
          column = column + 1;
          row = sycl::min(column, rows-raw_latency);
        }
        else{
          row = row + 1;
        }

      } // end of iteration


      // Go over the L matrix and write each element to the pipe
      int l_idx = 0;
      [[intel::loop_coalesce(2)]]
      for(int row = 0; row < rows; row++){
        for(int column = 0; column <= row; column++){
          TT to_write;
          TT current_l_value = l_result[l_idx];
          // The diagonal elements need to be inverted as the
          // inversion was removed from the above compute loop
          // to reduce the RAW latency
          if(row == column){
            to_write = 1 / current_l_value;
          }
          else{
            to_write = current_l_value;
          }
          LOut::write(to_write);

          l_idx++;
        }
      }

    }  // end of while(1)
  }    // end of operator
};     // end of struct

#endif /* __STREAMING_CHOLESKY_HPP__ */