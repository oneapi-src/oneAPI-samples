#ifndef __BACKWARD_SUBSTITUTION_HPP__
#define __BACKWARD_SUBSTITUTION_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// utility classes
#include "ParallelCopyArray.hpp"
#include "UnrolledLoop.hpp"

#include "mvdr_complex.hpp"

using namespace sycl;

// SubmitBackwardSubstitutionKernel
// Accept an upper-right square triangular matrix U (in row order) and a
// number of y input vectors, and solve Ux = y for each input vector.
// U must be real valued on the diagonal, and a vector of the reciprocals
// of the diagonal values (UDiagRecipVectorInPipe) must also be supplied.
template <typename BackwardSubstitutionKernelName,  // Name to use for the
                                                    // Kernel

          size_t k_vector_size,    // Number of elements in each vector, also
                                   // number of rows and columns in the
                                   // square U matrix
          size_t k_unroll_factor,  // Factor by which to unroll the innermost
                                   // calculation loop, so k_unroll_factor
                                   // operations will be performed in
                                   // parallel on each clock.
          size_t k_num_y_vectors,  // Number of y vectors to solve for with
                                   // each new L matrix.

          typename UMatrixInPipe,  // Receive the upper-triangular matrix U.
                                   // Diagonal must be real valued.
                                   // Only upper right elements are sent,
                                   // (known zero elements are skipped).
                                   // Sent starting with first row.
                                   // Receive one complex float per read.
          typename UDiagRecipVectorInPipe,  // Receive 1/ the diagonal elements
                                            // of U. Receive one float per read.
          typename YVectorsInPipe,  // Receive the Y vectors.
                                    // Receive one complex float per read.

          typename XVectorsOutPipe  // Send X vectors.
                                    // Send one complex float per write.
          >
event SubmitBackwardSubstitutionKernel(queue& q) {
  // Template parameter checking
  static_assert(k_vector_size > 0, "k_vector_size must be greater than 0");
  static_assert(std::numeric_limits<short>::max() > k_vector_size,
                "k_vector_size must fit in a short");
  static_assert(k_num_y_vectors > 0, "k_num_y_vectors must be greater than 0");
  static_assert(std::numeric_limits<char>::max() > k_num_y_vectors,
                "k_num_y_vectors must fit in a char");
  static_assert(k_vector_size % k_unroll_factor == 0,
                "k_vector_size must be evenly divisible by k_unroll_factor");

  // this type represents the number of samples to be processed in parallel
  // grouping these into an array helps the compiler generate the desired local
  // memory configurations
  using CalcType = ParallelCopyArray<ComplexType, k_unroll_factor>;
  constexpr short kNumCalcTypePerVector = k_vector_size / k_unroll_factor;

  auto e = q.submit([&](handler& h) {
    h.single_task<BackwardSubstitutionKernelName>([=] {
      while (1) {
        CalcType u_matrix[k_vector_size][kNumCalcTypePerVector];
        float u_diag_recip_vector[k_vector_size];

        // number of non-zero elements in the U matrix
        // 1+2+3+...+n = (n)(n+1)/2
        constexpr int kNumUElements = (k_vector_size) * (k_vector_size + 1) / 2;

        // receive the U matrix and diagonal reciprocal vector from the pipes
        short col = 0;
        short row = 0;
        for (int i = 0; i < kNumUElements; i++) {
          u_matrix[col][row / k_unroll_factor][row % k_unroll_factor] =
              UMatrixInPipe::read();
          if (col == row) {
            u_diag_recip_vector[row] = UDiagRecipVectorInPipe::read();
          }
          col++;
          if (col == (short)k_vector_size) {
            row++;      // row counts 0..k_vector_size
            col = row;  // col counts 0..k_vector_size, 1..k_vector_size, ...
          }
        }

        for (char vector_num = 0; vector_num < (char)k_num_y_vectors;
             vector_num++) {
          // y_vector_intial contains the unmodified current y vector.  y_vector
          // is used during processing.  Splitting these two vectors allows
          // each to be implemented in a local memory with only one read and
          // one write port.
          CalcType y_vector_initial[kNumCalcTypePerVector];
          CalcType y_vector[kNumCalcTypePerVector];

          // This variable represents the value of y[col] for the next iteration
          // of the col loop (see below).  Storing this value here prevents
          // the need for an additional read from y_vector[], which would result
          // in a second read port for the y_vector[] memory system.
          ComplexType y_at_next_col_pos;

          // load a new y vector from the pipe
          for (short i = 0; i < (short)k_vector_size; i++) {
            ComplexType y_pipe_in = YVectorsInPipe::read();
            ;
            y_vector_initial[i / k_unroll_factor][i % k_unroll_factor] =
                y_pipe_in;
            if (i == (short)k_vector_size - 1) {
              y_at_next_col_pos = y_pipe_in;
            }
          }

          // calculate x for the current y vector
          ComplexType x_vector[k_vector_size];

          // iterate over all columns in U
          for (short col = k_vector_size - 1; col >= 0; col--) {
            // y_scaled = y[col] / L[col][col] (diagonal value of L)
            ComplexType y_scaled;
            y_scaled = y_at_next_col_pos * u_diag_recip_vector[col];

            // store the calculated result to the x_vector
            x_vector[col] = y_scaled;

            // iterate over all rows of U in the current column
            // unroll by a factor of k_unroll_factor (so perform k_unroll_factor
            // operations in parallel)
            // NO-FORMAT comments are for clang-format
            [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
            for (short i = 0; i < kNumCalcTypePerVector; i++) {
              // These variables are used to help the compiler infer the
              // desired memory organization.
              CalcType u_val, y_val, y_initial_val, y_current, y_new;
              short row[k_unroll_factor];

              UnrolledLoop<k_unroll_factor>([&](auto j) {
                // calculate current location within the vector
                row[j] = j + (i * (short)k_unroll_factor);

                // force u values to 0 for upper right of matrix (including
                // the diagonal)
                u_val[j] = u_matrix[col][i][j];
                if (row[j] >= col) u_val[j] = 0;

                // If we place the accesses to y_vector and y_vector_intial
                // inside the if/else statement, it results in a very strange
                // configuration of the y_vector memory system, so use these
                // intermediate variables.
                y_val[j] = y_vector[i][j];
                y_initial_val[j] = y_vector_initial[i][j];
                if (col == (short)k_vector_size - 1) {
                  y_current[j] = y_initial_val[j];
                } else {
                  y_current[j] = y_val[j];
                }

                // perform the calculation on y and store the result back into
                // y_vector
                y_new[j] = y_current[j] - y_scaled * u_val[j];
                y_vector[i][j] = y_new[j];

                // update y_at_next_col_pos for the next iteration of the col
                // loop
                if (row[j] == col - 1) {
                  y_at_next_col_pos = y_new[j];
                }
              });  // end of unrolled loop

            }  // end of for( v... )

          }  // end of for( col... )

          // write the result to the output pipe
          for (short i = 0; i < (short)k_vector_size; i++) {
            XVectorsOutPipe::write(x_vector[i]);
          }

        }  // end of for( vector_num ... )

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitBackwardSubstitutionKernel()

#endif  // ifndef __BACKWARD_SUBSTITUTION_HPP__
