#ifndef __FORWARD_SUBSTITUTION_HPP__
#define __FORWARD_SUBSTITUTION_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// utility classes
#include "ParallelCopyArray.hpp"
#include "UnrolledLoop.hpp"

#include "mvdr_complex.hpp"

using namespace sycl;

// SubmitForwardSubstitutionKernel
// Accept a square lower triangular matrix L (in column order) and a
// number of y input vectors, and solve Lx = y for each input vector.
// L must be real valued on the diagonal, and a vector of the reciprocals
// of the diagonal values (LDiagRecipVectorInPipe) must also be supplied.
template <typename ForwardSubstitutionKernelName,  // Name to use for the Kernel

          size_t k_vector_size,    // Number of elements in each vector, also
                                   // number of rows and columns in the
                                   // square L matrix
          size_t k_unroll_factor,  // Factor by which to unroll the innermost
                                   // calculation loop, so k_unroll_factor
                                   // operations will be performed in
                                   // parallel on each clock.
          size_t k_num_y_vectors,  // Number of y vectors to solve for with
                                   // each new L matrix.

          typename LMatrixInPipe,  // Receive the lower-triangular matrix L.
                                   // Diagonal must be real valued.
                                   // Only lower left elements are sent,
                                   // (known zero elements are skipped).
                                   // Sent starting with first column.
                                   // Receive one complex float per read.
          typename LDiagRecipVectorInPipe,  // Receive 1/ the diagonal elements
                                            // of L. Receive one float per read.
          typename YVectorsInPipe,  // Receive the Y vectors.
                                    // Receive one complex float per read.
          typename UpdateYInPipe,   // A valid read from this pipe indicates
                                    // a full set of vectors are ready on the
                                    // YVectorsInPipe.
                                    // Accept a single bool per read

          typename YVectorsOutPipe,  // Forward the Y vectors used to calculate
                                     // each X vector to downstream kernels.
                                     // Send one complex float per write.
          typename XVectorsOutPipe   // Send X vectors.
                                     // Send one complex float per write.
          >
event SubmitForwardSubstitutionKernel(queue& q) {
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
    h.single_task<ForwardSubstitutionKernelName>([=] {
      // do not proceed until the first set of y vectors are available
      // Data in this pipe is always true, so we know update_y_vectors will be
      // set to true here.  Doing this assignment creates a memory dependency
      // to ensure this pipe read can't be re-ordered relative to reads from
      // YVectorsInPipe
      bool update_y_vectors = UpdateYInPipe::read();

      CalcType y_vectors[k_num_y_vectors][kNumCalcTypePerVector];

      while (1) {
        CalcType l_matrix[k_vector_size][kNumCalcTypePerVector];
        float l_diag_recip_vector[k_vector_size];

        // number of non-zero elements in the L matrix
        // 1+2+3+...+n = (n)(n+1)/2
        constexpr int kNumLElements = (k_vector_size) * (k_vector_size + 1) / 2;

        // number of elements in the set of y vectors
        constexpr int kNumYElements = k_vector_size * k_num_y_vectors;

        constexpr int kLoadLoopIterations =
            (kNumLElements > kNumYElements) ? kNumLElements : kNumYElements;

        // receive the L matrix and diagonal reciprocal vector from the pipes
        // receive new y_vectors if they are available
        // This odd loop is a fusion of two loops with different trip counts.
        short col = 0, row = 0, i = 0, j = 0;
        unsigned char vector_num = 0;
        for (int iteration = 0; iteration < kLoadLoopIterations; iteration++) {
          // Load the L and LDiagRecip values
          if (iteration < kNumLElements) {
            l_matrix[col][row / k_unroll_factor][row % k_unroll_factor] =
                LMatrixInPipe::read();
            if (col == row) {
              l_diag_recip_vector[row] = LDiagRecipVectorInPipe::read();
            }

            row++;

            if (row == (short)k_vector_size) {
              col++;      // col counts 0..k_vector_size
              row = col;  // row counts 0..k_vector_size, 1..k_vector_size, ...
            }
          }

          // load the Y values
          if ((iteration < kNumYElements) && (update_y_vectors)) {
            // non-blocking read is safe here, we know data is available
            bool temp;
            y_vectors[vector_num][i][j] = YVectorsInPipe::read(temp);

            j++;  // j counts 0..k_unroll_factor

            if (j == (short)k_unroll_factor) {
              j = 0;
              i++;  // i counts 0..kNumCalcTypePerVector

              if (i == kNumCalcTypePerVector) {
                i = 0;
                vector_num++;  // vector_num counts 0..k_num_y_vectors
              }
            }
          }
        }  // end of for(i...)

        // Loop through all the y vectors
        for (unsigned char vector_num = 0;
             vector_num < (unsigned char)k_num_y_vectors; vector_num++) {
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

          // Load a y vector from local storage
          for (short i = 0; i < kNumCalcTypePerVector; i++) {
            CalcType y_elements;

            UnrolledLoop<k_unroll_factor>([&](auto j) {
              y_elements[j] = y_vectors[vector_num][i][j];
              y_vector_initial[i][j] = y_elements[j];
              if (i == 0 && j == 0) {
                // initialize the first value of y_at_next_col_pos
                y_at_next_col_pos = y_elements[0];
              }
            });
          }

          // calculate x for the current y vector
          ComplexType x_vector[k_vector_size];

          // iterate over all columns in L
          for (short col = 0; col < k_vector_size; col++) {
            // y_scaled = y[col] / L[col][col] (diagonal value of L)
            auto y_scaled = y_at_next_col_pos * l_diag_recip_vector[col];

            // store the calculated result to the x_vector
            x_vector[col] = y_scaled;

            // iterate over all rows of L in the current column
            // unroll by a factor of k_unroll_factor (so perform k_unroll_factor
            // operations in parallel)
            // NO-FORMAT comments are for clang-format
            [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
            for (short i = 0; i < kNumCalcTypePerVector; i++) {
              // These variables are used to help the compiler infer the
              // desired memory organization.
              CalcType l_val, y_val, y_initial_val, y_current, y_new;
              short row[k_unroll_factor];

              UnrolledLoop<k_unroll_factor>([&](auto j) {
                // calculate current location within the vector
                row[j] = j + (i * (short)k_unroll_factor);

                // force l values to 0 for upper right of matrix (including
                // the diagonal)
                l_val[j] = l_matrix[col][i][j];
                if (row[j] <= col) l_val[j] = 0;

                // If we place the accesses to y_vector and y_vector_intial
                // inside the if/else statement, it results in a very strange
                // configuration of the y_vector memory system, so use these
                // intermediate variables.
                y_val[j] = y_vector[i][j];
                y_initial_val[j] = y_vector_initial[i][j];
                if (col == 0) {
                  y_current[j] = y_initial_val[j];
                } else {
                  y_current[j] = y_val[j];
                }

                // perform the calculation on y and store the result back into
                // y_vector
                y_new[j] = y_current[j] - y_scaled * l_val[j].conj();
                y_vector[i][j] = y_new[j];

                // update y_at_next_col_pos for the next iteration of the col
                // loop
                if (row[j] == col + 1) {
                  y_at_next_col_pos = y_new[j];
                }
              });  // end of unrolled loop

            }  // end of for( i... )

          }  // end of for( col... )

          // write the result to the output pipe
          for (short i = 0; i < (short)k_vector_size; i++) {
            XVectorsOutPipe::write(x_vector[i]);
            YVectorsOutPipe::write(
                y_vector_initial[i / k_unroll_factor][i % k_unroll_factor]);
          }

        }  // end of for( vector_num ... )

        // Determine if a new set of Y vectors are available with a non-blocking
        // read from the UpdateYInPipe pipe.
        // The value read from the pipe is discarded, any data in this pipe
        // indicates a new set of vectors are ready
        (void)UpdateYInPipe::read(update_y_vectors);

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitForwardSubstitutionrKernel()

#endif  // ifndef __FORWARD_SUBSTITUTION_HPP__
