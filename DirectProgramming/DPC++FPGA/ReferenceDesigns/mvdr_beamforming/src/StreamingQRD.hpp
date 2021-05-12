#ifndef __STREAMING_QRD_HPP__
#define __STREAMING_QRD_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// utility classes
#include "Tuple.hpp"
#include "UnrolledLoop.hpp"

#include "mvdr_complex.hpp"

using namespace sycl;

// helper functions
// computes 2^n where 'n' is a compile time constant
template <typename T>
static constexpr T Pow2(T n) {
  return T(1) << n;
}
// base-2 logarithm
template <typename T>
static constexpr T Log2(T n) {
  return ((n < 2) ? T(0) : T(1) + Log2(n / 2));
}
// round up Log2
template <typename T>
static constexpr T CeilLog2(T n) {
  return ((n == 1) ? T(0) : Log2(n - 1) + T(1));
}

// SubmitStreamingQRDKernel
// Accept an input matrix one column at a time from an array of pipes.  Perform
// Q R Decomposition on the array and send the result out through pipes.
template <typename StreamingQRDKernelName,  // Name to use for the Kernel

          size_t k_min_inner_loop_iterations,  // Minimum number of inner loop
                                               // iterations to achieve an outer
                                               // loop II of 1.  This value will
                                               // have to be tuned for optimal
                                               // performance.  Refer to the
                                               // Triangular Loop design pattern
                                               // tutorial.

          size_t k_a_num_rows,      // Number of rows in the incoming A matrix
          size_t k_a_num_cols,      // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          size_t k_pipe_width,      // number of elements read/written
                                    // (wrapped in NTuple) from/to pipes
          typename AMatrixInPipe,   // A matrix input, receive a full column
                                    // of complex numbers with each read,
                                    // wrapped in NTuple
          typename QMatrixOutPipe,  // Q output pipe, send a full column
                                    // of complex numbers with each write.
                                    // Column 0 is sent first, k_a_num_cols-1
                                    // is sent last
          typename RMatrixOutPipe,  // R output pipe.  Send one complex number
                                    // per write.  Only upper-right elements
                                    // of R are sent.  Sent in row order,
                                    // starting with row 0.
          typename RDiagRecipVectorOutPipe  // 1 / the value of each diagonal
                                            // entry of the R matrix.  Diagonals
                                            // of R are real valued, so only
                                            // send one float per write
          >
event SubmitStreamingQRDKernel(queue& q) {
  // Template parameter checking
  static_assert(std::numeric_limits<short>::max() > k_a_num_cols,
                "k_a_num_cols must fit in a short");
  static_assert(k_a_num_rows >= k_a_num_cols,
                "k_a_num_rows must be greater than or equal to k_a_num_cols");
  static_assert(std::numeric_limits<short>::max() > k_a_num_rows,
                "k_a_num_rows must fit in a short");
  static_assert(k_a_num_rows % k_pipe_width == 0,
                "k_a_num_rows must be evenly divisible by k_pipe_width");

  using PipeType = NTuple<ComplexType, k_pipe_width>;

  auto e = q.submit([&](handler& h) {
    h.single_task<StreamingQRDKernelName>([=] {
      // Constants used to implement the triagular loop structure of the design.
      // See the tutorial on triangular loop optimization for more details.

      // Calculation for the special case when the minimum number of iterations
      // is greater than the total number of columns to be processed.
      constexpr short kMinIterationsMinusCols =
          (k_min_inner_loop_iterations > k_a_num_cols)
              ? ((short)k_min_inner_loop_iterations - (short)k_a_num_cols)
              : 0;

      // Calculate the total number of iterations in the trangular loop,
      // including 'dummy' iterations
      constexpr int kNumIterations =
          // initial pass to load all columns into the local array
          k_a_num_cols +
          kMinIterationsMinusCols
          // 'real work' iterations = n + n-1 + ... + 1 = (n+1)(n)/2
          + (((k_a_num_cols + 1) * k_a_num_cols) / 2)
          // 'dummy' iterations = 1 + 2 + ... + k-1 = (k)(k-1)/2
          + ((k_min_inner_loop_iterations * (k_min_inner_loop_iterations - 1)) /
             2)
          // 'dummy' iterations we don't really have to do when min iterations
          // is greater than the number of columns
          - ((kMinIterationsMinusCols * (kMinIterationsMinusCols - 1)) / 2);

      while (1) {
        // Break memories up to store 4 complex numbers (32 bytes) per bank
        constexpr short kNumElementsPerBank = 4;
        constexpr short kBankwidth = kNumElementsPerBank * sizeof(ComplexType);
        constexpr short kNumBanks = k_a_num_rows / kNumElementsPerBank;

        // When specifying numbanks for a memory, it must be a power of 2.
        // Unused banks will be automatically optimized away.
        constexpr short kNumBanksNextPow2 = Pow2(CeilLog2(kNumBanks));

        // define a type that contains an entire column
        using AColumn = NTuple<ComplexType, k_a_num_rows>;

        // Three copies of the full matrix, so that each matrix has a single
        // load and a single store.
        // a_matrix_in is the initial matrix received from the pipe
        // a_matrix is used and modified during calculations
        // q_matrix is a copy of a_matrix and is used to send the final output
        // The compiler has difficulty automatically figuring out an optimal
        // configuration for these memories, so force all relevant parameters.
        // NO-FORMAT comments are for clang-format
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        AColumn a_matrix_in[k_a_num_cols];
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        AColumn a_matrix[k_a_num_cols];
        AColumn q_matrix[k_a_num_cols];

        // storage for output values
        constexpr size_t kNumRValues = k_a_num_cols * (k_a_num_cols + 1) / 2;
        ComplexType r_matrix[kNumRValues];
        short r_index = 0;
        float r_diag_recip_vector[k_a_num_cols];

        // receive a new A matrix from the input pipe
        // Each read contains k_pipe_width samples from the same column, then
        // the next read contains samples from the next column.
        // row/col loops have been manually coalesced
        for (short i = 0; i < (short)k_a_num_rows * (short)k_a_num_cols /
                                  (short)k_pipe_width;
             i++) {
          PipeType data_in = AMatrixInPipe::read();
          short col = i % (short)k_a_num_cols;
          short write_row_group = i / (short)k_a_num_cols;
          UnrolledLoop<k_a_num_rows / k_pipe_width>([&](auto row_group) {
            UnrolledLoop<k_pipe_width>([&](auto element) {
              constexpr short row = row_group * k_pipe_width + element;
              if (write_row_group == row_group) {
                a_matrix_in[col].template get<row>() =
                    data_in.template get<element>();
              }

              // Delay data signal to create a vine to lower signal fanout
              data_in.template get<element>() =
                  INTEL::fpga_reg(data_in.template get<element>());
            });

            write_row_group = INTEL::fpga_reg(write_row_group);
          });
        }

        // Intermediate variables used during calculations
        AColumn vector_ai;
        AColumn vector_ti;
        ComplexType s_or_i[k_a_num_cols];
        float p_ii_real;
        float i_r_ii_real;

        // loop iteration variables
        short i = -1;  // iterates from -1 to k_a_num_cols
        short j =      // iterates up to k_a_num_cols multiple times
            (((short)k_a_num_cols - (short)k_min_inner_loop_iterations) < 0)
                ? ((short)k_a_num_cols - (short)k_min_inner_loop_iterations)
                : 0;
        // version of j that is forced to never be negative, so it can safely
        // be used as an array index
        short j_nonneg = 0;

        // Calculate Q and R by iterating over A
        // NO-FORMAT comments are for clang-format
        [[intel::initiation_interval(1)]]              // NO-FORMAT: Attribute
        [[intel::ivdep(k_min_inner_loop_iterations)]]  // NO-FORMAT: Attribute
        for (int s = 0; s < kNumIterations; s++) {
          AColumn vector_t;

          // These variables provide pre-calculated results with replication
          // to reduce fanout problems.
          bool j_eq_i[kNumBanks];
          bool i_gt_0[kNumBanks];
          bool i_ge_0_j_ge_i[kNumBanks];
          bool j_eq_i_plus_1[kNumBanks];
          bool i_lt_0[kNumBanks];
          ComplexType sori[kNumBanks];

          UnrolledLoop<kNumBanks>([&](auto k) {
            j_eq_i[k] = INTEL::fpga_reg(j == i);
            i_gt_0[k] = INTEL::fpga_reg(i > 0);
            i_ge_0_j_ge_i[k] = INTEL::fpga_reg(i >= 0 && j >= i);
            j_eq_i_plus_1[k] = INTEL::fpga_reg(j == i + 1);
            i_lt_0[k] = INTEL::fpga_reg(i < 0);
            sori[k] = INTEL::fpga_reg(s_or_i[j_nonneg]);
          });

          // fetch data from a_matrix_in or a_matrix, based on value of i
          // Use of fpga_reg here is a workaround to prevent the compiler from
          // inferring some very complicated arbitrated local memory systems.
          UnrolledLoop<k_a_num_rows>([&](auto row) {
            // load vector_t from a_matrix_in
            vector_t.template get<row>() =
                INTEL::fpga_reg(a_matrix_in[j_nonneg].template get<row>());

            // overwrite vector_t from a_matrix if i > 0
            if (i_gt_0[row / kNumElementsPerBank]) {
              vector_t.template get<row>() =
                  INTEL::fpga_reg(a_matrix[j_nonneg].template get<row>());
            }

            // store the 'unaltered' column in vector_ai if j == i
            if (j_eq_i[row / kNumElementsPerBank]) {
              vector_ai.template get<row>() = vector_t.template get<row>();
            }
          });

          // perform calculations on the current column of data, and store
          // the result back to a_matrix (and q_matrix).
          UnrolledLoop<k_a_num_rows>([&](auto row) {
            // calculate the new vector_t
            ComplexType sori_or_0 = i_lt_0[row / kNumElementsPerBank]
                                        ? 0
                                        : sori[row / kNumElementsPerBank];
            ComplexType vector_ai_times_conj_sori_or_0 =
                vector_ai.template get<row>() * sori_or_0.conj();
            ComplexType vector_t_or_0 = j_eq_i[row / kNumElementsPerBank]
                                            ? 0
                                            : vector_t.template get<row>();
            vector_t.template get<row>() =
                vector_ai_times_conj_sori_or_0 + vector_t_or_0;

            // update the values in the A matrix (and its copy q_matrix)
            if (i_ge_0_j_ge_i[row / kNumElementsPerBank]) {
              a_matrix[j_nonneg].template get<row>() =
                  vector_t.template get<row>();
              q_matrix[j_nonneg].template get<row>() =
                  vector_t.template get<row>();
            }

            if (j_eq_i_plus_1[row / kNumElementsPerBank]) {
              vector_ti.template get<row>() = vector_t.template get<row>();
            }
          });

          ComplexType p_ij = 0;
          UnrolledLoop<k_a_num_rows>([&](auto row) {
            p_ij += vector_t.template get<row>() *
                    vector_ti.template get<row>().conj();
          });

          if (j == i + 1) {
            p_ii_real = p_ij.real();
            i_r_ii_real = rsqrt(p_ij.real());
          }

          ComplexType s_ij;
          s_ij.set_r(0.0f - (p_ij.real() / p_ii_real));
          s_ij.set_i(p_ij.imag() / p_ii_real);

          if (j >= 0) {
            if (j == i + 1) {
              s_or_i[j_nonneg] = i_r_ii_real;
            } else {
              s_or_i[j_nonneg] = s_ij;
            }
          }

          // calculate 1/R value
          // r values on the diagonal are real valued, and this number is only
          // used on the diagonal, so just take 1 / real(R)
          float r_ii_diag_recip;
          r_ii_diag_recip = sycl::rsqrt(p_ii_real);

          // calculate R value
          ComplexType r_ii;
          // On the diagonal, R is real valued
          if (j == i + 1) {
            r_ii = sycl::sqrt(p_ii_real);
          } else {
            r_ii = p_ij * i_r_ii_real;
          }

          // store the r and 1/r values when they are ready
          if (j >= i + 1 && i + 1 < k_a_num_cols) {
            r_matrix[r_index] = r_ii;
            r_index++;
          }
          if (j == i + 1 && i + 1 < k_a_num_cols) {
            r_diag_recip_vector[j_nonneg] = r_ii_diag_recip;
          }

          if (j == (short)k_a_num_cols - 1) {
            constexpr short tmp =
                (short)k_a_num_cols - (short)k_min_inner_loop_iterations;
            j = (tmp > i) ? (i + 1) : tmp;
            i++;
          } else {
            j++;
          }
          j_nonneg = j < 0 ? 0 : j;

        }  // for( s < kNumIterations )

        // write the r and r_diag_recip values out to the appropriate pipes
        for (int i = 0; i < kNumRValues; i++) {
          RMatrixOutPipe::write(r_matrix[i]);

          if (i < k_a_num_cols) {
            RDiagRecipVectorOutPipe::write(r_diag_recip_vector[i]);
          }
        }

        // this is where the A matrix would be written out to a pipe
        // using q_matrix, but we don't need this functionality in MVDR so
        // skipping this

      }  // end of while(1) - main processing loop for QRD kernel
    });  // end of h.single_task
  });    // end of q.submit

  return e;
}

#endif  // ifndef __STREAMING_QRD_HPP__
