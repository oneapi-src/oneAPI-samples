#ifndef __STREAMING_QRD_HPP__
#define __STREAMING_QRD_HPP__

#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/*
  Computes 1e-Is as a constexpr
*/
template <typename T, std::size_t... Is>
constexpr T negPow10(std::index_sequence<Is...> const &) {
  using unused = std::size_t[];

  T ret{1};

  (void)unused{0U, (ret /= 10, Is)...};

  return ret;
}

/*
  This function implements the QR iteration method to find the Eigen values
  and vectors of the input square matrices.
  In order to reduce the number of iterations to perform, the Wilkinson shift
  is applied at each iteration.

  Each matrix (input and output) are represented in a column wise (transposed).

  Then input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,       // The datatype for the computation
          int size,         // Number of rows/columns in the A matrices
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
          int zero_threshold_1e,     // Threshold from which we consider a
                                     // floating point value to be 0 (e.g. -4 ->
                                     // 10e-4)
          typename AIn,              // A matrix input pipe, receive pipe_size
                                     // elements from the pipe with each read
          typename EigenValuesOut,   // Eigen values output pipe, send 1
                                     // element to the pipe with each write
          typename EigenVectorsOut,  // Eigen vectors output pipe, send
                                     // pipe_size elements to the pipe with each
                                     // write.
          typename RankDeficientOut  // Outputs a 1 bit value per Eigen vector
                                     // matrix that is 1 if the input matrix
                                     // is considered rank deficient. In this
                                     // case, an Eigen value is 0, and the
                                     // associated Eigen vector is forced to 0.

          >
struct StreamingEigen {
  void operator()() const {
    // Functional limitations
    // static_assert(size >= 4,
    //               "only matrices of size 4x4 and over are supported");

    static_assert(zero_threshold_1e < 0,
                  "k_zero_threshold_1e must be negative");

    constexpr float k_zero_threshold =
        negPow10<float>(std::make_index_sequence<-zero_threshold_1e>{});

    // Type used to store the matrices in the QR decomposition compute loop
    using column_tuple = fpga_tools::NTuple<T, size>;

    // Fanout reduction factor for signals that fanout to size compute cores
    constexpr int kFanoutReduction = 8;
    // Number of signal replication required to cover all the size compute cores
    // given a kFanoutReduction factor
    constexpr int kBanksForFanout = (size % kFanoutReduction)
                                        ? (size / kFanoutReduction) + 1
                                        : size / kFanoutReduction;

    // Number of iterations performed without any dummy work added for the
    // triangular loop optimization
    constexpr int kVariableIterations = size - raw_latency;
    // Total number of dummy iterations
    static constexpr int kDummyIterations =
        raw_latency > size ? (size - 1) * size / 2 + (raw_latency - size) * size
                           : raw_latency * (raw_latency - 1) / 2;
    // Total number of iterations (including dummy iterations)
    static constexpr int kIterations =
        size + size * (size + 1) / 2 + kDummyIterations;

    // Size in bits of the "i" loop variable in the triangular loop
    // i starts from -1 as we are doing a full copy of the matrix read from the
    // pipe to a "compute" matrix before starting the decomposition
    constexpr int kIBitSize = fpga_tools::BitsForMaxValue<size + 1>() + 1;

    // j starts from i, so from -1 and goes up to size
    // So we need:
    // -> enough bits to encode size+1 for the positive iterations and
    //    the exit condition
    // -> one extra bit for the -1
    // But j may start below -1 if we perform more dummy iterations than the
    // number of size in the matrix.
    // In that case, we need:
    // -> enough bits to encode size+1 for the positive iterations and
    //    the exit condition
    // -> enough bits to encode the maximum number of negative iterations
    static constexpr int kJNegativeIterations =
        kVariableIterations < 0 ? -kVariableIterations : 1;
    static constexpr int kJBitSize =
        fpga_tools::BitsForMaxValue<size + 1>() +
        fpga_tools::BitsForMaxValue<kJNegativeIterations>();
    
    // Compute Eigen values and vectors as long as matrices are given as inputs
    while (1) {
      // ---------------------------------
      // -------- Load the matrix from DDR
      //----------------------------------

      // Break memories up to store 4 complex numbers (32 bytes) per bank
      constexpr short kBankwidth = pipe_size * sizeof(T);
      constexpr unsigned short kNumBanks = size / pipe_size;

      // When specifying numbanks for a memory, it must be a power of 2.
      // Unused banks will be automatically optimized away.
      constexpr short kNumBanksNextPow2 =
          fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      column_tuple a_load[size];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (size % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = size / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * size;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
          fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<T, pipe_size> pipe_read = AIn::read();

        int write_idx = li % kLoopIterPerColumn;
        int a_col_index = li / kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < size) {
                a_load[a_col_index].template get<k * pipe_size + t>() =
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

      // ------------------------------------------------
      // -------- Initialize matrices for the iteration
      //-------------------------------------------------

      T rq_matrix[size][size];
      T eigen_vectors_matrix[size][size];
      T eigen_vectors_matrix_output[size][size];

      // Vector to hold the computed Eigen values
      T eigen_values[size];

      // Initialize shift values

      // Compute the shift value of the current matrix
      T shift_value = 0;

      // First find where the shift should be applied
      // Start from the last submatrix
      int shift_row = size - 2;

      // At each iteration we are going to compute the shift as follows:
      // Take the submatrix:
      // [a b]
      // [b c]
      // where a and c are diagonal elements, and [a b] is on row shift_row
      // and compute the shift such as
      // mu = c - (sign(d)* b*b)/(abs(d) + sqrt(d*d + b*b))
      // where d = (a - c)/2
      T a = 0, b = 0, c = 0;

      // Because we won't know the actual shift_row location at the time of
      // retrieving these a b and c data, we will also collect the
      // submatrix one row above
      T a_above = 0, b_above = 0, c_above = 0;

      // ---------------------------------
      // -------- Start the QR iteration
      //----------------------------------
      bool continue_iterating = true;
      int iteration_count = 0;

      bool input_matrix_is_rank_deficient = false;
      while (continue_iterating) {
        // ---------------------------------------
        // -------- Compute the QR decomposition
        //----------------------------------------
        /*
        This code implements a OneAPI optimized variation of the following
        algorithm

        for i=0:n
          for j=max(i,1):n

            if(j==i)
              Q_i = a_i*ir
            else
              if(i>=0)
                a_j = a_j - s[j]*a_i

              if j=i+1
                pip1         = <a_{i+1},a_{i+1}>
                ir           = 1/sqrt(pip1)
                R_{i+1,i+1}  = sqrt(pip1)
              else
                p            = <a_{i+1}, a_j>
                s[j]         = p/pip1
                R_{i+1,j}    = p*ir


        Where:
        -> X_i represents the column i of the matrix X
        -> <x,y> represents the dot product of the vectors x and y
        */

        // Matrices used by the QR algorithm to:
        // - store intermediate results
        // - store the Q matrix
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        column_tuple a_compute[size],
            q_matrix[size];

        // Matrix to hold the R matrix
        [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
        T r_matrix[size][size];

        // a local copy of a_{i+1} that is used across multiple j iterations
        // for the computation of pip1 and p
        T a_ip1[size];
        // a local copy of a_ip1 that is used across multiple j iterations
        // for the computation of a_j
        T a_i[size];
        // Depending on the context, will contain:
        // -> -s[j]: for all the iterations to compute a_j
        // -> ir: for one iteration per j iterations to compute Q_i
        [[intel::fpga_memory]]        // NO-FORMAT: Attribute
        [[intel::private_copies(2)]]  // NO-FORMAT: Attribute
        T s_or_ir[size];

        T pip1, ir;

        // Initialization of the i and j variables for the triangular loop
        ac_int<kIBitSize, true> i = -1;
        ac_int<kJBitSize, true> j = 0;

        // We keep track of the value of the current column
        // If it's a 0 vector, then we need to skip the iteration
        // This will result in size in Q being set to 0
        // This occurs when the input matrix have linearly dependent columns
        bool projection_is_zero = false;

        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        [[intel::ivdep(raw_latency)]]      // NO-FORMAT: Attribute
        for (int s = 0; s < kIterations; s++) {
          // Pre-compute the next values of i and j
          ac_int<kIBitSize, true> next_i;
          ac_int<kJBitSize, true> next_j;
          if (j == size - 1) {
            // If i reached an index at which the j inner loop don't have
            // enough time to write its result for the next i iteration,
            // some "dummy" iterations are introduced
            next_j = (kVariableIterations > i)
                         ? ac_int<kJBitSize, true>{i + 1}
                         : ac_int<kJBitSize, true>{kVariableIterations};
            next_i = i + 1;
          } else {
            next_j = j + 1;
            next_i = i;
          }

          // Two matrix size for partial results.
          T col[size];
          T col1[size];

          // Current value of s_or_ir depending on the value of j
          // It is replicated kFanoutReduction times to reduce fanout
          T s_or_ir_j[kBanksForFanout];

          // All the control signals are precomputed and replicated
          // kFanoutReduction times to reduce fanout
          bool j_eq_i[kBanksForFanout], i_gt_0[kBanksForFanout],
              i_ge_0_j_ge_i[kBanksForFanout], j_eq_i_plus_1[kBanksForFanout],
              i_lt_0[kBanksForFanout], j_ge_0[kBanksForFanout];

          fpga_tools::UnrolledLoop<kBanksForFanout>([&](auto k) {
            i_gt_0[k] = sycl::ext::intel::fpga_reg(i > 0);
            i_lt_0[k] = sycl::ext::intel::fpga_reg(i < 0);
            j_eq_i[k] = sycl::ext::intel::fpga_reg(j == i);
            j_ge_0[k] = sycl::ext::intel::fpga_reg(j >= 0);
            i_ge_0_j_ge_i[k] = sycl::ext::intel::fpga_reg(i >= 0 && j >= i);
            j_eq_i_plus_1[k] = sycl::ext::intel::fpga_reg(j == i + 1);
            if (j >= 0) {
              s_or_ir_j[k] = sycl::ext::intel::fpga_reg(s_or_ir[j]);
            }
          });

          // Preload col and a_i with the correct data for the current iteration
          // These are going to be use to compute the dot product of two
          // different size of the input matrix.
          fpga_tools::UnrolledLoop<size>([&](auto k) {
            // find which fanout bank this unrolled iteration is going to use
            constexpr auto fanout_bank_idx = k / kFanoutReduction;

            // Load col with the current column of matrix A.
            // At least one iteration of the outer loop i is required
            // for the "working copy" a_compute to contain data.
            // If no i iteration elapsed, we must read the column of
            // matrix A directly from the a_load; col then contains a_j

            if (i_gt_0[fanout_bank_idx] && j_ge_0[fanout_bank_idx]) {
              col[k] = a_compute[j].template get<k>();
            }
            // Using an else statement makes the compiler throw an
            // inexplicable warning when using non complex types:
            // "Compiler Warning: Memory instruction with unresolved
            // pointer may lead to bad QoR."
            if (!i_gt_0[fanout_bank_idx] && j_ge_0[fanout_bank_idx]) {
              if (iteration_count == 0) {
                col[k] = a_load[j].template get<k>();
              } else {
                T to_sub = k == j ? shift_value : T{0};
                col[k] = rq_matrix[j][k] - to_sub;
              }
            }

            // Load a_i for reuse across j iterations
            if (i_lt_0[fanout_bank_idx]) {
              a_i[k] = 0;
            } else if (j_eq_i[fanout_bank_idx]) {
              a_i[k] = col[k];
            }
          });

          fpga_tools::UnrolledLoop<size>([&](auto k) {
            // find which fanout bank this unrolled iteration is going to use
            constexpr auto fanout_bank_idx = k / kFanoutReduction;

            // Depending on the iteration this code will compute either:
            // -> If i=j, a column of Q: Q_i = a_i*ir
            //    In that case, no term is added to the mult_add construct
            // -> If i!=j, an updated column of a: a_j - s[j]*a_i
            //    There is a special case if i<0 where a_j is unmodified
            //    but the i iteration is still required to fill ir and s
            //    for subsequent iterations
            auto prod_lhs = a_i[k];
            auto prod_rhs =
                i_lt_0[fanout_bank_idx] ? T{0.0} : s_or_ir_j[fanout_bank_idx];
            auto add = j_eq_i[fanout_bank_idx] ? T{0.0} : col[k];
            col1[k] = prod_lhs * prod_rhs + add;

            // Store Q_i in q_matrix and the modified a_j in a_compute
            // To reduce the amount of control, q_matrix and a_compute
            // are both written to for each iteration of i>=0 && j>=i
            // In fact:
            // -> q_matrix could only be written to at iterations i==j
            // -> a_compute could only be written to at iterations
            //    j!=i && i>=0
            // The extra writes are harmless as the locations written to
            // are either going to be:
            // -> overwritten for the matrix Q (q_matrix)
            // -> unused for the a_compute
            if (i_ge_0_j_ge_i[fanout_bank_idx] && j_ge_0[fanout_bank_idx]) {
              q_matrix[j].template get<k>() = col1[k];
              a_compute[j].template get<k>() = col1[k];
            }

            // Store a_{i+1} for subsequent iterations of j
            if (j_eq_i_plus_1[fanout_bank_idx]) {
              a_ip1[k] = col1[k];
            }
          });

          // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
          T p_ij{0.0};
          fpga_tools::UnrolledLoop<size>(
              [&](auto k) { p_ij += col1[k] * a_ip1[k]; });

          bool projection_is_zero_local = false;

          // Compute pip1 and ir based on the results of the dot product
          if (j == i + 1) {
            // If the projection is 0, we won't be able to divide by pip1 (=p_ij)
            projection_is_zero_local = p_ij < k_zero_threshold*k_zero_threshold;
            projection_is_zero |= projection_is_zero_local;
            ir = sycl::rsqrt(p_ij);
            pip1 = p_ij;
          }

          // Compute the value of -s[j]
          T s_j;
          if (projection_is_zero_local) {
            s_j = T{0};
          } else {
            s_j = -p_ij / pip1;
          }

          // j may be negative if the number of "dummy" iterations is
          // larger than the matrix size
          if (j >= 0) {
            s_or_ir[j] = j == i + 1 ? ir : s_j;
          }

          // Compute the R_{i+1,i+1} or R_{i+1,j}
          T r_ip1j = j == i + 1 ? sycl::sqrt(pip1) : ir * p_ij;

          // Write the computed R value when j is not a "dummy" iteration
          if ((j >= i + 1) && (i + 1 < size)) {
            r_matrix[i + 1][j] = r_ip1j;
          }

          // Update loop indexes
          if (j == (size - 1)) {
            // If i reached an index at which the j inner loop doesn't have
            // enough time to write its result for the next i iteration,
            // some "dummy" iterations are introduced
            j = (kVariableIterations > i)
                    ? ac_int<kJBitSize, true>{i + 1}
                    : ac_int<kJBitSize, true>{kVariableIterations};
            i = i + 1;
          } else {
            j = j + 1;
          }

        }  // end of for:s

        // ---------------------------------------------------
        // -------- Compute R*Q and update the Eigen vectors
        //----------------------------------------------------

        bool row_is_zero = true;
        T rq_matrix_copy[size][size];
        for (int row = size-1; row >= 0; row--) {
          T eigen_vectors_row[size];
          fpga_tools::UnrolledLoop<size>([&](auto t) {
            if (iteration_count == 0) {
              eigen_vectors_row[t] = t == row ? 1 : 0;
            } else {
              eigen_vectors_row[t] = eigen_vectors_matrix[row][t];
            }
          });

          for (int column = 0; column < size; column++) {
            T dot_product_rq = 0;
            T dot_product_eigen_vectors = 0;
            fpga_tools::UnrolledLoop<size>([&](auto t) {
              T r_elem = r_matrix[row][t];
              T r = row > t ? T{0} : r_elem;
              dot_product_rq += r * q_matrix[column].template get<t>();
              dot_product_eigen_vectors +=
                  eigen_vectors_row[t] * q_matrix[column].template get<t>();
            });

            T rq_value =
                row == column ? dot_product_rq + shift_value : dot_product_rq;
            if (column > row) {
              rq_matrix[row][column] = rq_matrix_copy[column][row];

            } else if (row <= (shift_row + 1) && column <= (shift_row + 1)) {
              rq_matrix[row][column] = rq_value;
              rq_matrix_copy[row][column] = rq_value;
            }
            eigen_vectors_matrix[row][column] = dot_product_eigen_vectors;
            eigen_vectors_matrix_output[column][row] =
                dot_product_eigen_vectors;

            if ((row == shift_row) && (column == shift_row)) {
              a = rq_value;
              c_above = rq_value;
            } else if ((row == shift_row - 1) && (column == (shift_row - 1))) {
              a_above = rq_value;
            } else if ((row == shift_row) && (column == (shift_row + 1))) {
              b = rq_value;
            } else if ((row == shift_row - 1) && (column == shift_row)) {
              b_above = rq_value;
            } else if ((row == shift_row + 1) && (column == shift_row + 1)) {
              c = rq_value;
            }

            if ((row > column) && (row == shift_row + 1)) {
              row_is_zero &= fabs(rq_value) < k_zero_threshold;
            }

            if (row == column) {
              eigen_values[row] = rq_value;
            }
          }
        }

        // Compute the shift value

        T d = (a - c) / 2;
        T b_squared = b * b;
        T d_squared = d * d;
        T b_squared_signed = d < 0 ? -b_squared : b_squared;
        T shift_value_current_shift_row =
            c - b_squared_signed / (abs(d) + sqrt(d_squared + b_squared));

        T d_above = (a_above - c_above) / 2;
        T b_squared_above = b_above * b_above;
        T d_squared_above = d_above * d_above;
        T b_squared_signed_above =
            d_above < 0 ? -b_squared_above : b_squared_above;
        T shift_value_above =
            c_above -
            b_squared_signed_above /
                (abs(d_above) + sqrt(d_squared_above + b_squared_above));

        if ((shift_row < 0) || (row_is_zero && (shift_row == 0))) {
          shift_value = 0;
        } else {
          shift_value =
              row_is_zero ? shift_value_above : shift_value_current_shift_row;
        }

        shift_value *= 0.99;

        if (row_is_zero) {
          shift_row--;
        }

        input_matrix_is_rank_deficient |= projection_is_zero;

        if (shift_row == -2) {
          continue_iterating = false;
        }

        iteration_count++;

      }  // end if while(continue_iterating)


      // -----------------------------------------------------------------
      // -------- Sort the Eigen Values/Vectors by weight
      //------------------------------------------------------------------

      // Instead of sorting the values and vectors, we sort the order in which
      // we are going to traverse the outputs when writing to the pipe
      int sorted_indexes[size];

      // We are going to traverse the Eigen values to find the current maximum
      // value. We use a mask to remember which values have already been used.
      ac_int<size, false> mask = 0;

      for (int current_index = 0; current_index < size; current_index++) {
        int sorted_index = 0;
        T max_value = -1;
        for (int k = size - 1; k >= 0; k--) {
          // Make sure the current Eigen value was not used already
          if (mask[k] == 0) {
            // Get the Eigen value
            T eigen_value = eigen_values[k];

            // Get its absolute value
            T absolute_value = eigen_value < 0 ? -eigen_value : eigen_value;

            // Check if the current Eigen value is larger (in absolute terms)
            // than the current max
            if (absolute_value > max_value) {
              max_value = absolute_value;
              sorted_index = k;
            }
          }
        }

        sorted_indexes[current_index] = sorted_index;
        mask[sorted_index] = 0b1;
      }

      // -----------------------------------------------------------------
      // -------- Write the Eigen values and vectors to the output pipes
      //------------------------------------------------------------------

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int k = 0; k < size; k++) {
        EigenValuesOut::write(eigen_values[sorted_indexes[k]]);
      }
      
      ac_int<1, false> to_pipe = input_matrix_is_rank_deficient ? 1 : 0;
      RankDeficientOut::write(to_pipe);

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int column_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<T, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < size) {
              pipe_write.template get<k>() =
                  get[t] ? eigen_vectors_matrix_output
                               [sorted_indexes[li / kLoopIterPerColumn]]
                               [t * pipe_size + k]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        EigenVectorsOut::write(pipe_write);
      }  // end for:li

    }    // end of while(1)
  }      // end of operator
};       // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_QRD_HPP__ */