#ifndef __STREAMING_QRD_HPP__
#define __STREAMING_QRD_HPP__

#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"


#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

namespace fpga_linalg {

/*
  QRD (QR decomposition) - Computes Q and R matrices such that A=QR where:
  - A is the input matrix
  - Q is a unitary/orthogonal matrix
  - R is an upper triangular matrix

  This function implements a OneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan
  Pasca.

  Each matrix (input and output) are represented in a column wise (transposed).

  Then input and output matrices are consumed/produced from/to pipes.
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
          int pipe_size,     // Number of elements read/write per pipe
                             // operation
          typename AIn,      // A matrix input pipe, receive pipe_size
                             // elements from the pipe with each read
          typename RQOut,     // Q matrix output pipe, send pipe_size
                             // elements to the pipe with each write
          typename QQOut      // R matrix output pipe, send pipe_size
                             // elements to the pipe with each write.
                             // Only upper-right elements of R are
                             // sent in row order, starting with row 0.
          >
struct StreamingQRD {
  void operator()() const {
    // Functional limitations
    static_assert(rows >= columns,
                  "only rectangular matrices with rows>=columns are supported");
    // static_assert(columns >= 4,
    //               "only matrices of size 4x4 and over are supported");

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

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<TT, rows>;
    using row_tuple = fpga_tools::NTuple<TT, columns>;

    // Number of upper-right elements in the R output matrix
    // constexpr int kMatrixSize = rows * rows;
    // constexpr int kMatrixBitSize = fpga_tools::BitsForMaxValue<kMatrixSize + 1>()+1;
    // Fanout reduction factor for signals that fanout to rows compute cores
    constexpr int kFanoutReduction = 8;
    // Number of signal replication required to cover all the rows compute cores
    // given a kFanoutReduction factor
    constexpr int kBanksForFanout = (rows % kFanoutReduction)
                                    ? (rows / kFanoutReduction) + 1
                                    : rows / kFanoutReduction;

    // Number of iterations performed without any dummy work added for the
    // triangular loop optimization
    constexpr int kVariableIterations = columns - raw_latency;
    // Total number of dummy iterations
    static constexpr int kDummyIterations =
        raw_latency > columns
            ? (columns - 1) * columns / 2 + (raw_latency - columns) * columns
            : raw_latency * (raw_latency - 1) / 2;
    // Total number of iterations (including dummy iterations)
    static constexpr int kIterations =
        columns + columns * (columns + 1) / 2 + kDummyIterations;

    // Size in bits of the "i" loop variable in the triangular loop
    // i starts from -1 as we are doing a full copy of the matrix read from the
    // pipe to a "compute" matrix before starting the decomposition
    constexpr int kIBitSize = fpga_tools::BitsForMaxValue<rows + 1>() + 1;

    // j starts from i, so from -1 and goes up to columns
    // So we need:
    // -> enough bits to encode columns+1 for the positive iterations and
    //    the exit condition
    // -> one extra bit for the -1
    // But j may start below -1 if we perform more dummy iterations than the
    // number of columns in the matrix.
    // In that case, we need:
    // -> enough bits to encode columns+1 for the positive iterations and
    //    the exit condition
    // -> enough bits to encode the maximum number of negative iterations
    static constexpr int kJNegativeIterations =
        kVariableIterations < 0 ? -kVariableIterations : 1;
    static constexpr int kJBitSize = fpga_tools::BitsForMaxValue<columns + 1>()
                          + fpga_tools::BitsForMaxValue<kJNegativeIterations>();





    // Compute QRDs as long as matrices are given as inputs
    while(1) {
      // Three copies of the full matrix, so that each matrix has a single
      // load and a single store.
      // a_load is the initial matrix received from the pipe
      // a_compute is used and modified during calculations
      // q_result is a copy of a_compute and is used to send the final output

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
      column_tuple a_load[columns], a_compute[columns], q_result[columns];

      // Contains the values of matrix R in a row by row
      // fashion, starting by row 0
      row_tuple r_matrix[rows];

            // Initial Identity Eigen vector matrix 
      row_tuple QQ_matrix[rows];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
                                  fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      // plceholder for releigh shift 
      TT R_shift;
      TT a_wilk, b_wilk, c_wilk;
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read = AIn::read();

        int write_idx = li % kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < rows) {
                a_load[li / kLoopIterPerColumn].template get<k * pipe_size
                                          + t>() = pipe_read.template get<t>();
              }

              if(li / kLoopIterPerColumn == columns -1 && k * pipe_size + t == rows -1){
                c_wilk = pipe_read.template get<t>();
              }

              if(li / kLoopIterPerColumn == columns -2 && k * pipe_size + t == rows -2){
                a_wilk = pipe_read.template get<t>();
              }

              if(li / kLoopIterPerColumn == columns -2 && k * pipe_size + t == rows -1){
                b_wilk = pipe_read.template get<t>();
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

      TT lamda = (a_wilk-c_wilk)/2.0;
      TT sign_lamda = (lamda > 0) - (lamda < 0);
      TT l_shift = c_wilk - (sign_lamda*b_wilk*b_wilk)/(fabs(lamda) + sqrt(lamda * lamda + b_wilk*b_wilk));

      R_shift = RELSHIFT ? c_wilk : l_shift;
      R_shift -= R_shift*SHIFT_NOISE; //SHIFT_NOISE;
  
      // size of Deflated matrix
      int kDM_size = rows;

      // int converge_itr = 1;


      bool QR_iteration_done = 0;
      const int iterPerEigen = 8;
      // this implementation assumes fiding eigen each eigen value 
      // doesn't require no more than iterPerEigen
      const int QR_RQ_iterations = (rows-1)*iterPerEigen;
      constexpr int kIBitSize_QR_RQ_itr = fpga_tools::BitsForMaxValue<QR_RQ_iterations + 1>() + 1;
      
      // Iterative loop for QR and RQ/QQ coputation
      for(ac_int<kIBitSize_QR_RQ_itr, false> itr = 0; itr < QR_RQ_iterations; itr++){
        // PRINTF("Itr is: %d\n", (int)itr);

        // Compute the QR Decomposition

        // aconverged local copy of a_{i+1} that is used across multiple j iterations
        // for the computation of pip1 and p
        TT a_ip1[rows];
        // a local copy of a_ip1 that is used across multiple j iterations
        // for the computation of a_j
        TT a_i[rows];
        // Depending on the context, will contain:
        // -> -s[j]: for all the iterations to compute a_j
        // -> ir: for one iteration per j iterations to compute Q_i
        [[intel::fpga_memory]]
        [[intel::private_copies(2)]] // NO-FORMAT: Attribute
        TT s_or_ir[columns];

        T pip1, ir;

        // Initialization of the i and j variables for the triangular loop
        ac_int<kIBitSize, true> i = -1;
        ac_int<kJBitSize, true> j = 0;


        // [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        [[intel::ivdep(raw_latency)]]       // NO-FORMAT: Attribute
        for (int s = 0; s < kIterations; s++) {
          // Two matrix columns for partial results.
          TT col[rows];
          TT col_dummy[rows];
          TT col1[rows];

          // Current value of s_or_ir depending on the value of j
          // It is replicated kFanoutReduction times to reduce fanout
          TT s_or_ir_j[kBanksForFanout];

          // All the control signals are precomputed and replicated
          // kFanoutReduction times to reduce fanout
          bool j_eq_i[kBanksForFanout], i_gt_0[kBanksForFanout],
              i_ge_0_j_ge_i[kBanksForFanout], j_eq_i_plus_1[kBanksForFanout],
              i_lt_0[kBanksForFanout];

          fpga_tools::UnrolledLoop<kBanksForFanout>([&](auto k) {
            i_gt_0[k] = sycl::ext::intel::fpga_reg(i > 0);
            i_lt_0[k] = sycl::ext::intel::fpga_reg(i < 0);
            j_eq_i[k] = sycl::ext::intel::fpga_reg(j == i);
            i_ge_0_j_ge_i[k] = sycl::ext::intel::fpga_reg(i >= 0 && j >= i);
            j_eq_i_plus_1[k] = sycl::ext::intel::fpga_reg(j == i + 1);
            s_or_ir_j[k] = sycl::ext::intel::fpga_reg(s_or_ir[j]);
          });

          // Preload col and a_i with the correct data for the current iteration
          // These are going to be use to compute the dot product of two
          // different columns of the input matrix.

          TT diag_val = 0;
          fpga_tools::UnrolledLoop<rows>([&](auto k) {
            // find which fanout bank this unrolled iteration is going to use
            constexpr auto fanout_bank_idx = k / kFanoutReduction;

            // Load col with the current column of matrix A.
            // At least one iteration of the outer loop i is required
            // for the "working copy" a_compute to contain data.
            // If no i iteration elapsed, we must read the column of
            // matrix A directly from the a_load; col then contains a_j

            if (i_gt_0[fanout_bank_idx]) {
              col[k] = a_compute[j].template get<k>();
            }
            // Using an else statement makes the compiler throw an
            // inexplicable warning when using non complex types:
            // "Compiler Warning: Memory instruction with unresolved
            // pointer may lead to bad QoR."
            if (!i_gt_0[fanout_bank_idx]) {
              // supporting matrix deflation
              TT load_val = a_load[j].template get<k>();
              diag_val = (k == j) ? load_val : diag_val;
              col_dummy[k] =  load_val ; // write_val;
            }

          });

          diag_val = diag_val - R_shift;

          // Preload col and a_i with the correct data for the current iteration
          // These are going to be use to compute the dot product of two
          // different columns of the input matrix.
          fpga_tools::UnrolledLoop<rows>([&](auto k) {
            // find which fanout bank this unrolled iteration is going to use
            constexpr auto fanout_bank_idx = k / kFanoutReduction;
            

            if (!i_gt_0[fanout_bank_idx]) {
              // supporting matrix deflation
              TT load_val = col_dummy[k];
              TT update_val = (k == j) ? diag_val : load_val;
              TT write_val = (k >= kDM_size || j >= kDM_size) ? 0 : update_val;
              col[k] =  write_val;
            }

            // Load a_i for reuse across j iterations
            if (j_eq_i[fanout_bank_idx]) {
              a_i[k] = col[k];
            }
          });
        

          fpga_tools::UnrolledLoop<rows>([&](auto k) {
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
            auto prod_rhs = i_lt_0[fanout_bank_idx] ?
                                            TT{0.0} : s_or_ir_j[fanout_bank_idx];
            auto add = j_eq_i[fanout_bank_idx] ? TT{0.0} : col[k];
            if constexpr (is_complex) {
              col1[k] = prod_lhs * prod_rhs.conj() + add;
            } else {
              col1[k] = prod_lhs * prod_rhs + add;
            }

            // making invalid calculation to zero 
            col1[k] = (k >= kDM_size || j >= kDM_size) ? 0 : col1[k];

            // Store Q_i in q_result and the modified a_j in a_compute
            // To reduce the amount of control, q_result and a_compute
            // are both written to for each iteration of i>=0 && j>=i
            // In fact:
            // -> q_result could only be written to at iterations i==j
            // -> a_compute could only be written to at iterations
            //    j!=i && i>=0
            // The extra writes are harmless as the locations written to
            // are either going to be:
            // -> overwritten for the matrix Q (q_result)
            // -> unused for the a_compute
            if (i_ge_0_j_ge_i[fanout_bank_idx]) {
              q_result[j].template get<k>() = col1[k];
              a_compute[j].template get<k>() = col1[k];
            }

            // Store a_{i+1} for subsequent iterations of j
            if (j_eq_i_plus_1[fanout_bank_idx]) {
              a_ip1[k] = col1[k];
            }
          });

          // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
          TT p_ij{0.0};
          fpga_tools::UnrolledLoop<rows>([&](auto k) {
            if constexpr (is_complex) {
              p_ij = p_ij + col1[k] * a_ip1[k].conj();
            } else {
              p_ij = p_ij + col1[k] * a_ip1[k];
            }
          });

          // Compute pip1 and ir based on the results of the dot product
          if (j == i + 1) {
            if constexpr (is_complex) {
              pip1 = p_ij.r();
              ir = sycl::rsqrt(p_ij.r());
            } else {
              pip1 = p_ij;
              ir = sycl::rsqrt(p_ij);
            }
          }

          // Compute the value of -s[j]
          TT s_j;
          if constexpr (is_complex) {
            s_j = TT{0.0f - (p_ij.r()) / pip1, p_ij.i() / pip1};
          } else {
            s_j = -p_ij / pip1;
          }

          // j may be negative if the number of "dummy" iterations is
          // larger than the matrix size
          if (j >= 0) {
            if constexpr (is_complex) {
              s_or_ir[j] =
                  TT{j == i + 1 ? ir : s_j.r(), j == i + 1 ? 0.0f : s_j.i()};
            } else {
              s_or_ir[j] = j == i + 1 ? ir : s_j;
            }
          }

          // Compute the R_{i+1,i+1} or R_{i+1,j}
          TT r_ip1j;
          if constexpr (is_complex) {
            r_ip1j = j == i + 1 ? TT{sycl::sqrt(pip1), 0.0}
                                : TT{ir * p_ij.r(), ir * p_ij.i()};
          } else {
            r_ip1j = j == i + 1 ? sycl::sqrt(pip1) : ir * p_ij;
          }

          row_tuple rowR_write;
          fpga_tools::UnrolledLoop<columns>([&](auto t) {
            TT tra_val = (j >= i + 1 && (i+1) < kDM_size) ? r_ip1j : 0;
            rowR_write.template get<t>() = 
              ((i + 1 < columns) && t == j) ? tra_val : rowR_write.template get<t>();
          });

          if(i+1 < rows){
            r_matrix[i+1] = rowR_write;
          }

          // Update loop indexes
          if (j == (columns - 1)) {
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

        }  // end of s


        // Eigen vector QQ computation
        row_tuple QQ_write;
        row_tuple QQ_load;
        column_tuple Q_load_RQ, Q_load_RQ_tmp;

        // RQ computation and writig the results back in a_load 
        column_tuple colA_write;
        // const float threshold = 1e-4;
        bool converged = 1;
        // TT R_shift_tmp = 0;
        // TT R_shift_1BF = 0;

        TT a_wilk, b_wilk, c_wilk, d_wilk, e_wilk;


        [[intel::initiation_interval(1)]]  
        [[intel::loop_coalesce(2)]]
        for(ac_int<kIBitSize , false> i_ll = 0; i_ll < columns; i_ll++){
          for(ac_int<kIBitSize , false> j_ll = 0; j_ll < rows; j_ll++){
            ///////////////////////////////////////////////////////
            // QQ computation 
            ///////////////////////////////////////////////////////
            if(j_ll == 0){          
              QQ_load = QQ_matrix[i_ll];
            }

            column_tuple Q_load = q_result[j_ll];

            fpga_tools::UnrolledLoop<rows> ([&] (auto k) {
                TT Ival = (j_ll == k) ? 1 : 0; 
                Q_load.template get<k>() = (k >= kDM_size || j_ll >= kDM_size ? Ival : Q_load.template get<k>());
            });
            TT sum_QQ = 0;
            fpga_tools::UnrolledLoop<rows> ([&] (auto k){
              TT Ival = (k == i_ll) ? 1 : 0;
              TT QQ_final_val = (itr == 0) ? Ival : QQ_load.template get<k>();
              sum_QQ +=   QQ_final_val * Q_load.template get<k>();
            });

            fpga_tools::UnrolledLoop<rows> ([&] (auto k) {
              if(k == j_ll){
                QQ_write.template get<k> () = sum_QQ;
              }
            });

            if(j_ll ==columns - 1 && QR_iteration_done == 0){
              QQ_matrix[i_ll] = QQ_write;
            }

            ////////////////////////////////////////////////////////
            // RQ computation 
            ////////////////////////////////////////////////////////
            if(j_ll == i_ll+1){
              Q_load_RQ_tmp = Q_load;
            }

            if(j_ll == 0 && i_ll == 0){
              Q_load_RQ = Q_load;
            } else if(j_ll == 0){
              Q_load_RQ = Q_load_RQ_tmp;
            }
            row_tuple r_load = r_matrix[j_ll]; 

            TT sum_RQ = 0;
            fpga_tools::UnrolledLoop<rows> ([&] (auto k){
              sum_RQ += r_load.template get<k>() * Q_load_RQ.template get<k>();
            });

            converged = (j_ll == kDM_size-1 && i_ll < kDM_size-1 && fabs(sum_RQ) > KTHRESHOLD) ? 0 : converged;
            sum_RQ = (j_ll == i_ll) ? sum_RQ + R_shift : sum_RQ;

            // updatig the shift value 
            if(j_ll == i_ll && i_ll == kDM_size-3){
              d_wilk = sum_RQ;
            }

            if(j_ll == i_ll-1 && i_ll == kDM_size-2){
              e_wilk = sum_RQ;
            }

            if(j_ll == i_ll && i_ll == kDM_size-2){
              a_wilk = sum_RQ;
            }

            if(j_ll == i_ll -1 && i_ll == kDM_size-1){
              b_wilk = sum_RQ;
            }

            if(j_ll == i_ll && i_ll == kDM_size-1){
              c_wilk = sum_RQ;
            }

            fpga_tools::UnrolledLoop<columns> ([&] (auto k){
              bool cmp = (k==j_ll && j_ll < kDM_size && i_ll < kDM_size);
              if(cmp && QR_iteration_done == 0){
                a_load[i_ll].template get<k>() = sum_RQ;
              }
            });
          }
        }


        TT lamda = (a_wilk-c_wilk)/2.0;
        TT sign_lamda = (lamda > 0) - (lamda < 0);
        TT l_shift = c_wilk - (sign_lamda*b_wilk*b_wilk)/(fabs(lamda) + sqrt(lamda * lamda + b_wilk*b_wilk));

        R_shift = RELSHIFT ? c_wilk : l_shift;

        if(converged && kDM_size == KDEFLIM){
          QR_iteration_done = 1;
        }  
        
        if(converged){
          TT lamda = (d_wilk-a_wilk)/2.0;
          TT sign_lamda = (lamda > 0) - (lamda < 0);
          TT h_shift = RELSHIFT ? a_wilk : a_wilk - (sign_lamda*e_wilk*e_wilk)/(fabs(lamda) + sqrt(lamda * lamda + e_wilk*e_wilk));


          R_shift = h_shift;
          kDM_size = kDM_size -1;
        }

        R_shift -= R_shift*SHIFT_NOISE; //SHIFT_NOISE;
        


      } // End iterative loop


      // writing out eigen vector matrix QQ row by row 
      // Eigen vectors are the columns of this matrix 
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int row_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = row_iter == k;
          row_iter = sycl::ext::intel::fpga_reg(row_iter);
        });

        fpga_tools::NTuple<TT, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < rows) {
              pipe_write.template get<k>() = 
                  get[t] ? QQ_matrix[li / kLoopIterPerColumn]
                               .template get<t * pipe_size + k>()
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        QQOut::write(pipe_write);
      }
      

      // Wrriting out the computer RQ to streaming interface
      // RQ is stored back in a_load matrix
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
                  get[t] ? a_load[li / kLoopIterPerColumn]
                               .template get<t * pipe_size + k>()
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        // if(li == 0){pipe_write.template get<0> () = converge_itr;}
        RQOut::write(pipe_write);
      }

    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_QRD_HPP__ */