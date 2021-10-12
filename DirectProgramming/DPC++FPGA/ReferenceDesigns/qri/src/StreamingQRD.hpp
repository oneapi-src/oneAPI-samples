#pragma once 

#include "QRInversionDim.hpp"

template <typename kernelName,      // Name to use for the Kernel
          bool isComplex,           // Helps identify the correct bank size
          typename T,               // The datatype for the computation
          int RAWLatency,           // Minimum number of inner loop
                                    // iterations to achieve an outer
                                    // loop II of 1.  This value will
                                    // have to be tuned for optimal
                                    // performance.  Refer to the
                                    // Triangular Loop design pattern
                                    // tutorial.
          int rows,                 // Number of rows in the incoming A matrix
          int columns,              // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          typename AIn,   // A matrix input, receive a full column
                                    // of complex numbers with each read,
                                    // wrapped in NTuple
          typename QOut,  // Q output pipe, send a full column
                                    // of complex numbers with each write.
                                    // Column 0 is sent first, columns-1
                                    // is sent last
          typename ROut   // R output pipe.  Send one complex number
                                    // per write.  Only upper-right elements
                                    // of R are sent.  Sent in row order,
                                    // starting with row 0.
          >
sycl::event StreamingQRDKernel(sycl::queue& q) {

  typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

  using dim = QRInversionDim<isComplex, rows, columns, RAWLatency>;
  using Column = NTuple<TT, rows>;

  constexpr int kRMatrixSize = dim::RMatrixSize;
  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kBankwidth = dim::BankWidth;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kNumBanksNextPow2 = dim::NumBanksNextPow2;
  constexpr bool kNonCompleteIter = dim::NonCompleteIter;
  constexpr int kExtraIter = dim::ExtraIter;
  constexpr int kLoadIter = dim::LoadIter;
  constexpr int kStoreIter = dim::StoreIter;
  constexpr int kLoadIterBitSize = dim::LoadIterBitSize;
  constexpr int kStoreIterBitSize = dim::StoreIterBitSize;
  constexpr int kLiNumBankBitSize = dim::LiNumBankBitSize;
  constexpr int kSiNumBankBitSize = dim::SiNumBankBitSize;
  constexpr int kNValue = dim::NValue;
  constexpr int kVariableIterations = dim::VariableIterations;
  constexpr int kIterations = dim::Iterations;
  constexpr int kIBitSize = dim::IBitSize;
  constexpr int kJBitSize = dim::JBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn;
  constexpr int kNumRBanks = dim::NumRBanks;

  using PipeType = NTuple<TT, kNumElementsPerBank>;

  auto e = q.submit([&](sycl::handler& h) {
    h.single_task<kernelName>([=] {

      while (1) {

        // Three copies of the full matrix, so that each matrix has a single
        // load and a single store.
        // A_load is the initial matrix received from the pipe
        // a_matrix is used and modified during calculations
        // q_matrix is a copy of a_matrix and is used to send the final output
        // The compiler has difficulty automatically figuring out an optimal
        // configuration for these memories, so force all relevant parameters.
        // NO-FORMAT comments are for clang-format
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        Column A_load[columns];
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        Column A_compute[columns];
        Column Q_Result[columns];
        
        TT R_result[kRMatrixSize];

        /*
          ================================================================
          Loop 1: Copy data from the pipe to a local memory.
          ================================================================
        */

        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; 
                                                                  li++) {
          // Load a single bank of the input matrix 
          PipeType pipeData = AIn::read();

          // Write the current bank to the A_load matrix.
          ac_int<BitsForMaxValue<kNumBanks>(), false> write_row_group = 
                                                        li % (kNumBanks);
          ac_int<kLiNumBankBitSize, false> liNumBank = li / kNumBanks;

          UnrolledLoop<kNumBanks>([&](auto k) {
            UnrolledLoop<kNumElementsPerBank>([&](auto t) {
              constexpr auto rowIdx = k * kNumElementsPerBank + t;
              if constexpr (rowIdx < rows){
                if ((write_row_group == k)) {
                  A_load[liNumBank].template get<rowIdx>() = 
                                        pipeData.template get<t>();
                }
              }

              // Delay data signals to create a vine-based data 
              // distribution to lower signal fanout.
              pipeData = sycl::ext::intel::fpga_reg(pipeData);
            });

            write_row_group = sycl::ext::intel::fpga_reg(write_row_group);
          });
        }


        /*
          ================================================================
          Loop 2: Main computation the QR Decomposition.
          ================================================================
        
          Main computation of the QR Decomposition.

          This code implements a OneAPI optimized variation of the 
          following algorithm:

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

        // Get index in the output at which to start writing the outputs 
        // for input matrix l.
        int RElementIndex = 0;

        // a local copy of a_{i+1} that is used across multiple j 
        // iterations for the computation of pip1 and p
        TT a_ip1[rows];
        // a local copy of a_ip1 that is used across multiple j iterations 
        // for the computation of a_j
        TT a_i[rows];
        // Depending on the context, will contain:
        // -> -s[j]: for all the iterations to compute a_j
        // -> ir: for one iteration per j iterations to compute Q_i
        constexpr int kSuper_dummy_iterations = RAWLatency - columns;
        constexpr int kIncreasedBufferSize = kSuper_dummy_iterations < 0 ? 
                                              0 : kSuper_dummy_iterations; 
        TT s_or_i[columns + kIncreasedBufferSize];

        // Adding kIncreasedBufferSize is a waste of resource because we 
        // are going to read and write only to "columns" different places
        // If we don't add it, the compiler does not achieve II 1 because 
        // it is not able to determine that we are not going to read at 
        // the same location the last iteration just wrote.
        // This is probably due to the fact that when the access index is 
        // negative, we are not actually reading/writing to it (gated by 
        // an if statement) but it seems to make the compiler confused.
       
        T pip1, ir;

        // The triangular loop over i and j has been optimized using the 
        // method described in the triangular loop optimization tutorial.
        // Therefore, the code iterates over the total number of 
        // iterations, including "dummy" ones (that ensures II=1).
        ac_int<kIBitSize, true> i = -1;
        ac_int<kJBitSize, true> j = 0;
        
        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        [[intel::ivdep(RAWLatency)]]  // NO-FORMAT: Attribute
        for (int s = 0; s < kIterations; s++) {

          ac_int<kIBitSize, true> next_i;
          ac_int<kJBitSize, true> next_j;
          // Update the loop indexes.
          if (j == kNValue - 1) {
            // If i reached an index at which the j inner loop don't have
            // enough time to write its result for the next i iteration,
            // some "dummy" iterations are introduced 
            next_j = (kVariableIterations > i) ? 
                            ac_int<kJBitSize, true>{i + 1} : 
                            ac_int<kJBitSize, true>{kVariableIterations};
            next_i = i + 1;
          } else {
            next_j = j + 1;
            next_i = i;
          }

          // Temporary storage for a column of the input matrix and for
          // partial results.
          [[intel::fpga_register]] TT col[rows];
          [[intel::fpga_register]] TT col1[rows];

          // Current value of s_or_i depending on the value of j
          // It is replicated kNumBanks times to reduce fanout
          TT sori[kNumBanks];

          // All the control signals are precomputed and replicated
          // kNumBanks times to reduce fanout
          bool  j_eq_i[kNumBanks], 
                i_gt_0[kNumBanks],
                i_ge_0_j_ge_i[kNumBanks], 
                j_eq_i_plus_1[kNumBanks],
                i_lt_0[kNumBanks];

          UnrolledLoop<kNumBanks>([&](auto k) {
            i_gt_0[k] = sycl::ext::intel::fpga_reg(i > 0);
            i_lt_0[k] = sycl::ext::intel::fpga_reg(i < 0);
            j_eq_i[k] = sycl::ext::intel::fpga_reg(j == i);
            i_ge_0_j_ge_i[k] = sycl::ext::intel::fpga_reg(i >= 0 && j >= i);
            j_eq_i_plus_1[k] = sycl::ext::intel::fpga_reg(j == i + 1);
            int idx = j + kIncreasedBufferSize;
            sori[k] = sycl::ext::intel::fpga_reg(s_or_i[idx]);
          });

          // Preload col and a_i with the correct data for the current 
          // iteration.
          // These are going to be use to compute the dot product of 
          // two different column of the input matrix.
          UnrolledLoop<rows>([&](auto k) {
            // find which bank this unrolled iteration is going to use
            constexpr auto bank = k / kNumElementsPerBank;

            // Load col with the current column of matrix a.
            // At least one iteration of the outer loop i is required
            // for the "working copy" A_compute to contain data.
            // If no i iteration elapsed, we must read the column of 
            // matrix a directly from the A_load col then contains a_j

            if(i_gt_0[bank]){
              // col[k] = A_compute[int(j) + k*columns];
              // col[k] = A_compute[k].d[j];
              col[k] = A_compute[j].template get<k>();
            }
            // Using an else statement makes the compiler throw an
            // inexplicable warning when using non complex types:
            // "Compiler Warning: Memory instruction with unresolved 
            // pointer may lead to bad QoR."
            if(!i_gt_0[bank]){
              // col[k] = A_load[k].d[j];
              col[k] = A_load[j].template get<k>();
            }

            // Load a_i for reuse across j iterations
            if (j_eq_i[bank]) {
              a_i[k] = col[k];
            }

          });

          UnrolledLoop<rows>([&](auto k) {
            // find which bank this unrolled iteration is going to use
            constexpr auto bankIdx = k / kNumElementsPerBank;

            // Depending on the iteration this code will compute either:
            // -> If i=j, a column of Q: Q_i = a_i*ir
            //    In that case, no term is added to the mult_add construct
            // -> If i!=j, an updated column of a: a_j - s[j]*a_i
            //    There is a special case if i<0 where a_j is unmodified 
            //    but the i iteration is still required to fill ir and s 
            //    for subsequent iterations
            auto prod_lhs = a_i[k];
            auto prod_rhs = i_lt_0[bankIdx] ? TT{0.0} : sori[bankIdx];
            auto add = j_eq_i[bankIdx] ? TT{0.0} : col[k];
            if constexpr(isComplex){
              col1[k] = prod_lhs * prod_rhs.conj() + add;
            }
            else{
              col1[k] = prod_lhs * prod_rhs + add;
            }

            // Store Q_i in Q_Result and the modified a_j in A_compute
            // To reduce the amount of control, Q_Result and A_compute
            // are both written to for each iteration of i>=0 && j>=i
            // In fact:
            // -> Q_Result could only be written to at iterations i==j
            // -> A_compute could only be written to at iterations 
            //    j!=i && i>=0  
            // The extra writes are harmless as the locations written to 
            // are either going to be:
            // -> overwritten for the matrix Q (Q_Result)
            // -> unused for the A_compute
            if (i_ge_0_j_ge_i[bankIdx]) {
              // Q_Result[k].d[j] = A_compute[k].d[j] = col1[k];
              Q_Result[j].template get<k>() = col1[k];
              A_compute[j].template get<k>() = col1[k];
            }

            // Store a_{i+1} for subsequent iterations of j
            if (j_eq_i_plus_1[bankIdx]) {
              a_ip1[k] = col1[k];
            }
          });

          // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
          TT p_ij{0.0};

          UnrolledLoop<rows>([&](auto k) {
            if constexpr(isComplex){
              p_ij = p_ij + col1[k] * a_ip1[k].conj();
            }
            else{
              p_ij = p_ij + col1[k] * a_ip1[k];
            }
          });

          if (j == i + 1) {
            if constexpr(isComplex){
              pip1 = p_ij.r();
              ir = sycl::rsqrt(p_ij.r());
            }
            else{
              pip1 = p_ij;
              ir = sycl::rsqrt(p_ij); 
            }
          }

          TT s_j;
          if constexpr(isComplex){
            s_j = TT{0.0f - (p_ij.r()) / pip1, p_ij.i() / pip1};
          }
          else{
            s_j = - p_ij / pip1;
          }

          // j may be negative if the number of "dummy" iterations is 
          // larger than the matrix size
          if (j >= 0) {
            int idx = j + kIncreasedBufferSize;
            if constexpr(isComplex){
              s_or_i[idx] = TT{j == i + 1 ? ir : s_j.r(),
                                  j == i + 1 ? 0.0f : s_j.i()};
            }
            else{
              s_or_i[idx] = j == i + 1 ? ir : s_j; 
            }
          }

          // Compute the R_{i+1,i+1} or R_{i+1,j} 
          TT r_ip1j;
          if constexpr(isComplex){
            r_ip1j = j == i + 1 ? TT{sycl::sqrt(pip1), 0.0} : 
                                        TT{ir * p_ij.r(), ir * p_ij.i()};
          }
          else{
            r_ip1j = j == i + 1 ? sycl::sqrt(pip1) : ir * p_ij;
          }

          // Write the computed R value when j is not a "dummy" iteration
          // introduced to optimized the triangular loop
          if (j >= i + 1 && i + 1 < kNValue) {
            R_result[RElementIndex] = r_ip1j;
            RElementIndex++;
          }

          j = next_j;
          i = next_i;

        } // end for s=0:kIterations-1



        /*
          ======================================================================
          Loop 3: Copy the R matrix result from local memory to the output pipe
          ======================================================================
        */

        // int r_idx = 0;
        // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        // for (int si = 0; si < kNumRBanks; si++) {
        //   PipeType pipeData;
        //   UnrolledLoop<kNumElementsPerBank>([&](auto kk) {
        //     if ((r_idx + kk) < kRMatrixSize){
        //       pipeData.template get<kk>() = R_result[r_idx + kk];
        //     }
        //   });
        //   ROut::write(pipeData);
        //   r_idx += kNumElementsPerBank;
        // }

        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for (int r_idx = 0; r_idx < kRMatrixSize; r_idx++) {
          ROut::write(R_result[r_idx]);
        }

        /*
          ======================================================================
          Loop 4: Copy the Q matrix result from local memory to the output pipe
          ======================================================================
        */
        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                  si++) {
          // Only one bank is going to be stored per si iteration
          // To reduce fanout on si, a "get" table will contain for each 
          // bank a boolean to check if it should store this si iteration
          ac_int<BitsForMaxValue<kNumBanks>(), false> desired = 
                                                        si % (kNumBanks);
          bool get[kNumBanks];
          UnrolledLoop<kNumBanks>([&](auto k) {
            get[k] = desired == k;
            desired = sycl::ext::intel::fpga_reg(desired);
          });

          // Each bank will then check the get table to potentially 
          // read kNumElementsPerBank from Q_Result and store the elements
          // in bank
          PipeType pipeData;

          ac_int<kSiNumBankBitSize, false> siNumBank = si / kNumBanks;
          UnrolledLoop<kNumBanks>([&](auto t) {
            UnrolledLoop<kNumElementsPerBank>([&](auto k) {
              constexpr auto rowIdx = t * kNumElementsPerBank + k;
              if constexpr(rowIdx < rows){
                pipeData.template get<k>() = get[t] ? 
                                    Q_Result[siNumBank].template get<rowIdx>() : 
                         sycl::ext::intel::fpga_reg(pipeData.template get<k>());
              }
            });
          });

          QOut::write(pipeData);
             
        } // end for si=0:kStoreIter-1

        // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        // for (int col = 0; col < columns; col++) {
        //   QOut::write(Q_Result[col]);
        // } // end for col

        // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        // for (int col = 0; col < columns; col++) {
        //   TT tmp[rows];
        //   UnrolledLoop<rows>([&](auto k) {
        //     tmp[k] = Q_Result[col].template get<k>();
        //   });
        //   for (int row = 0; row < rows; row++) {
        //     QOut::write(tmp[row]);
        //   } // end for col
        // } // end for col


      } //while (1)
    }); // end of h.single_task
  }); // end of q.submit

  return e;
}