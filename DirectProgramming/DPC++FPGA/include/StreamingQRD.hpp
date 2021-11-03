#pragma once 
#include "Tuple.hpp"
#include "Utils.hpp"

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
template <typename kernelName,  // Name to use for the Kernel
          typename T,           // The datatype for the computation
          bool isComplex,       // True if T is ac_complex<X>
          int rows,             // Number of rows in the incoming A matrices
          int columns,          // Number of columns in the incoming A
                                // matrices, must be <= rows
          int RAWLatency,       // Read after write latency (in iterations) of 
                                // the triangular loop of this kernel.
                                // This value depends on the FPGA target, the 
                                // datatype, the target frequency, etc.
                                // This value will have to be tuned for optimal
                                // performance. Refer to the Triangular Loop 
                                // design pattern tutorial.
                                // In general, find a high value for which the
                                // compiler is able to achieve an II of 1 and 
                                // go down from there.
          int matrixCount,      // Number of matrices to read from the input 
                                // pipe sequentially
          typename AIn,         // A matrix input pipe, receive a full column
                                // of TT elements with each read
          typename QOut,        // Q matrix output pipe, send a full column
                                // of TT elements with each write
          typename ROut         // R matrix output pipe, send one TT element
                                // per write. Only upper-right elements of R are
                                // sent in row order, starting with row 0.
          >
sycl::event StreamingQRDKernel(sycl::queue& q // Device queue
                              ) {

  // Functional limitations
  static_assert(rows>=columns, 
                "only rectangular matrices with rows>=columns are supported");
  static_assert((columns <= 512) && (columns >= 4), 
                        "only matrices of size 4x4 to 512x512 are supported");

  /*
    This code implements a OneAPI optimized variation of the following algorithm

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
  // of isComplex
  typedef typename std::conditional<isComplex, ac_complex<T>, T>::type TT;

  // Type used to store the matrices in the compute loop
  using ColumnTuple = NTuple<TT, rows>;

  // Number of upper-right elements in the R output matrix 
  constexpr int kRMatrixSize = columns * (columns + 1) / 2;
  // Fanout reduction factor for signals that fanout to rows compute cores 
  constexpr int kFanoutReduction = 8;
  // Number of signal replication required to cover all the rows compute cores
  // given a kFanoutReduction factor
  constexpr int kBanksForFanout = (rows % kFanoutReduction) ? 
                        (rows / kFanoutReduction) + 1 : rows / kFanoutReduction;

  // Number of iterations performed without any dummy work added for the 
  // triangular loop optimization
  constexpr int kVariableIterations = columns - RAWLatency;
  // Total number of dummy iterations
  static constexpr int kDummyIterations = RAWLatency > columns ?
              (columns - 1) * columns / 2 + (RAWLatency - columns) * columns :
              RAWLatency * (RAWLatency - 1) / 2;
  // Total number of iterations (including dummy iterations)
  static constexpr int kIterations = columns + columns * (columns+1) / 2 +  
                                                              kDummyIterations;
  
  // Size in bits of the "i" loop variable in the triangular loop
  // i starts from -1 as we are doing a full copy of the matrix read from the 
  // pipe to a "compute" matrix before starting the decomposition 
  constexpr int kIBitSize = BitsForMaxValue<rows + 1>() + 1;

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
  static constexpr int kJNegativeIterations = kVariableIterations < 0 ? 
                                                      -kVariableIterations : 1;
  static constexpr int kJBitSize =  BitsForMaxValue<columns + 1>() + 
                                    BitsForMaxValue<kJNegativeIterations>();

  auto e = q.submit([&](sycl::handler& h) {
    h.single_task<kernelName>([=] {

      // Iterate over the number of matrices to decompose per kernel call
      for (int matrixIter = 0; matrixIter < matrixCount; matrixIter++) {

        // Three copies of the full matrix, so that each matrix has a single
        // load and a single store.
        // ALoad is the initial matrix received from the pipe
        // ACompute is used and modified during calculations
        // QResult is a copy of ACompute and is used to send the final output
        ColumnTuple ALoad[columns];
        ColumnTuple ACompute[columns];
        ColumnTuple QResult[columns];
        
        // Contains the values of the upper-right part of R in a row by row 
        // fashion, starting by row 0
        TT R_result[kRMatrixSize];

        /*
          ======================================================================
          Copy a matrix from the pipe to a local memory
          ======================================================================
        */
        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (int col=0; col<columns; col++) {
          // Read a column of the input matrix from the pipe 
          pipeTable<rows, TT> pipeData = AIn::read();

          // Write the current column to the ALoad matrix.
          UnrolledLoop<rows>([&](auto k) {
            ALoad[col].template get<k>() = pipeData.elem[k];
          });
        }

        /*
          ======================================================================
          Compute the QR Decomposition
          ======================================================================
        */
        // R_result write index
        int RElementIndex = 0;

        // a local copy of a_{i+1} that is used across multiple j iterations 
        // for the computation of pip1 and p
        TT a_ip1[rows];
        // a local copy of a_ip1 that is used across multiple j iterations 
        // for the computation of a_j
        TT a_i[rows];
        // Depending on the context, will contain:
        // -> -s[j]: for all the iterations to compute a_j
        // -> ir: for one iteration per j iterations to compute Q_i
        [[intel::fpga_memory]]
        TT sOrIr[columns];
       
        T pip1, ir;

        // Initialization of the i and j variables for the triangular loop
        ac_int<kIBitSize, true> i = -1;
        ac_int<kJBitSize, true> j = 0;
        
        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        [[intel::ivdep(RAWLatency)]]        // NO-FORMAT: Attribute
        for (int s = 0; s < kIterations; s++) {
          // Pre-compute the next values of i and j
          ac_int<kIBitSize, true> nextI;
          ac_int<kJBitSize, true> nextJ;
          if (j == columns - 1) {
            // If i reached an index at which the j inner loop don't have
            // enough time to write its result for the next i iteration,
            // some "dummy" iterations are introduced 
            nextJ = (kVariableIterations > i) ? 
                            ac_int<kJBitSize, true>{i + 1} : 
                            ac_int<kJBitSize, true>{kVariableIterations};
            nextI = i + 1;
          } else {
            nextJ = j + 1;
            nextI = i;
          }

          // Two matrix columns for partial results.
          TT col[rows];
          TT col1[rows];

          // Current value of sOrIr depending on the value of j
          // It is replicated kFanoutReduction times to reduce fanout
          TT sOrIrJ[kBanksForFanout];

          // All the control signals are precomputed and replicated
          // kFanoutReduction times to reduce fanout
          bool  jEqI[kBanksForFanout], 
                iGt0[kBanksForFanout],
                iGe0JGeI[kBanksForFanout], 
                jEqI_plus_1[kBanksForFanout],
                iLt0[kBanksForFanout];

          UnrolledLoop<kBanksForFanout>([&](auto k) {
            iGt0[k] = sycl::ext::intel::fpga_reg(i > 0);
            iLt0[k] = sycl::ext::intel::fpga_reg(i < 0);
            jEqI[k] = sycl::ext::intel::fpga_reg(j == i);
            iGe0JGeI[k] = sycl::ext::intel::fpga_reg(i >= 0 & j >= i);
            jEqI_plus_1[k] = sycl::ext::intel::fpga_reg(j == i + 1);
            sOrIrJ[k] = sycl::ext::intel::fpga_reg(sOrIr[j]);
          });

          // Preload col and a_i with the correct data for the current iteration
          // These are going to be use to compute the dot product of two 
          // different columns of the input matrix.
          UnrolledLoop<rows>([&](auto k) {
            // find which fanout bank this unrolled iteration is going to use
            constexpr auto fanoutBankIdx = k / kFanoutReduction;

            // Load col with the current column of matrix A.
            // At least one iteration of the outer loop i is required
            // for the "working copy" ACompute to contain data.
            // If no i iteration elapsed, we must read the column of 
            // matrix A directly from the ALoad; col then contains a_j

            if(iGt0[fanoutBankIdx]){
              col[k] = ACompute[j].template get<k>();
            }
            // Using an else statement makes the compiler throw an
            // inexplicable warning when using non complex types:
            // "Compiler Warning: Memory instruction with unresolved 
            // pointer may lead to bad QoR."
            if(!iGt0[fanoutBankIdx]){
              col[k] = ALoad[j].template get<k>();
            }

            // Load a_i for reuse across j iterations
            if (jEqI[fanoutBankIdx]) {
              a_i[k] = col[k];
            }
          });

          UnrolledLoop<rows>([&](auto k) {
            // find which fanout bank this unrolled iteration is going to use
            constexpr auto fanoutBankIdx = k / kFanoutReduction;

            // Depending on the iteration this code will compute either:
            // -> If i=j, a column of Q: Q_i = a_i*ir
            //    In that case, no term is added to the mult_add construct
            // -> If i!=j, an updated column of a: a_j - s[j]*a_i
            //    There is a special case if i<0 where a_j is unmodified 
            //    but the i iteration is still required to fill ir and s 
            //    for subsequent iterations
            auto prodLHS = a_i[k];
            auto prodRHS = iLt0[fanoutBankIdx] ? TT{0.0} : 
                                                  sOrIrJ[fanoutBankIdx];
            auto add = jEqI[fanoutBankIdx] ? TT{0.0} : col[k];
            if constexpr(isComplex){
              col1[k] = prodLHS * prodRHS.conj() + add;
            }
            else{
              col1[k] = prodLHS * prodRHS + add;
            }

            // Store Q_i in QResult and the modified a_j in ACompute
            // To reduce the amount of control, QResult and ACompute
            // are both written to for each iteration of i>=0 && j>=i
            // In fact:
            // -> QResult could only be written to at iterations i==j
            // -> ACompute could only be written to at iterations 
            //    j!=i && i>=0  
            // The extra writes are harmless as the locations written to 
            // are either going to be:
            // -> overwritten for the matrix Q (QResult)
            // -> unused for the ACompute
            if (iGe0JGeI[fanoutBankIdx]) {
              QResult[j].template get<k>() = col1[k];
              ACompute[j].template get<k>() = col1[k];
            }

            // Store a_{i+1} for subsequent iterations of j
            if (jEqI_plus_1[fanoutBankIdx]) {
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

          // Compute pip1 and ir based on the results of the dot product
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

          // Compute the value of -s[j]
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
            if constexpr(isComplex){
              sOrIr[j] = TT{j == i + 1 ? ir : s_j.r(),
                            j == i + 1 ? 0.0f : s_j.i()};
            }
            else{
              sOrIr[j] = j == i + 1 ? ir : s_j; 
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
          if ((j >= i + 1) & (i + 1 < columns)) {
            R_result[RElementIndex] = r_ip1j;
            RElementIndex++;
          }

          // Update loop indexes
          j = nextJ;
          i = nextI;

        } // end of s

        /*
          ======================================================================
          Copy the R matrix result to the ROut output pipe
          ======================================================================
        */
        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for (int r_idx = 0; r_idx < kRMatrixSize; r_idx++) {
          ROut::write(R_result[r_idx]);
        }

        /*
          ======================================================================
          Copy the Q matrix result to the QOut output pipe
          ======================================================================
        */
        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for (int col = 0; col < columns; col++) {
          // Load a full column of Q to the correct pipe type
          pipeTable<rows, TT> pipeData;
          UnrolledLoop<rows>([&](auto k) {
            pipeData.elem[k] = QResult[col].template get<k>();
          });

          // Write the Q column to the pipe
          QOut::write(pipeData);
        } // end of col
      } // end of matrixIter
    }); // end of h
  }); // end of q submit

  return e;
}
