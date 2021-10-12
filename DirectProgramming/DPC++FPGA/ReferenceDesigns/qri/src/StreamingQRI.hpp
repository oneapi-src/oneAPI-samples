#pragma once 

#include "QRInversionDim.hpp"

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }


template <typename kernelName,          // Name to use for the Kernel
          bool isComplex,               // Helps identify the correct bank size
          typename T,                   // The datatype for the computation
          int RAWLatency,               // Minimum number of inner loop
                                        // iterations to achieve an outer
                                        // loop II of 1.  This value will
                                        // have to be tuned for optimal
                                        // performance.  Refer to the
                                        // Triangular Loop design pattern
                                        // tutorial.
          int rows,                     // Number of rows in the incoming A 
                                        // matrix
          int columns,                  // Number of columns in the incoming A
                                        // matrix, must be <= kNumRows
          typename QIn,       // Q input pipe, recieve a full column
                                        // with each read.
          typename RIn,       // R input pipe. Recieve one element 
                                        // per read.  
                                        // Only upper-right elements
                                        // of R are sent.  Sent in row order,
                                        // starting with row 0.
          typename IOut // Inverse matrix output pipe.
                                        // The output is written column by 
                                        // column
          >
sycl::event StreamingQRIKernel(sycl::queue& q) {

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
    sycl::stream out(21387, 21387, h);

    h.single_task<kernelName>([=] {
      int i =0;
      while (i++ == 0 ) {

        // Three copies of the full matrix, so that each matrix has a single
        // load and a single store.
        // A_load is the initial matrix received from the pipe
        // a_matrix is used and modified during calculations
        // q_matrix is a copy of a_matrix and is used to send the final output
        // The compiler has difficulty automatically figuring out an optimal
        // configuration for these memories, so force all relevant parameters.
        // NO-FORMAT comments are for clang-format
        // [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        // [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        // [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        // Column R_matrix[columns];
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        Column Q_matrix[columns];

        TT R[rows][columns];
        TT Q[rows][columns];
        TT inverse[rows][columns];

        /*
          ======================================================================
          Read the R matrix from the pipe
          ======================================================================
        */

        // int loaded = 0;
        // TT bank[kNumElementsPerBank];
        // for(int i = 0; i < rows; i++){
        //   for(int j = 0; j < columns; j++){
        //     if(loaded == 0){
        //       PipeType pipeData = RIn::read();
        //       UnrolledLoop<kNumElementsPerBank>([&](auto k) {
        //         bank[k] = pipeData.template get<k>();
        //       });
        //       loaded = kNumElementsPerBank;
        //     }
        //     if(j>=i){
        //       R[i][j] = bank[kNumElementsPerBank-loaded];
        //       loaded--;
        //     }
        //     else{
        //       R[i][j] = {0.0};
        //     }
        //   }
        // }

        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for(int i = 0; i < rows; i++){
          for(int j = 0; j < columns; j++){
            if(j>=i){
              R[i][j] = RIn::read();
            }
            else{
              R[i][j] = {0.0};
            }
          }
        }

        /*
          ======================================================================
          Read Q from the pipe
          ======================================================================
        */
        // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        // for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
        //                                                           si++) {
        //   // Only one bank is going to be stored per si iteration
        //   // To reduce fanout on si, a "get" table will contain for each 
        //   // bank a boolean to check if it should store this si iteration
        //   ac_int<BitsForMaxValue<kNumBanks>(), false> desired = 
        //                                                 si % (kNumBanks);
        //   bool get[kNumBanks];
        //   UnrolledLoop<kNumBanks>([&](auto k) {
        //     get[k] = desired == k;
        //     desired = sycl::ext::intel::fpga_reg(desired);
        //   });

        //   // Each bank will then check the get table to potentially 
        //   // read kNumElementsPerBank from Q_matrix and store the elements
        //   // in bank
        //   PipeType pipeData = QIn::read();

        //   ac_int<kSiNumBankBitSize, false> siNumBank = si / kNumBanks;
        //   UnrolledLoop<kNumBanks>([&](auto t) {
        //     UnrolledLoop<kNumElementsPerBank>([&](auto k) {
        //       constexpr auto rowIdx = t * kNumElementsPerBank + k;
        //       if constexpr(rowIdx < rows){
        //         // Q_matrix[siNumBank].template get<rowIdx>() = 
        //         Q[siNumBank][rowIdx] = 
        //         get[t] ? pipeData.template get<k>() : 
        //         sycl::ext::intel::fpga_reg(Q[siNumBank][rowIdx]);
        //         // sycl::ext::intel::fpga_reg(Q_matrix[siNumBank].template get<rowIdx>());
        //       }
        //     });
        //   });   
        // } // end for si=0:kStoreIter-1

        // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        // for (int col =0; col<columns; col++) {
        //   Column c = QIn::read();
        //   UnrolledLoop<rows>([&](auto k) {
        //     Q[col][k] = c.template get<k>(); 
        //   });
        // } // end for si=0:kStoreIter-1

        // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        // for (int col =0; col<columns; col++) {
        //   for (int row =0; row<columns; row++) {
        //     Q[col][row] = QIn::read(); 
        //   } 
        // } 

        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; 
                                                                  li++) {
          // Load a single bank of the input matrix 
          PipeType pipeData = QIn::read();

          // Write the current bank to the A_load matrix.
          ac_int<BitsForMaxValue<kNumBanks>(), false> writeRowGroup = 
                                                        li % (kNumBanks);
          ac_int<kLiNumBankBitSize, false> liNumBank = li / kNumBanks;

          UnrolledLoop<kNumBanks>([&](auto k) {
            UnrolledLoop<kNumElementsPerBank>([&](auto t) {
              constexpr auto rowIdx = k * kNumElementsPerBank + t;
              if constexpr (rowIdx < rows){
                if ((writeRowGroup == k)) {
                  Q_matrix[liNumBank].template get<rowIdx>() = 
                                        pipeData.template get<t>();
                }
              }

              // Delay data signals to create a vine-based data 
              // distribution to lower signal fanout.
              pipeData = sycl::ext::intel::fpga_reg(pipeData);
            });

            writeRowGroup = sycl::ext::intel::fpga_reg(writeRowGroup);
          });
        }



        // out << "R MATRIX" << sycl::endl;
        // for(int i = 0; i < rows; i++){
        //   for(int j = 0; j < columns; j++){
        //     out << R[i][j] << " ";
        //   }
        //   out << sycl::endl;
        // }

        TT col[rows][columns];
        for(int i = 0; i < columns; i++){
          UnrolledLoop<rows>([&](auto k) {
            col[k][i] = Q_matrix[i].template get<k>();
          });
        }

        out << "Q MATRIX" << sycl::endl;
        for(int i = 0; i < rows; i++){
          for(int j = 0; j < columns; j++){
            out << col[i][j] << " ";
          }
          out << sycl::endl;
        }

        /*
          ======================================================================
          Compute the inverse of R
          ======================================================================
        */
        TT inverseOfR[rows][columns];
        for(int i=0; i<columns; i++){
          UnrolledLoop<columns>([&](auto k) {
            inverseOfR[i][k] = {0};
          });
        }
/*
        // TT error = {0};
        for (int iGap = 0; iGap < columns; iGap++){
          for(int j = columns - 1; j-iGap >= 0; j--){
            int i = j-iGap;
            TT idMatrixValue = i == j ? TT{1} : TT{0};

            TT current_sum = {0};
            // TT div_val = {0};
            TT div_val = R[i][i];

            // #pragma unroll
            // for(int k = 0; k < columns; k++){
            UnrolledLoop<columns>([&](auto k) {
              auto inputIK = R[i][k];
              auto mul_rhs = k!=i ? inverseOfR[k][j] : TT{0};
              current_sum += inputIK * mul_rhs;
              // if(i==0 and j==62){
              //   if(k!=i){
              //     out << sycl::setprecision (50) << " input " << inputIK << " * output " << inverseOfR[k][j] << sycl::endl;
              //   }
              // }
              // if(k==i){
              //   div_val = inputJK
              // }
            });
            // }
            TT result = (idMatrixValue - current_sum)/div_val;

            inverseOfR[i][j] = result;


            // // Check result
            // TT new_current_sum = {0};
            // for(int k = columns-1; k >= 0 ; k--){
            //   new_current_sum += R[i][k]*inverseOfR[k][j];
            // }
            // if(abs(new_current_sum - idMatrixValue) > error){
            //   error = abs(new_current_sum - idMatrixValue);
            // }
            // // if(abs(new_current_sum - idMatrixValue) > 1e-4){
            //   // out << "error at " << j << " " << i << ": new_current_sum= " << new_current_sum << " idMatrixValue= " << idMatrixValue << " error= " << abs(new_current_sum - idMatrixValue) << sycl::endl;
            //   // out << "result (" << result << ") = (idMatrixValue (" << idMatrixValue << ") - new_current_sum (" << new_current_sum << "))/div_val (" << div_val << ")" << sycl::endl;
            // // }
            // if(i==0 and j==62){
            //   out << sycl::setprecision (50) << " current sum " << new_current_sum << sycl::endl;
            //   TT test = {0};
            //   PRINTF("Test value: %.25f\n", test);
            //   for(int k = columns-1; k >= 0; k--){
            //     PRINTF("%.25f * %.25f (%.25f)\n", R[i][k], inverseOfR[k][j], R[i][k] * inverseOfR[k][j]);
            //     test = test + R[i][k] * inverseOfR[k][j];
            //     PRINTF("Test value: %.25f\n", test);
            //   }
            //   PRINTF("error at %d %d: new_current_sum= %.25f idMatrixValue= %.25f  error= %.25f\n", i, j, new_current_sum, idMatrixValue, abs(new_current_sum - idMatrixValue));
            //   PRINTF("result (%.25f) = (idMatrixValue (%.25f) - new_current_sum (%.25f))/div_val ( %.25f)\n", result, idMatrixValue, current_sum, div_val);
            //   PRINTF("added afterward: %.25f\n", result*div_val + current_sum);

            // }
          }
        }
*/
        constexpr int RAWTriang = 64;
        constexpr int kNormalIterations = rows * (rows+1) / 2;
        constexpr int kExtraIterations =
        RAWTriang > rows ?
        (RAWTriang-2)*(RAWTriang-2+1)/2 - (RAWTriang-rows-1)*(RAWTriang-rows)/2 :
        (RAWTriang-2)*(RAWTriang-2+1)/2;
        constexpr int kTotalIterations = kNormalIterations + kExtraIterations;

        int iGap = 0;
        int j = columns - 1;
        int diagSize = rows;

        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        [[intel::ivdep(RAWTriang)]]  // NO-FORMAT: Attribute
        for(int it = 0; it < kTotalIterations; it++){
            int i = j-iGap;
            if(i>=0 && j >=0){
              TT idMatrixValue = i == j ? TT{1} : TT{0};

              TT current_sum = {0};
              TT div_val = R[i][i];

              UnrolledLoop<columns>([&](auto k) {
                auto inputIK = R[i][k];
                auto mul_rhs = k!=i ? inverseOfR[k][j] : TT{0};
                current_sum += inputIK * mul_rhs;
              });
              TT result = (idMatrixValue - current_sum)/div_val;

              inverseOfR[i][j] = result;
            }
            // PRINTF("i: %d, j: %d\n", i, j);

            int jm1 = j-1;
            int jmiGap = jm1-iGap;
            if(jmiGap < sycl::min(diagSize-RAWTriang, 0)){
              iGap = iGap + 1;
              j = columns - 1;
              diagSize = diagSize - 1;
            }
            else{
              j = jm1;
            }
        }
        
        // out << "R INVERSE MATRIX" << sycl::endl;
        // for(int i = 0; i < rows; i++){
        //   for(int j = 0; j < columns; j++){
        //     out << inverseOfR[i][j] << " ";
        //   }
        //   out << sycl::endl;
        // }

        /*
          ======================================================================
          Multiply the inverse of R by the transposition of Q
          ======================================================================
        */
        for(int i = 0; i < rows; i++){
          for(int j = 0; j < columns; j++){
            TT dotProduct = {0.0};
            // for(int k = 0; k < rows; k++){
            UnrolledLoop<rows>([&](auto k) {
              // dotProduct += inverseOfR[i][k] * Q[k][j]; 
              dotProduct += inverseOfR[i][k] * Q_matrix[j].template get<k>(); 
              PRINTF("Q_matrix[%d].template get<%u>() %f\n", j, k, Q_matrix[j].template get<k>());  
            });
            // }
            inverse[i][j] = dotProduct;
          }
        }

        /*
          ======================================================================
          Write result to the output pipe
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
          // read kNumElementsPerBank from Q_matrix and store the elements
          // in bank
          PipeType pipeData;

          ac_int<kSiNumBankBitSize, false> siNumBank = si / kNumBanks;
          UnrolledLoop<kNumBanks>([&](auto t) {
            UnrolledLoop<kNumElementsPerBank>([&](auto k) {
              constexpr auto rowIdx = t * kNumElementsPerBank + k;
              if constexpr(rowIdx < rows){
                pipeData.template get<k>() = get[t] ? 
                inverse[siNumBank][rowIdx] :
                                    // Q_matrix[siNumBank].template get<rowIdx>() : 
                              sycl::ext::intel::fpga_reg(pipeData.template get<k>());
              }
            });
          });

          IOut::write(pipeData);
             
        } // end for si=0:kStoreIter-1

      } //while (1)
    }); // end of h.single_task
  }); // end of q.submit

  return e;
}