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
  using Row = NTuple<TT, columns>;

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

/*      // [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      // [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      Column Q_matrix[columns];
*/

      TT Q_matrix[rows][columns];
      TT QT_matrix[rows][columns];

      [[intel::numbanks(1)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT R[rows][columns];

      TT RT_matrix[rows][columns];

      // [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT inverse[rows][columns];

      /*
        ========================================================================
        Read the R matrix from the pipe
        ========================================================================
      */

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for(int i = 0; i < rows; i++){
        TT Rcol[columns];
        for(int j = 0; j < columns; j++){
          if(j>=i){
            Rcol[j] = RIn::read();
          }
          else{
            Rcol[j] = {0.0};
          }
        }

        UnrolledLoop<columns>([&](auto k) {
          R[i][k] = Rcol[k];
        });
      }

      // PRINTF("R\n");
      // for(int i = 0; i < rows; i++){
      //   for(int j = 0; j < columns; j++){
      //     PRINTF("(%f, %f) ", R[i][j].r(), R[i][j].i());
      //   }
      //   PRINTF("\n");
      // }

      for(int row=0; row<rows; row++){
        for(int col=0; col<columns; col++){
          RT_matrix[row][col] = R[col][row];
        }
      }

      /*
        ========================================================================
        Read Q from the pipe
        ========================================================================
      */

      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (int col=0; col<columns; col++) {
        // Load a single bank of the input matrix 
        column<rows, TT> pipeData = QIn::read();

        // Write the current column to the A_load matrix.
        UnrolledLoop<columns>([&](auto k) {
          Q_matrix[col][k] = pipeData.row[k];
        });
      }


      /*
        ========================================================================
        Compute the transpose of Q
        ========================================================================
      */

      // {
      //   constexpr int iterations = (rows-1) * rows / 2; 
      //   int row = 0;
      //   int col = 1;
      //   int colp1 = 2;
      //   int rowp1 = 1;
      //   int rowp2 = 2;

      //   [[intel::ivdep(iterations)]]  // NO-FORMAT: Attribute
      //   for(int it=0; it<iterations; it++){
      //     PRINTF("row %d, col %d\n", row, col);
      //     TT tmp = Q_matrix[row][col];
      //     Q_matrix[row][col] = Q_matrix[col][row];
      //     Q_matrix[col][row] = tmp;

      //     if(col==columns-1){
      //       row = rowp1;
      //       col = rowp2;
      //     }
      //     else{
      //       colp1 = col + 1;
      //       col = colp1;
      //       rowp1 = row + 1;
      //       rowp2 = row + 2;          
      //     }
      //   }
      // }

      for(int row=0; row<rows; row++){
        for(int col=0; col<columns; col++){
          QT_matrix[row][col] = Q_matrix[col][row];
        }
      }

      /*
        ========================================================================
        Compute the inverse of R
        ========================================================================
      */
/*
          [[intel::bankwidth(kBankwidth), intel::numbanks(kNumBanks)]] 
          struct {
            TT c[rows];
          } inverseOfR[columns];


*/
      // [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      // [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      // [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute


      TT inverseOfR[rows][columns];

      // TT R_inverse[rows][columns] = {0};
      TT R_inverse[rows][columns];

      // Row R_inverse[columns];
      for(int i=0; i<rows; i++){
        UnrolledLoop<columns>([&](auto k) {
          // R_inverse[i].template get<k>() = {0};
          R_inverse[i][k] = {0};
          // inverseOfR[i][k] = {0};
          // inverseOfR[k][i] = {0};
        });
      }

      constexpr int RAWTriang = 300;
      constexpr int kNormalIterations = rows * (rows+1) / 2;
      constexpr int kExtraIterations =
      RAWTriang > rows ?
      (RAWTriang-2)*(RAWTriang-2+1)/2 - (RAWTriang-rows-1)*(RAWTriang-rows)/2 :
      (RAWTriang-2)*(RAWTriang-2+1)/2;
      constexpr int kTotalIterations = kNormalIterations + kExtraIterations;

      int row = 0;
      int col = 0;
      int cp1 = 1;
      int iter = 0;
      int ip1 = 1;
      int ip2 = 2;
      int diagSize = columns;
      int diagSizem1 = columns - 1;
      int cp1Limit = RAWTriang-columns-columns > 0 ? 
                                                    RAWTriang-columns : columns;
      int nextcp1Limit = RAWTriang-columns-1-columns > 0 ? 
                                                  RAWTriang-columns-1 : columns;

      // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      [[intel::ivdep(RAWTriang)]]  // NO-FORMAT: Attribute
      for(int it = 0; it < kTotalIterations; it++){
        if(row<rows && col<columns){
          TT idMatrixValue = row == col ? TT{1} : TT{0};

          TT current_sum = {0};
          // TT div_val = R[col][col];
          TT div_val;
           // = RT_matrix[col][col];

          UnrolledLoop<columns>([&](auto k) {
          // #pragma unroll
          // for(int k = 0; k < columns; k++){

            auto lhs = RT_matrix[col][k];
            // auto lhs = R[k][col];
            // auto rhs = R_inverse[row].template get<k>();
            auto rhs = R_inverse[row][k];


            // auto rhs = inverseOfR[row][k];
            // auto rhs = inverseOfR[k][row];

            if(k==col){
              div_val = lhs;
            }

            if(k!=col){
              current_sum += lhs * rhs;
            }
          // }
          });

          TT result = (idMatrixValue - current_sum)/div_val;
          // PRINTF("divided by R[%d][%d]\n", col, col);
          // inverseOfR[col][row] = result;
            
          R_inverse[row][col] = result;

          // UnrolledLoop<columns>([&](auto k) {
          // // #pragma unroll
          // // for(int k = 0; k < columns; k++){
          //   if(k==col){
          //     // R_inverse[row].template get<k>() = result;
          //     R_inverse[row][k] = result;
          //     // inverseOfR[row][k] = result;
          //     // inverseOfR[k][row] = result;
          //   }
          // // }
          // });
        }

        // PRINTF("i: %d, j: %d\n", i, j);


        if(cp1 >= cp1Limit){
          col = ip1;
          cp1 = ip2;
          iter = ip1;
          row = 0;
          diagSize = diagSizem1;
          cp1Limit = nextcp1Limit;
        }
        else{
          col = cp1;
          cp1 = col + 1;
          row = row + 1;
          ip1 = iter + 1;
          ip2 = iter + 2;
          nextcp1Limit = sycl::max(RAWTriang-(diagSize - 1), columns);
          diagSizem1 = diagSize - 1;
        }

      }


      // TT rinv[rows][columns];
      // for(int i = 0; i < rows; i++){
      //   UnrolledLoop<rows>([&](auto k) {
      //      rinv[i][k] = R_inverse[i].template get<k>(); 
      //   });
      // }
      // PRINTF("R inverse\n");
      // for(int i = 0; i < rows; i++){
      //   for(int j = 0; j < columns; j++){
      //     PRINTF("(%f, %f) ", rinv[i][j].r(), rinv[i][j].i());
      //   }
      //   PRINTF("\n");
      // }

      // TT qp[rows][columns];
      // for(int i = 0; i < columns; i++){
      //   UnrolledLoop<columns>([&](auto k) {
      //      qp[k][i] = Q_matrix[i][k]; 
      //      // qp[k][i] = Q_matrix[i].template get<k>(); 
      //   });
      // }
      // PRINTF("QRI Q\n");
      // for(int i = 0; i < rows; i++){
      //   for(int j = 0; j < columns; j++){
      //     PRINTF("%f ", qp[i][j]);
      //     // PRINTF("(%f, %f) ", qp[i][j].r(), qp[i][j].i());
      //   }
      //   PRINTF("\n");
      // }

      // PRINTF("QR\n");
      // for(int i = 0; i < rows; i++){
      //   for(int j = 0; j < columns; j++){
      //     TT value = {0};
      //     for(int k = 0; k < columns; k++){
      //       value += qp[i][k] * R[k][j];
      //     }
      //     PRINTF("(%f, %f) ", value.r(), value.i());
      //   }
      //   PRINTF("\n");
      // }



      /*
        ========================================================================
        Multiply the inverse of R by the transposition of Q
        ========================================================================
      */
      for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
          TT dotProduct = {0.0};
          UnrolledLoop<rows>([&](auto k) {
            if constexpr(isComplex){
              // dotProduct += R_inverse[i].template get<k>() * 
              //               Q_matrix[j].template get<k>().conj(); 
            }
            else{
              // dotProduct += R_inverse[i].template get<k>() * 
              dotProduct += R_inverse[i][k] * 
                            // Q_matrix[j][k];
                            QT_matrix[j][k];
                            // Q_matrix[j].template get<k>();
            }
             // dotProduct += inverseOfR[i][k] * Q_matrix[j].template get<k>();
             // dotProduct += inverseOfR[k][i] * Q_matrix[j].template get<k>();
          });
          inverse[i][j] = dotProduct;
        }
      }

      /*
        ========================================================================
        Write result to the output pipe
        ========================================================================
      */

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (int col = 0; col < columns; col++) {

        // Load a single bank of the input matrix 
        column<rows, TT> pipeData;

        // Write the current column to the A_load matrix.
        UnrolledLoop<columns>([&](auto k) {
          pipeData.row[k] = inverse[col][k];
        });

        IOut::write(pipeData);
          
      } // end for col=0:columns-1


    }); // end of h.single_task
  }); // end of q.submit

  return e;
}