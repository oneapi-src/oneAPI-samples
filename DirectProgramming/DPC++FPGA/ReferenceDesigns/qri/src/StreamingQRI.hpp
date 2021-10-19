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

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      // [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      Column Q_matrix[columns];

      TT R[rows][columns];

      TT inverse[rows][columns];

      /*
        ======================================================================
        Read the R matrix from the pipe
        ======================================================================
      */

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

      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; 
                                                                li++) {
        // Load a single bank of the input matrix 
        PipeType pipeData = QIn::read();

        // Write the current bank to the A_load matrix.
        ac_int<BitsForMaxValue<kNumBanks>(), false> bankToWrite = 
                                                      li % (kNumBanks);
        ac_int<kLiNumBankBitSize, false> rowToWrite = li / kNumBanks;

        
        UnrolledLoop<rows>([&](auto r) {
          bool getr = rowToWrite == r;
          UnrolledLoop<kNumBanks>([&](auto b) {
            bool getb = getr && (bankToWrite == b);

            UnrolledLoop<kNumElementsPerBank>([&](auto t) {
              constexpr auto idx = b * kNumElementsPerBank + t;
              if(getb){
                 if constexpr (idx < rows){
                   Q_matrix[idx].template get<r>() = 
                                      pipeData.template get<t>();
                 }
              }
            });

          });
          // Delay data signals to create a vine-based data 
          // distribution to lower signal fanout.
          pipeData = sycl::ext::intel::fpga_reg(pipeData);
        });
      }

      /*
        ======================================================================
        Compute the inverse of R
        ======================================================================
      */
/*
          [[intel::bankwidth(kBankwidth), intel::numbanks(kNumBanks)]] 
          struct {
            TT c[rows];
          } inverseOfR[columns];


*/
      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      // [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      Row R_inverse[columns];


      TT inverseOfR[rows][columns];
      for(int i=0; i<rows; i++){
        UnrolledLoop<columns>([&](auto k) {
          R_inverse[i].template get<k>() = {0};
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
      int cp1Limit = RAWTriang-columns-columns > 0 ? RAWTriang-columns : columns;
      int nextcp1Limit = RAWTriang-columns-1-columns > 0 ? RAWTriang-columns-1 : columns;

      // [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      [[intel::ivdep(RAWTriang)]]  // NO-FORMAT: Attribute
      for(int it = 0; it < kTotalIterations; it++){
        if(row<rows && col<columns){
          TT idMatrixValue = row == col ? TT{1} : TT{0};

          TT current_sum = {0};
          TT div_val = R[col][col];

          UnrolledLoop<columns>([&](auto k) {
          // #pragma unroll
          // for(int k = 0; k < columns; k++){
            auto lhs = R[k][col];
            auto rhs = R_inverse[row].template get<k>();
            // auto rhs = inverseOfR[row][k];
            // auto rhs = inverseOfR[k][row];

            if(k!=col){
              current_sum += lhs * rhs;
            }
          // }
          });

          TT result = (idMatrixValue - current_sum)/div_val;
          // PRINTF("divided by R[%d][%d]\n", col, col);
          // inverseOfR[col][row] = result;

          UnrolledLoop<columns>([&](auto k) {
          // #pragma unroll
          // for(int k = 0; k < columns; k++){
            if(k==col){
              R_inverse[row].template get<k>() = result;
              // inverseOfR[row][k] = result;
              // inverseOfR[k][row] = result;
            }
          // }
          });
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

      /*
        ======================================================================
        Multiply the inverse of R by the transposition of Q
        ======================================================================
      */
      for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
          TT dotProduct = {0.0};
          UnrolledLoop<rows>([&](auto k) {
             dotProduct += R_inverse[i].template get<k>() * Q_matrix[j].template get<k>(); 
             // dotProduct += inverseOfR[i][k] * Q_matrix[j].template get<k>();
             // dotProduct += inverseOfR[k][i] * Q_matrix[j].template get<k>();
          });
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
              pipeData.template get<k>() = get[t] ? inverse[siNumBank][rowIdx] :
                                // Q_matrix[siNumBank].template get<rowIdx>() : 
                        sycl::ext::intel::fpga_reg(pipeData.template get<k>());
            }
          });
        });

        IOut::write(pipeData);
           
      } // end for si=0:kStoreIter-1

    }); // end of h.single_task
  }); // end of q.submit

  return e;
}