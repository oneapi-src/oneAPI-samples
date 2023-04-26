#ifndef __STREAMING_CovMM_HPP__
#define __STREAMING_CovMM_HPP__

#include "tuple.hpp"
#include "constexpr_math.hpp"
#include "unrolled_loop.hpp"

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

namespace fpga_linalg {

template <typename T,        // The datatype for the computation
          unsigned rows,          // Number of rows in the A matrices
          unsigned columns,       // Number of columns in the A matrices

          unsigned blockSize,  // number of parallel mult and add 
          unsigned pipe_size,     // Number of elements read/write per pipe
                             // operation
          typename AIn,      // A matrix input pipe, receive pipe_size
                             // elements from the pipe with each read
          typename AOut     // Q matrix output pipe, send pipe_size
                             // elements to the pipe with each write
          >




// this is a kernel for preprocessing input samples with multiple features 
// input is essentially Nxp matrix - A 
// Preprocessing Algorithm  
// 1. compute mean feature, subtract the mean feature from all the samples(A_new) 
// 2. Standardize the samples for each feature and make the variance to one  
// 3. compute the covariance matrix C = A_new@transpose(A_new)

// In this impmentation step 3 is executed first and computed value is adjusted for 
// step 1 and 2. This allows blockwise matrix multiplication, saving required memory 
// for larger matrices 


// Inorder to handle the larger matrix size input block is partioned into pxp blocks 
// input is organized such that samples for each feature is sequential 
// A@transpose(A) = Sigma_{0}^{blocks} A_blk@transpose(A_blk) 

// A_new[i][j] = (A[i][j] - mean[i])/variance[i]; // j - sample index, i - feature index
// C[i][j] = Dot(A_new[i][] , A_new[j][] )
// C[i][j] = Dot((A[i][] - mean[i])/variance[i] , (A[j][] - mean[j])/variance[j])
// C[i][j] = (1.0f/variance[i]^2 ) * Dot((A[i][] - mean[i]), (A[j][] - mean[j]))
// C[i][j] = (1.0f/(variance[i]*variance[j])) * Dot((A[i][] - mean[i]), (A[j][] - mean[j]))
// C[i][j] = (1.0f/(variance[i]*variance[j])) * (Dot(A[i][], A[j][]) - N* mean[i]*mean[j])


// mean[i] = sum(Dot(A[i][])/N
// var[i] = sqrt((Dot(A[i][], A[j][]) - N* mean[i]*mean[j])/N)

struct StreamingMM{
    void operator()() const {
    
    using row_tuple = fpga_tools::NTuple<T, rows>;
    using pipe_tuple = fpga_tools::NTuple<T, pipe_size>;

    constexpr int kColBlocks = (columns+rows-1)/rows;
    constexpr int kRowBlocks = (rows+pipe_size-1)/pipe_size;
    constexpr int kLoopItr = rows*kRowBlocks;

    constexpr int kColBlockBitSize = fpga_tools::BitsForMaxValue<kColBlocks + 1>();
    constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopItr + 1>();

    constexpr int maxRow = (rows > pipe_size) ? rows : pipe_size;
    constexpr int kRowBitSize = fpga_tools::BitsForMaxValue<maxRow + 1>();



    while(1){

      // storing in a internal matrix 

        // NO-FORMAT: Attribute
      double MatrixC[rows][rows], MatrixCW[rows][rows];
      // row_tuple  Avg_G;
      double Avg[rows];
      pipe_tuple pipe_read;
      double digValM[rows];




      for(ac_int<kColBlockBitSize, false> blk = 0; blk < kColBlocks; blk++){

        // loading block data onchip memory 
        // samples for a feature comes sequentially 
        row_tuple MatrixA[rows];
        for(ac_int<kLoopIterBitSize, false> itr = 0; itr < kLoopItr; itr++){
          ac_int<kRowBitSize, false> i_ll = itr / kRowBlocks;
          ac_int<kRowBitSize, false> j_ll = itr % kRowBlocks;

          pipe_read = AIn::read();
          row_tuple rowblk;
          fpga_tools::UnrolledLoop<kRowBlocks>([&](auto k) {
            fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
              if(k == j_ll){
                if constexpr (k*pipe_size+t < rows){
                  rowblk.template get<k*pipe_size+t> () = pipe_read.template get<t>();
                }
              }
            });
          });

          MatrixA[i_ll] = rowblk;

        }


        // computing the covariance matrix block wise and accumulating 
        T row1[rows], row2[rows], row_temp[rows];
        for(ac_int<kRowBitSize, false> i_ll = 0; i_ll < rows; i_ll++){


          fpga_tools::UnrolledLoop<rows>([&](auto t) {
            row1[t] = row_temp[t];
          });

          if(blk == 0){
            Avg[i_ll] = 0;
          }

          T avg = 0;
          [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
          // [[intel::ivdep(rowSumL, rows)]]
          for(int j_ll = 0; j_ll < rows; j_ll++){

            fpga_tools::UnrolledLoop<rows>([&](auto t) {
              row2[t] = MatrixA[j_ll].template get <t>();
              if(j_ll == i_ll + 1){
                row_temp[t] = row2[t];
              }
              if(i_ll == 0 && j_ll == 0){
                row1[t] = row2[t];
              } 
            });

            T rowSum = 0;
            fpga_tools::UnrolledLoop<rows>([&](auto t) {
              T row1Elem = row1[t];
              rowSum += row1Elem * row2[t];
            });

            avg += row1[j_ll];

            T sum_a = rowSum;
            double sum_b = blk == 0 ? 0 : MatrixC[i_ll][j_ll];
            double sum = sum_a + sum_b; 

            MatrixC[i_ll][j_ll] = sum;
            MatrixCW[i_ll][j_ll] = sum;
            
            if (i_ll == j_ll){
              digValM[i_ll] = sum;
            }

          } // end of j_ll 

          Avg[i_ll] += avg/columns;
        } // end of i_ll

      } // end of blk

      
      // adjusting based on variance and mean
      // C[i][j] = (1.0f/(variance[i]*variance[j])) * (Dot(A[i][], A[j][]) - N* mean[i]*mean[j])
      // mean[i] = sum(Dot(A[i][])/N
      // var[i] = sqrt((Dot(A[i][], A[j][]) - N* mean[i]*mean[j])/N)
      pipe_tuple pipe_write;
      double avg1, avg2, avg_temp;
      double digVal1, digVal2, dig_temp; 
      for(ac_int<kRowBitSize, false> i_ll = 0; i_ll < rows; i_ll++){
        for(ac_int<kRowBitSize, false> j_ll = 0; j_ll < rows; j_ll++){
          T loadVal;
          row_tuple loadRow;
          fpga_tools::UnrolledLoop<rows>([&](auto t) {
            loadRow.template get<t>() = MatrixCW[i_ll][t];
            if(j_ll == t){
              loadVal = loadRow.template get<t>();
            }
          });

          //---------------------------
          digVal2 = digValM[j_ll];
          avg2 = Avg[j_ll];
          if(j_ll == i_ll + 1){
            dig_temp = digVal2;
            avg_temp = avg2;
          }

          if(i_ll == 0 && j_ll == 0){
            digVal1 = digVal2;
            avg1 = avg2;
          } else if(j_ll == 0){
            digVal1 = dig_temp;
            avg1 = avg_temp;
          }

          
          T cov_i_i = digVal1 - columns * avg1 * avg1;
          T cov_j_j = digVal2 - columns * avg2 * avg2;


          T cov_i_j_tmp = loadVal - columns * avg1 * avg2;
          T cov_i_j = cov_i_j_tmp/sqrt(cov_i_i*cov_j_j);





          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if(t == j_ll % pipe_size){
              pipe_write.template get<t> () = cov_i_j;
            }
          });

          if(j_ll % pipe_size == pipe_size -1 || j_ll == rows-1){
            AOut::write(pipe_write);
          }

        } // end of j_ll
      } // endo of i_ll



    } // end of while

  }; // end of operator()
}; // end of struct{}

} // namespace fpga_linalg


#endif 
