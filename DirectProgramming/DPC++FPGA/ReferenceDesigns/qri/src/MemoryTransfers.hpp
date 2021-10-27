#pragma once

#include "Utils.hpp"
#include "QRInversionDim.hpp"

template <typename kernelName,      // Name to use for the Kernel
          bool isComplex,           // Helps identify the correct bank size
          typename TT,              // The datatype for the computation
          int rows,                 // Number of rows in the incoming A matrix
          int columns,              // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          typename AMatrixOutPipe,  // A matrix input, receive a full column
                                    // of complex numbers with each read,
                                    // wrapped in NTuple
          short numBuffers          // number of buffers to rotate with
          >
sycl::event DDRToLocalMemoryCopy( sycl::queue& q, 
                                  sycl::buffer<TT, 1> * A_buffer[numBuffers],
                                  size_t bufferIdx) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kAMatrixSize = dim::AMatrixSize;
  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kBankWidth = dim::BankWidth;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kNumBanksNextPow2 = dim::NumBanksNextPow2;
  constexpr bool kNonCompleteIter = dim::NonCompleteIter;
  constexpr int kExtraIter = dim::ExtraIter;
  constexpr int kLoadIter = dim::LoadIter;
  constexpr int kLoadIterBitSize = dim::LoadIterBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn; 

  using PipeType = NTuple<TT, kNumElementsPerBank>;
  // using PipeTypeFull = NTuple<TT, columns>;

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffers
    sycl::accessor A_matrix_accessor(*A_buffer[bufferIdx], h, sycl::read_only);

    sycl::stream out(64000, 64000, h);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      /*
        ================================================================
        Copy data from DDR memory to on-chip memory.
        ================================================================
      */
      // Get the index of the first bank of the current matrix l
      int loadBankIndex = 0;

      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; li++) {
        column<rows, TT> readColumn;

        bool lastRow = false;

        // if constexpr(kNonCompleteIter){
          lastRow = (li%kLoadItersPerColumn) == kLoadItersPerColumn - 1;
        // }

        UnrolledLoop<kNumElementsPerBank>([&](auto k) {

          bool outOfBounds = false;
          if constexpr(kNonCompleteIter){
           outOfBounds = lastRow && 
                ((k % kNumElementsPerBank) > ((rows-1) % kNumElementsPerBank));
          }

          if(!outOfBounds){
            readColumn.row[k + 
                            (int(li)%kLoadItersPerColumn)*kNumElementsPerBank] =
                                           A_matrix_accessor[loadBankIndex + k];
          }
        });

        if constexpr(kNonCompleteIter){
          int readElements = (rows % kNumElementsPerBank != 0) 
                                        && lastRow ?
                                        rows % kNumElementsPerBank :  
                                        kNumElementsPerBank;

          // Increase the bank index
          loadBankIndex += readElements;
        }
        else{
          loadBankIndex += kNumElementsPerBank;
        }


        if(lastRow){
          AMatrixOutPipe::write(readColumn);
        }

      } // end of li

    }); // end of h
  }); // end of q submit

  return e;

}

template <typename kernelName,    // Name to use for the Kernel
          bool isComplex,         // Helps identify the correct bank size
          typename TT,            // The datatype for the computation
          int rows,               // Number of rows in the incoming A matrix
          int columns,            // Number of columns in the incoming A
                                  // matrix, must be <= rows
          typename InverseMatrixOutPipe, // Q matrix input pipe from the compute kernel
          short numBuffers        // number of buffers to rotate with
          >
sycl::event LocalMemoryToDDRCopy( sycl::queue& q, 
                                  sycl::buffer<TT, 1> * inverse_matrix_buffer[numBuffers],
                                  size_t bufferIdx) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kBankWidth = dim::BankWidth;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kRMatrixSize = dim::RMatrixSize;
  constexpr bool kNonCompleteIter = dim::NonCompleteIter;
  constexpr int kExtraIter = dim::ExtraIter;
  constexpr int kStoreIter = dim::StoreIter;
  constexpr int kStoreIterBitSize = dim::StoreIterBitSize;
  constexpr bool kNonCompleteBank = dim::NonCompleteBank;
  constexpr int kExtraBank = dim::ExtraBank;
  constexpr int kNumRBanks = dim::NumRBanks;

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffers
    sycl::accessor inverse_matrix_accessor(*inverse_matrix_buffer[bufferIdx], h, sycl::write_only, 
                                                                sycl::no_init);

    sycl::stream out(64000, 64000, h);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {
      
      /*
        ================================================================
        Copy the result from on-chip memory to DDR memory.
        ================================================================
      */
      int qr_idx = 0;
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                si++) {
        column<rows, TT> pipeData;
        if((si % kNumBanks) == 0){
          pipeData = InverseMatrixOutPipe::read();
        }


        bool lastRow = false;
        if constexpr(kNonCompleteIter){
          lastRow = si % kNumBanks == kNumBanks-1; 
        } 

        #pragma unroll 
        for(int k = 0; k<kNumElementsPerBank; k++){
          bool outOfBounds = false;
          if constexpr(kNonCompleteIter){
            outOfBounds = lastRow && 
                  (k > ((rows-1) % kNumElementsPerBank));
          }

          if(!outOfBounds){
            inverse_matrix_accessor[qr_idx + k] = pipeData.row[k +  (si % kNumBanks)*kNumElementsPerBank];
          }
        }

        if constexpr(kNonCompleteIter){
          int wroteElements = lastRow ? rows % kNumElementsPerBank :  
                                                    kNumElementsPerBank;
          qr_idx += wroteElements;
        }
        else{
          qr_idx += kNumElementsPerBank;
        }                
      } // end for si=0:kStoreIter-1
    }); // end of single task
  }); // end of q submit

  return e;

}