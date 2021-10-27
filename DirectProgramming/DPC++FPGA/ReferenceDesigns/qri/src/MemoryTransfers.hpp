#pragma once

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

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
                                  size_t bufferIdx,
      typename std::enable_if<
    (rows % QRInversionDim<isComplex, rows, columns>::NumElementsPerBank) == 0
                              >::type* = 0
                                ) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kLoadIter = dim::LoadIter;
  constexpr int kLoadIterBitSize = dim::LoadIterBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn; 

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
      TT banks[kLoadItersPerColumn][kNumElementsPerBank];
      
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; li++) {

        bool lastRow = (li%kLoadItersPerColumn) == kLoadItersPerColumn - 1;

        #pragma unroll
        for(int i=0; i<kLoadItersPerColumn-1; i++){
          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        #pragma unroll
        for(int k=0; k<kNumElementsPerBank; k++){
          banks[kLoadItersPerColumn-1][k] = 
                          A_matrix_accessor[(int)(li*kNumElementsPerBank + k)];
        }
       
        if(lastRow){
          column<rows, TT> readColumn;

          #pragma unroll
          for(int i=0; i<kLoadItersPerColumn; i++){
            #pragma unroll
            for(int k=0; k<kNumElementsPerBank; k++){
              readColumn.row[i*kNumElementsPerBank+k] = banks[i][k];
            }
          }

          AMatrixOutPipe::write(readColumn);
        }

      } // end of li

    }); // end of h
  }); // end of q submit

  return e;

}


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
                                  size_t bufferIdx,
      typename std::enable_if<
    (rows % QRInversionDim<isComplex, rows, columns>::NumElementsPerBank) != 0
                              >::type* = 0
                                ) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kLoadIter = dim::LoadIter;
  constexpr int kLoadIterBitSize = dim::LoadIterBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn; 

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
      int loadIndex = 0;

      TT banks[kLoadItersPerColumn][kNumElementsPerBank];
      
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; li++) {

        bool lastRow = (li%kLoadItersPerColumn) == kLoadItersPerColumn - 1;

        #pragma unroll
        for(int i=0; i<kLoadItersPerColumn-1; i++){
          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        #pragma unroll
        for(int k=0; k<kNumElementsPerBank; k++){

          bool outOfBounds = lastRow && 
                ((k % kNumElementsPerBank) > ((rows-1) % kNumElementsPerBank));

          if(!outOfBounds){
            banks[kLoadItersPerColumn-1][k] = A_matrix_accessor[loadIndex + k];
          }
        }

        int readElements = (rows % kNumElementsPerBank != 0) 
                                      && lastRow ?
                                      rows % kNumElementsPerBank :  
                                      kNumElementsPerBank;
        // Increase the bank index
        loadIndex += readElements;



        if(lastRow){
          column<rows, TT> readColumn;

          #pragma unroll
          for(int i=0; i<kLoadItersPerColumn; i++){
            #pragma unroll
            for(int k=0; k<kNumElementsPerBank; k++){
              if(i*kNumElementsPerBank+k < rows){
                readColumn.row[i*kNumElementsPerBank+k] = banks[i][k];
              }
            }
          }

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
          typename InverseMatrixInPipe, // Inverse matrix input pipe from the 
                                  // compute kernel
          short numBuffers        // number of buffers to rotate with
          >
sycl::event LocalMemoryToDDRCopy( 
                        sycl::queue& q, 
                        sycl::buffer<TT, 1> * inverse_matrix_buffer[numBuffers],
                        size_t bufferIdx,
                        typename std::enable_if<
    (rows % QRInversionDim<isComplex, rows, columns>::NumElementsPerBank) == 0
                                                >::type* = 0
                                ) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kStoreIter = dim::StoreIter;
  constexpr int kStoreIterBitSize = dim::StoreIterBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn; 
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffers
    sycl::accessor inverse_matrix_accessor(*inverse_matrix_buffer[bufferIdx], h,
                                              sycl::write_only, sycl::no_init);

    sycl::stream out(64000, 64000, h);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {
      
      /*
        ================================================================
        Copy the result from on-chip memory to DDR memory.
        ================================================================
      */
      TT banks[kLoadItersPerColumn][kNumElementsPerBank];
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                si++) {

        // [[intel::bankwidth(kBankWidth)]]        // NO-FORMAT: Attribute
        if((si % kNumBanks) == 0){
          column<rows, TT> pipeData;
          pipeData = InverseMatrixInPipe::read();

          #pragma unroll
          for(int i=0; i<kLoadItersPerColumn; i++){
            UnrolledLoop<kNumElementsPerBank>([&](auto k) {
              banks[i][k] = pipeData.row[i*kNumElementsPerBank+k];
            });
          }
        }

        #pragma unroll 
        for(int k = 0; k<kNumElementsPerBank; k++){
          inverse_matrix_accessor[(int)(si*kNumElementsPerBank + k)] = 
                                                                    banks[0][k];
        }

        #pragma unroll
        for(int i=0; i<kLoadItersPerColumn; i++){
          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }
      } // end for si=0:kStoreIter-1
    }); // end of single task
  }); // end of q submit

  return e;

}

template <typename kernelName,    // Name to use for the Kernel
          bool isComplex,         // Helps identify the correct bank size
          typename TT,            // The datatype for the computation
          int rows,               // Number of rows in the incoming A matrix
          int columns,            // Number of columns in the incoming A
                                  // matrix, must be <= rows
          typename InverseMatrixInPipe, // Inverse matrix input pipe from the 
                                  // compute kernel
          short numBuffers        // number of buffers to rotate with
          >
sycl::event LocalMemoryToDDRCopy( 
                        sycl::queue& q, 
                        sycl::buffer<TT, 1> * inverse_matrix_buffer[numBuffers],
                        size_t bufferIdx,
                        typename std::enable_if<
    (rows % QRInversionDim<isComplex, rows, columns>::NumElementsPerBank) != 0
                                                >::type* = 0
                                ) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kStoreIter = dim::StoreIter;
  constexpr int kStoreIterBitSize = dim::StoreIterBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn; 
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffers
    sycl::accessor inverse_matrix_accessor(*inverse_matrix_buffer[bufferIdx], h,
                                              sycl::write_only, sycl::no_init);

    sycl::stream out(64000, 64000, h);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {
      
      /*
        ================================================================
        Copy the result from on-chip memory to DDR memory.
        ================================================================
      */
      int qr_idx = 0;
      TT banks[kLoadItersPerColumn][kNumElementsPerBank];
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                si++) {

        // [[intel::bankwidth(kBankWidth)]]        // NO-FORMAT: Attribute
        if((si % kNumBanks) == 0){
          column<rows, TT> pipeData;
          pipeData = InverseMatrixInPipe::read();

          #pragma unroll
          for(int i=0; i<kLoadItersPerColumn; i++){
            UnrolledLoop<kNumElementsPerBank>([&](auto k) {
              banks[i][k] = pipeData.row[i*kNumElementsPerBank+k];
            });
          }
        }

        bool lastRow = si % kNumBanks == kNumBanks-1; 

        #pragma unroll 
        for(int k = 0; k<kNumElementsPerBank; k++){

          bool outOfBounds = lastRow && (k > ((rows-1) % kNumElementsPerBank));

          if(!outOfBounds){
            inverse_matrix_accessor[qr_idx + k] = banks[0][k];
          }
        }

        #pragma unroll
        for(int i=0; i<kLoadItersPerColumn; i++){
          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        int wroteElements = lastRow ? rows % kNumElementsPerBank :  
                                                            kNumElementsPerBank;
        qr_idx += wroteElements;
              
      } // end for si=0:kStoreIter-1
    }); // end of single task
  }); // end of q submit

  return e;

}