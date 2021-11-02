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

/*
  Read a matrix from DDR and write it to a pipe, column by column
  This implementation is used for matrices that have a number of rows
  that is a multiple of the number of elements per DDR burst read.
  Another version of this function  is written below and will be selected 
  automatically at compile time if the row count is not a multiple of the burst 
  size.
*/
template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // Datatype of the elements of the matrix
          int rows,                 // Number of rows of the matrix
          int columns,              // Number of columns of the matrix
          int numElemPerBank,       // Number of TT elems per DDR burst access
          int matrixCount,          // Number of matrices to read
                                    // from the buffer sequentially
          typename matrixPipe       // Output matrix pipe, send a full column
                                    // of potentially complex TT elements 
                                    // with each write
          >
sycl::event MatrixReadFromDDRToPipeByColumns( 
            sycl::queue& q,                     // Device queue
            sycl::buffer<TT, 1> * MatrixBuffer, // Input matrix buffer
            typename std::enable_if<(rows % numElemPerBank) == 0>::type* = 0) {

  // Number of DDR burst reads of numElemPerBank required to read a full column 
  constexpr int kLoopIterPerColumn = rows/numElemPerBank; 
  // Number of DDR burst reads of numElemPerBank to read the entire matrix
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the input matrix
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::read_only);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // "Shift register" that will contain a full column after 
      // kLoopIterPerColumn iterations.
      // Each DDR burst read will write to banks[kLoopIterPerColumn-1] and
      // and each loop iteration each banks[x] will be assigned banks[x-1]
      // This ensures that the fanout is kept to a minimum
      TT banks[kLoopIterPerColumn][numElemPerBank];
      
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        // Perform the register shifting of the banks
        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn-1; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        // Perform the DDR burst read of numElemPerBank elements
        #pragma unroll
        for(int k=0; k<numElemPerBank; k++){
          banks[kLoopIterPerColumn-1][k] = 
                                  matrixAccessor[(int)(li*numElemPerBank + k)];
        }

        // Check if we just read the last DDR burst of the current column 
        bool lastBurstOfCol = (li%kLoopIterPerColumn) == kLoopIterPerColumn - 1;

        // If so, we are going to copy the column stored in banks to the pipe
        if(lastBurstOfCol){
          // The pipe type
          column<rows, TT> readColumn;

          // Copy the banks data to the correct datatype for the pipe write
          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            #pragma unroll
            for(int k=0; k<numElemPerBank; k++){
              readColumn.row[i*numElemPerBank+k] = banks[i][k];
            }
          }

          // Send the column over the pipe
          matrixPipe::write(readColumn);
        }
      } // end of li
    }); // end of h
  }); // end of q submit

  return e;
}

/*
  Read a matrix from DDR and write it to a pipe, column by column
  This implementation is used for matrices that have a number of rows
  that is not a multiple of the number of elements per DDR burst read.
  Another version of this function is written above and will be selected 
  automatically at compile time if the row count is a multiple of the burst 
  size.
*/
template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // Datatype of the elements of the matrix
          int rows,                 // Number of rows of the matrix
          int columns,              // Number of columns of the matrix
          int numElemPerBank,       // Number of TT elems per DDR burst access
          int matrixCount,          // Number of matrices to read
                                    // from the buffer sequentially
          typename matrixPipe       // Output matrix pipe, send a full column
                                    // of potentially complex TT elements 
                                    // with each write
          >
sycl::event MatrixReadFromDDRToPipeByColumns( 
            sycl::queue& q,                     // Device queue
            sycl::buffer<TT, 1> * MatrixBuffer, // Input matrix buffer
            typename std::enable_if<(rows % numElemPerBank) != 0>::type* = 0) {

  // Number of DDR burst reads of numElemPerBank required to read a full column 
  constexpr int kLoopIterPerColumn = rows/numElemPerBank + 1; 
  // Number of DDR burst reads of numElemPerBank to read the entire matrix
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the input matrix
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::read_only);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // Keep track of the current element index in the read buffer
      int loadIndex = 0;

      // "Shift register" that will contain a full column after 
      // kLoopIterPerColumn iterations.
      // Each DDR burst read will write to banks[kLoopIterPerColumn-1] and
      // and each loop iteration each banks[x] will be assigned banks[x-1]
      // This ensures that the fanout is kept to a minimum
      TT banks[kLoopIterPerColumn][numElemPerBank];
      
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        // Perform the register shifting of the banks
        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn-1; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        // Check if we are reading the last DDR burst of the current column 
        bool lastBurstOfCol = (li%kLoopIterPerColumn) == kLoopIterPerColumn - 1;

        #pragma unroll
        for(int k=0; k<numElemPerBank; k++){
          // Check if the current read index is beyond the end of the current
          // matrix column
          bool outOfBounds = lastBurstOfCol && 
                          ((k % numElemPerBank) > ((rows-1) % numElemPerBank));

          // Only perform the DDR reads that are relevant (and don't access a
          // memory address that may be beyond the buffer last address)
          if(!outOfBounds){
            banks[kLoopIterPerColumn-1][k] = matrixAccessor[loadIndex + k];
          }
        }

        // Update the current element index in the read buffer according
        // to the read size of the current iteration
        loadIndex += lastBurstOfCol ? rows % numElemPerBank : numElemPerBank;

        // If we read the last burst of the current columns, we are going to 
        // copy the column stored in banks to the pipe
        if(lastBurstOfCol){
          // The pipe type
          column<rows, TT> readColumn;

          // Copy the banks data to the correct datatype for the pipe write
          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            #pragma unroll
            for(int k=0; k<numElemPerBank; k++){
              if(i*numElemPerBank+k < rows){
                readColumn.row[i*numElemPerBank+k] = banks[i][k];
              }
            }
          }

          // Send the column over the pipe
          matrixPipe::write(readColumn);
        }
      } // end of li
    }); // end of h
  }); // end of q submit

  return e;
}

template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // The datatype for the computation
          int rows,                 // Number of rows in the incoming A matrix
          int columns,              // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          int numElemPerBank,       // Number of TT elems per DDR burst access
          int matrixCount,          // Number of matrices to read
                                    // from the buffer sequentially
          typename matrixPipe       // Input Matrix, receive a full column
                                    // of potentially complex TT elements 
                                    // with each read
          >
sycl::event MatrixReadPipeByColumnsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * MatrixBuffer,
            typename std::enable_if<(rows % numElemPerBank) == 0>::type* = 0) {

  constexpr int kLoopIterPerColumn = rows / numElemPerBank; 
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      TT banks[kLoopIterPerColumn][numElemPerBank];
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> si = 0; si < kLoopIter; si++) {

        if((si % kLoopIterPerColumn) == 0){
          column<rows, TT> pipeRead;
          pipeRead = matrixPipe::read();

          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            UnrolledLoop<numElemPerBank>([&](auto k) {
              banks[i][k] = pipeRead.row[i*numElemPerBank+k];
            });
          }
        }

        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          matrixAccessor[(int)(si*numElemPerBank + k)] = banks[0][k];
        }

        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }
      } // end for si=0:kLoopIter-1
    }); // end of single task
  }); // end of q submit

  return e;
}

template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // The datatype for the computation
          int rows,                 // Number of rows in the incoming A matrix
          int columns,              // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          int numElemPerBank,       // Number of TT elems per DDR burst access
          int matrixCount,          // Number of matrices to read
                                    // from the buffer sequentially
          typename matrixPipe       // Input Matrix, receive a full column
                                    // of potentially complex TT elements 
                                    // with each read
          >
sycl::event MatrixReadPipeByColumnsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * MatrixBuffer,
            typename std::enable_if<(rows % numElemPerBank) != 0>::type* = 0) {

  constexpr int kLoopIterPerColumn = rows / numElemPerBank + 1; 
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      TT banks[kLoopIterPerColumn][numElemPerBank];
      int writeIdx = 0;
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> si = 0; si < kLoopIter; si++) {

        if((si % kLoopIterPerColumn) == 0){
          column<rows, TT> pipeRead;
          pipeRead = matrixPipe::read();

          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            UnrolledLoop<numElemPerBank>([&](auto k) {
              banks[i][k] = pipeRead.row[i*numElemPerBank+k];
            });
          }
        }

        bool lastRow = si % kLoopIterPerColumn == kLoopIterPerColumn-1; 

        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){

          bool outOfBounds = lastRow && (k > ((rows-1) % numElemPerBank));

          if(!outOfBounds){
            matrixAccessor[writeIdx + k] = banks[0][k];
          }
        }

        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        int wroteElements = lastRow ? rows % numElemPerBank : numElemPerBank;
        writeIdx += wroteElements;
              
      } // end for si=0:kLoopIter-1
    }); // end of single task
  }); // end of q submit

  return e;
}

template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // The datatype for the computation
          int size,                 // Number of rows in the incoming A matrix
          int numElemPerBank,       // Number of TT elems per DDR burst access
          int vectorCount,          // Number of vectors to read
                                    // from the buffer sequentially
          typename vectorPipe       // Input vector, receive an element
                                    // of potentially complex TT  with each read
          >
sycl::event VectorReadPipeByElementsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * VectorBuffer,
            typename std::enable_if<(size % numElemPerBank) == 0>::type* = 0) {

  constexpr int kLoopIter = (size / numElemPerBank) * vectorCount;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor vectorAccessor(*VectorBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        TT bank[numElemPerBank];
        for(int k = 0; k<numElemPerBank; k++){
          bank[k] = vectorPipe::read();
        }

        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          vectorAccessor[(int)(li*numElemPerBank + k)] = bank[k];
        }              
      } // end for li=0:kLoopIter-1
    }); // end of single task
  }); // end of q submit

  return e;
}

template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // The datatype for the computation
          int size,                 // Number of rows in the incoming A matrix
          int numElemPerBank,       // Number of TT elems per DDR burst access
          int vectorCount,          // Number of vectors to read
                                    // from the buffer sequentially
          typename vectorPipe       // Input vector, receive an element
                                    // of potentially complex TT  with each read
          >
sycl::event VectorReadPipeByElementsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * VectorBuffer,
            typename std::enable_if<(size % numElemPerBank) != 0>::type* = 0) {

  constexpr int kLoopIter = (size / numElemPerBank + 1) * vectorCount;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor vectorAccessor(*VectorBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      int vectorIdx = 0;
      int vectorCountIdx = 0;
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      [[intel::ivdep]]                    // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        TT bank[numElemPerBank];
        for(int k = 0; k<numElemPerBank; k++){
          if(vectorIdx + k < size){
            bank[k] = vectorPipe::read();
          }
        }

        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          if ((vectorIdx + k) < size){
            vectorAccessor[vectorCountIdx * size + vectorIdx + k] = bank[k];
          }
        }             

        int vectorIdxPlusNumElemPerBank = vectorIdx + numElemPerBank;
        if(vectorIdxPlusNumElemPerBank > size){
          vectorIdx = 0;
          vectorCountIdx += 1;
        }
        else{
          vectorIdx = vectorIdxPlusNumElemPerBank;
        }

      } // end for li=0:kLoopIter-1
    }); // end of single task
  }); // end of q submit

  return e;
}