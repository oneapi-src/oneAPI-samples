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

template <typename kernelName,      // Name to use for the Kernel
          typename TT,              // The datatype for the computation
          int rows,                 // Number of rows in the incoming A matrix
          int columns,              // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          int numElemPerBank,       // Number of TT elems per DDR burst access
          typename matrixPipe       // Output Matrix, send a full column
                                    // of potentially complex TT elements 
                                    // with each read
          >
sycl::event MatrixReadFromDDRToPipeByColumns( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * MatrixBuffer,
            typename std::enable_if<(rows % numElemPerBank) == 0>::type* = 0) {

  constexpr int kLoopIterPerColumn = rows/numElemPerBank; 
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::read_only);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      TT banks[kLoopIterPerColumn][numElemPerBank];
      
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        bool lastRow = (li%kLoopIterPerColumn) == kLoopIterPerColumn - 1;

        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn-1; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        #pragma unroll
        for(int k=0; k<numElemPerBank; k++){
          banks[kLoopIterPerColumn-1][k] = 
                                  matrixAccessor[(int)(li*numElemPerBank + k)];
        }

        if(lastRow){
          column<rows, TT> readColumn;

          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            #pragma unroll
            for(int k=0; k<numElemPerBank; k++){
              readColumn.row[i*numElemPerBank+k] = banks[i][k];
            }
          }
          
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
          typename matrixPipe       // Output matrix, send a full column
                                    // of potentially complex TT elements 
                                    // with each read
          >
sycl::event MatrixReadFromDDRToPipeByColumns( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * MatrixBuffer,
            typename std::enable_if<(rows % numElemPerBank) != 0>::type* = 0) {


  constexpr int kLoopIterPerColumn = rows/numElemPerBank + 1; 
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::read_only);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // Get the index of the first bank of the current matrix l
      int loadIndex = 0;

      TT banks[kLoopIterPerColumn][numElemPerBank];
      
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        bool lastRow = (li%kLoopIterPerColumn) == kLoopIterPerColumn - 1;

        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn-1; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        #pragma unroll
        for(int k=0; k<numElemPerBank; k++){

          bool outOfBounds = lastRow && 
                ((k % numElemPerBank) > ((rows-1) % numElemPerBank));

          if(!outOfBounds){
            banks[kLoopIterPerColumn-1][k] = matrixAccessor[loadIndex + k];
          }
        }

        loadIndex += (rows % numElemPerBank != 0) && lastRow ?
                                                        rows % numElemPerBank :  
                                                        numElemPerBank;

        if(lastRow){
          column<rows, TT> readColumn;

          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            #pragma unroll
            for(int k=0; k<numElemPerBank; k++){
              if(i*numElemPerBank+k < rows){
                readColumn.row[i*numElemPerBank+k] = banks[i][k];
              }
            }
          }

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
          typename matrixPipe       // Input Matrix, receive a full column
                                    // of potentially complex TT elements 
                                    // with each read
          >
sycl::event MatrixReadPipeByColumnsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * MatrixBuffer,
            typename std::enable_if<(rows % numElemPerBank) == 0>::type* = 0) {

  constexpr int kLoopIterPerColumn = rows / numElemPerBank; 
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
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
          typename matrixPipe       // Input Matrix, receive a full column
                                    // of potentially complex TT elements 
                                    // with each read
          >
sycl::event MatrixReadPipeByColumnsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * MatrixBuffer,
            typename std::enable_if<(rows % numElemPerBank) != 0>::type* = 0) {

  constexpr int kLoopIterPerColumn = rows / numElemPerBank + 1; 
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
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
          typename vectorPipe       // Input vector, receive an element
                                    // of potentially complex TT  with each read
          >
sycl::event VectorReadPipeByElementsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * VectorBuffer,
            typename std::enable_if<(size % numElemPerBank) == 0>::type* = 0) {

  constexpr int kLoopIter = size / numElemPerBank;
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
          typename vectorPipe       // Input vector, receive an element
                                    // of potentially complex TT  with each read
          >
sycl::event VectorReadPipeByElementsToDDR( 
            sycl::queue& q, 
            sycl::buffer<TT, 1> * VectorBuffer,
            typename std::enable_if<(size % numElemPerBank) != 0>::type* = 0) {

  constexpr int kLoopIter = size / numElemPerBank + 1;
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer
    sycl::accessor vectorAccessor(*VectorBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      int vectorIdx = 0;
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      [[intel::ivdep]]                    // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        TT bank[numElemPerBank];
        for(int k = 0; k<numElemPerBank; k++){
          if(vectorIdx + k < size){
            bank[k] = vectorPipe::read();
            PRINTF("%f \n", bank[k]);
          }
        }

        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          if ((vectorIdx + k) < size){
            vectorAccessor[vectorIdx + k] = bank[k];
          }
        }             

        vectorIdx += numElemPerBank;

      } // end for li=0:kLoopIter-1
    }); // end of single task
  }); // end of q submit

  return e;
}