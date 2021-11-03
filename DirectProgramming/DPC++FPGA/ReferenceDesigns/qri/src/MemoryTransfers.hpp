#pragma once

#include "Utils.hpp"

/*
  Read "matrixCount" matrices of type TT from DDR by bursts of numElemPerBank 
  elements, and write the matrices to the "matrixPipe" pipe column by column.
  This implementation is used for matrices that have a number of rows that is a 
  multiple of the number of elements per DDR burst read (numElemPerBank).
  Another version of this function is written below and will be selected at 
  compile time if the row count is not a multiple numElemPerBank.
*/
template <typename kernelName,    // Name to use for the Kernel
          typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int numElemPerBank,     // Number of TT elements per DDR burst access
          int matrixCount,        // Number of matrices to read
                                  // from the buffer sequentially
          typename matrixPipe     // Output matrix pipe, send a full column
                                  // with each write
          >
sycl::event MatrixReadFromDDRToPipeByColumns( 
            sycl::queue& q,                     // Device queue
            sycl::buffer<TT, 1> * MatrixBuffer, // Input matrix buffer
            typename std::enable_if<(rows % numElemPerBank) == 0>::type* = 0) {

  // Number of DDR burst reads of numElemPerBank required to read a full column 
  constexpr int kLoopIterPerColumn = rows/numElemPerBank; 
  // Number of DDR burst reads of numElemPerBank to read all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the input matrices
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::read_only);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // "Shift register" that will contain a full column after 
      // kLoopIterPerColumn iterations.
      // Each DDR burst read will write to banks[kLoopIterPerColumn-1] and
      // at each loop iteration each banks[x] will be assigned banks[x+1]
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
          pipeTable<rows, TT> readColumn;

          // Copy the banks data to the correct datatype for the pipe write
          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            #pragma unroll
            for(int k=0; k<numElemPerBank; k++){
              readColumn.elem[i*numElemPerBank+k] = banks[i][k];
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
  Read "matrixCount" matrices of type TT from DDR by bursts of numElemPerBank 
  elements, and write the matrices to the "matrixPipe" pipe column by column.
  This implementation is used for matrices that have a number of rows that is 
  not a multiple of the number of elements per DDR burst read (numElemPerBank).
  Another version of this function is written above and will be selected at 
  compile time if the row count is a multiple numElemPerBank.
*/
template <typename kernelName,    // Name to use for the Kernel
          typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int numElemPerBank,     // Number of TT elements per DDR burst access
          int matrixCount,        // Number of matrices to read
                                  // from the buffer sequentially
          typename matrixPipe     // Output matrix pipe, send a full column
                                  // with each write
          >
sycl::event MatrixReadFromDDRToPipeByColumns( 
            sycl::queue& q,                     // Device queue
            sycl::buffer<TT, 1> * MatrixBuffer, // Input matrix buffer
            typename std::enable_if<(rows % numElemPerBank) != 0>::type* = 0) {

  // Number of DDR burst reads of numElemPerBank required to read a full column 
  constexpr int kLoopIterPerColumn = rows/numElemPerBank + 1; 
  // Number of DDR burst reads of numElemPerBank to read all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the input matrices
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::read_only);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // Keep track of the current element index in the read buffer
      int loadIndex = 0;

      // "Shift register" that will contain a full column after 
      // kLoopIterPerColumn iterations.
      // Each DDR burst read will write to banks[kLoopIterPerColumn-1] and
      // and each loop iteration each banks[x] will be assigned banks[x+1]
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
          pipeTable<rows, TT> readColumn;

          // Copy the banks data to the correct datatype for the pipe write
          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            #pragma unroll
            for(int k=0; k<numElemPerBank; k++){
              if(i*numElemPerBank+k < rows){
                readColumn.elem[i*numElemPerBank+k] = banks[i][k];
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

/*
  Read "matrixCount" matrices of type TT from a pipe, column by column and 
  write them to DDR by bursts of numElemPerBank elements.
  This implementation is used for matrices that have a number of rows that is 
  a multiple of the number of elements per DDR burst write (numElemPerBank).
  Another version of this function is written below and will be selected at 
  compile time if the row count is not a multiple numElemPerBank.
*/
template <typename kernelName,    // Name to use for the Kernel
          typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int numElemPerBank,     // Number of TT elements per DDR burst access
          int matrixCount,        // Number of matrices to write to the 
                                  // buffer sequentially
          typename matrixPipe     // Input matrix, receive a full column
                                  // with each read
          >
sycl::event MatrixReadPipeByColumnsToDDR( 
            sycl::queue& q,                      // Device queue
            sycl::buffer<TT, 1> * MatrixBuffer,  // Output matrix buffer
            typename std::enable_if<(rows % numElemPerBank) == 0>::type* = 0) {

  // Number of DDR burst of numElemPerBank required to write a full column 
  constexpr int kLoopIterPerColumn = rows / numElemPerBank; 
  // Number of DDR burst of numElemPerBank to write all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the output matrices
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // "Shift register" that will contain a full column read from the pipe.
      // Each DDR burst write will then read banks[0]
      // At each loop iteration each banks[x] will be assigned banks[x+1]
      // This ensures that the fanout is kept to a minimum
      TT banks[kLoopIterPerColumn][numElemPerBank];
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        // Read a new column from the pipe every kLoopIterPerColumn iterations
        if((li % kLoopIterPerColumn) == 0){
          // Read the column from the pipe
          pipeTable<rows, TT> pipeRead = matrixPipe::read();

          // Place the column into the banks "shift register"
          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            UnrolledLoop<numElemPerBank>([&](auto k) {
              banks[i][k] = pipeRead.elem[i*numElemPerBank+k];
            });
          }
        }

        // Write the banks[0] to DDR
        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          matrixAccessor[(int)(li*numElemPerBank + k)] = banks[0][k];
        }

        // Perform the register shifting of the banks
        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }
      } // end of li
    }); // end of h
  }); // end of q submit

  return e;
}

/*
  Read "matrixCount" matrices of type TT from a pipe, column by column and 
  write them to DDR by bursts of numElemPerBank elements.
  This implementation is used for matrices that have a number of rows that is 
  not a multiple of the number of elements per DDR burst read (numElemPerBank).
  Another version of this function is written above and will be selected at 
  compile time if the row count is a multiple numElemPerBank.
*/
template <typename kernelName,    // Name to use for the Kernel
          typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int numElemPerBank,     // Number of TT elements per DDR burst access
          int matrixCount,        // Number of matrices to write to the 
                                  // buffer sequentially
          typename matrixPipe     // Input matrix, receive a full column
                                  // with each read
          >
sycl::event MatrixReadPipeByColumnsToDDR( 
            sycl::queue& q,                      // Device queue
            sycl::buffer<TT, 1> * MatrixBuffer,  // Output matrix buffer
            typename std::enable_if<(rows % numElemPerBank) != 0>::type* = 0) {

  // Number of DDR burst of numElemPerBank required to write a full column 
  constexpr int kLoopIterPerColumn = rows / numElemPerBank + 1; 
  // Number of DDR burst of numElemPerBank to write all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns * matrixCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the output matrices
    sycl::accessor matrixAccessor(*MatrixBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // "Shift register" that will contain a full column read from the pipe.
      // Each DDR burst write will then read banks[0]
      // At each loop iteration each banks[x] will be assigned banks[x+1]
      // This ensures that the fanout is kept to a minimum
      TT banks[kLoopIterPerColumn][numElemPerBank];

      // Keep track of the current element index in the write buffer
      int writeIdx = 0;
      
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        // Read a new column from the pipe every kLoopIterPerColumn iterations
        if((li % kLoopIterPerColumn) == 0){
          // Read the column from the pipe
          pipeTable<rows, TT> pipeRead = matrixPipe::read();

          // Place the column into the banks "shift register"
          #pragma unroll
          for(int i=0; i<kLoopIterPerColumn; i++){
            UnrolledLoop<numElemPerBank>([&](auto k) {
              banks[i][k] = pipeRead.elem[i*numElemPerBank+k];
            });
          }
        }

        // Check if we are writing the last DDR burst of the current column 
        bool lastBurstOfCol = li % kLoopIterPerColumn == kLoopIterPerColumn - 1;

        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          // Check if the current write index is beyond the end of the current
          // matrix column
          bool outOfBounds =  lastBurstOfCol && 
                              (k > ((rows-1) % numElemPerBank));

          // Only perform the DDR writes that are relevant (and don't access a
          // memory address that may be beyond the buffer last address)
          if(!outOfBounds){
            matrixAccessor[writeIdx + k] = banks[0][k];
          }
        }

        // Perform the register shifting of the banks
        #pragma unroll
        for(int i=0; i<kLoopIterPerColumn; i++){
          UnrolledLoop<numElemPerBank>([&](auto k) {
            banks[i][k] = banks[i+1][k];
          });
        }

        // Update the current element index in the write buffer according
        // to the write size of the current iteration
        writeIdx += lastBurstOfCol ?  rows % numElemPerBank : 
                                              numElemPerBank;
      } // end of li
    }); // end of h
  }); // end of q submit

  return e;
}

/*
  Read "vectorCount" vectors of type TT from a pipe, element by element and 
  write them to DDR by bursts of numElemPerBank elements.
  This implementation is used for vectors that have a size that is a multiple 
  of the number of elements per DDR burst write (numElemPerBank).
  Another version of this function is written below and will be selected 
  automatically at compile time if the size is not a multiple of numElemPerBank.
*/
template <typename kernelName,    // Name to use for the Kernel
          typename TT,            // Datatype of the elements of the matrix
          int size,               // Number of elements in the vector
          int numElemPerBank,     // Number of TT elements per DDR burst access
          int vectorCount,        // Number of vectors to read from the buffer 
                                  // sequentially
          typename vectorPipe     // Input vector pipe, receive an element
                                  // with each read
          >
sycl::event VectorReadPipeByElementsToDDR( 
            sycl::queue& q,                      // Device queue
            sycl::buffer<TT, 1> * VectorBuffer,  // Output vector buffer
            typename std::enable_if<(size % numElemPerBank) == 0>::type* = 0) {

  // Number of DDR burst of numElemPerBank required to write all the vectors 
  constexpr int kLoopIter = (size / numElemPerBank) * vectorCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the output vectors
    sycl::accessor vectorAccessor(*VectorBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        // Read numElemPerBank elements from the input pipe
        TT bank[numElemPerBank];
        for(int k = 0; k<numElemPerBank; k++){
          bank[k] = vectorPipe::read();
        }

        // Write a burst of numElemPerBank elements to DDR
        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          vectorAccessor[(int)(li*numElemPerBank + k)] = bank[k];
        }              
      } // end of li
    }); // end of h
  }); // end of q submit

  return e;
}

/*
  Read "vectorCount" vectors of type TT from a pipe, element by element and 
  write them to DDR by bursts of numElemPerBank elements.
  This implementation is used for vectors that have a size that is a not a 
  multiple of the number of elements per DDR burst write (numElemPerBank).
  Another version of this function is written above and will be selected 
  automatically at compile time if the size is a multiple of numElemPerBank.
*/
template <typename kernelName,    // Name to use for the Kernel
          typename TT,            // Datatype of the elements of the matrix
          int size,               // Number of elements in the vector
          int numElemPerBank,     // Number of TT elements per DDR burst access
          int vectorCount,        // Number of vectors to read from the buffer 
                                  // sequentially
          typename vectorPipe     // Input vector pipe, receive an element
                                  // with each read
          >
sycl::event VectorReadPipeByElementsToDDR( 
            sycl::queue& q,                      // Device queue
            sycl::buffer<TT, 1> * VectorBuffer,  // Output vector buffer
            typename std::enable_if<(size % numElemPerBank) != 0>::type* = 0) {

  // Number of DDR burst of numElemPerBank required to write all the vectors 
  constexpr int kLoopIter = (size / numElemPerBank + 1) * vectorCount;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize = BitsForMaxValue<kLoopIter + 1>();
  
  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffer containing the output vectors
    sycl::accessor vectorAccessor(*VectorBuffer, h, sycl::write_only, 
                                                                sycl::no_init);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // Keep track of the current element index in the current vector
      int vectorIdx = 0;
      // Keep track of the current vector index
      int vectorCountIdx = 0;

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      [[intel::ivdep]]                    // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        // Read up to numElemPerBank elements from the input pipe by checking
        // that the read index is not beyond the vector size
        TT bank[numElemPerBank];
        for(int k = 0; k<numElemPerBank; k++){
          if(vectorIdx + k < size){
            bank[k] = vectorPipe::read();
          }
        }

        // Write a burst of numElemPerBank elements to DDR
        #pragma unroll 
        for(int k = 0; k<numElemPerBank; k++){
          if ((vectorIdx + k) < size){
            vectorAccessor[vectorCountIdx * size + vectorIdx + k] = bank[k];
          }
        }             

        // Update the indexes
        int vectorIdxPlusNumElemPerBank = vectorIdx + numElemPerBank;
        if(vectorIdxPlusNumElemPerBank > size){
          vectorIdx = 0;
          vectorCountIdx += 1;
        }
        else{
          vectorIdx = vectorIdxPlusNumElemPerBank;
        }

      } // end of li
    }); // end of h
  }); // end of q submit

  return e;
}