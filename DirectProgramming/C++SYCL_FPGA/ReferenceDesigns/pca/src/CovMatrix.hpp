#ifndef __STREAMING_EIGEN_HPP__
#define __STREAMING_EIGEN_HPP__

template <typename T,        // The datatype for the computation
          bool is_complex,   // True if T is ac_complex<X>
          int rows,          // Number of rows in the A matrices
          int columns,       // Number of columns in the A matrices

          int blockSize,	 // number of parallel mult and add 
          int pipe_size,     // Number of elements read/write per pipe
                             // operation
          typename AIn,      // A matrix input pipe, receive pipe_size
                             // elements from the pipe with each read
          typename AOut     // Q matrix output pipe, send pipe_size
                             // elements to the pipe with each write
          >



// input matrix will be A with order NxP (rows x columns)
// output will be A x transpose(A) 
// sample size N would be larger 
// this makes doing a full dot product inefficient 



struct StreamingMM{
  void operator()() const {
  	
  	using TT = std::conditional_t<is_complex, ac_complex<T>, T>;
  	using block_tuple = fpga_tools::NTuple<TT, blockSize>;

  	constexpr int kIBitSizeRows = fpga_tools::BitsForMaxValue<rows + 1>() + 1;
  	constexpr int kIBitSizeColumns = fpga_tools::BitsForMaxValue<columns + 1>() + 1;

  	constexpr int kRamSize = (rows/blockSize + 1) x columns;


    // NO-FORMAT: Attribute
  	block_tuple MatrixA[kRamSize];
  	while(1){


  	}

  }
}

#endif 