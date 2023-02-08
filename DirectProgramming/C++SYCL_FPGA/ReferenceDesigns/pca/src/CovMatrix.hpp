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

/*
Matrix - each row contains samples of a feature 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
*/

struct StreamingMM{
    void operator()() const {
  	
  	using TT = std::conditional_t<is_complex, ac_complex<T>, T>;
  	using block_tuple = fpga_tools::NTuple<TT, blockSize>;
  	using pipe_tuple = fpga_tools::NTuple<TT, pipe_size>;


  	constexpr int kRowBlocks = (columns + blockSize-1)/blockSize;
  	constexpr int kRowpipeBlk = (columns + pipe_size-1)/pipe_size ;


  	constexpr int kIBitSizeRows = fpga_tools::BitsForMaxValue<rows + 1>() + 1;
  	constexpr int kIBitSizeColumnBlks = fpga_tools::BitsForMaxValue<kRowBlocks + 1>() + 1;

  	constexpr int kRamSize = kRowBlocks x rows;
  	constexpr int kLoopIter = kRowpipeBlk x rows;

  	constexpr int kBlkFold  = blockSize/pipe_size;

  	constexpr int kColpipeBlk = (rows + pipe_size-1)/pipe_size;
  	constexpr int kOutMatrixSize = kColpipeBlk*columns;

    // NO-FORMAT: Attribute
  	block_tuple MatrixA[kRamSize];
  	block_tuple blkRow;
  	while(1){

  		// storing in a internal matrix 
	  	[[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
		for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
		    fpga_tools::NTuple<TT, pipe_size> pipe_read = AIn::read();
		    int MatrixA_addr = li / kBlkFold;
		    int wordId = li % kBlkFold;

		    fpga_tools::UnrolledLoop<kBlkFold>([&](auto k) {
		      fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
		      	blkRow.template get<k*pipe_size+t> = pipe_read.template get<t>(); 
		      });
		    });
		    MatrixA[MatrixA_addr] = blkRow;
		}

		// computing the eigen vectors
		for (ac_int<kIBitSizeColumns, false> li = 0; li < columns; li++) {
			pipe_tuple pipe_write;
			for (ac_int<kIBitSizeColumns, false> lj = 0; lj < columns; lj++) {
				T sum = 0;
				// need get dot product of li row and lj row
				for (ac_int<kIBitSizeColumnBlks, false> lk = 0; lk < kRowBlocks; lk++) {
					int add1 = li*kRowBlocks + lk;
					int add2 = lj*kRowBlocks + lk;

					fpga_tools::UnrolledLoop<blockSize>([&](auto t) {
						sum += MatrixA[add1].template get<t> * MatrixA[add2].template get<t>

					});
				}
				fpga_tools::UnrolledLoop<kBlkFold>([&](auto t) {
					if(t == lj % pipe_size){
						pipe_write.template get<t> = sum;
					}
				});

				if(lj % pipe_size = pipe_size -1 || lj == columns-1){
					AOut::write(pipe_write);
				}

			}
		}


  	}

 }
#endif 