#ifndef __STREAMING_CovMM_HPP__
#define __STREAMING_CovMM_HPP__

namespace fpga_linalg {

template <typename T,        // The datatype for the computation
          bool is_complex,   // True if T is ac_complex<X>
          unsigned rows,          // Number of rows in the A matrices
          unsigned columns,       // Number of columns in the A matrices

          unsigned blockSize,	 // number of parallel mult and add 
          unsigned pipe_size,     // Number of elements read/write per pipe
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


  	constexpr int kRamSize = kRowBlocks * rows;
  	// constexpr int kLoopIter = kRowpipeBlk * rows;

  	constexpr int kIBitSizeRows = fpga_tools::BitsForMaxValue<rows + 1>() + 1;
  	// constexpr int kIBitSizeColumns = fpga_tools::BitsForMaxValue<columns + 1>() + 1;
  	constexpr int kIBitSizeColumnspipes = fpga_tools::BitsForMaxValue<kRowpipeBlk + 1>() + 1;
  	constexpr int kIBitSizeColumnBlks = fpga_tools::BitsForMaxValue<kRowBlocks + 1>() + 1;
  	// constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>() + 1;

  	constexpr int kBlkFold  = blockSize/pipe_size;

  	// constexpr int kColpipeBlk = (rows + pipe_size-1)/pipe_size;
  	// constexpr int kOutMatrixSize = kColpipeBlk*columns;


  	while(1){

  		// storing in a internal matrix 

  		// NO-FORMAT: Attribute
		block_tuple MatrixA[kRamSize];
		block_tuple blkRow[kRowBlocks];
		block_tuple blk_W, blk_R;


  		T sum, mu, mu_old; 
  		pipe_tuple pipe_read;

  		// PRINTF("Normalised matrix is: \n");
	  	// [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
	  	[[intel::loop_coalesce(2)]]
	  	for (ac_int<kIBitSizeRows, false> li = 0; li < rows+1; li++) {
			for (ac_int<kIBitSizeColumnspipes, false> lj = 0; lj < kRowpipeBlk; lj++) {
				if(lj == 0){
					sum  = 0;
				}
				if(li < rows){
		    		pipe_read = AIn::read();
		    	}
		    	int li_1 = (li-1);
		    	int MatrixA_addr = li_1*kRowBlocks + lj/kBlkFold;
		    	int wordId = lj % kBlkFold;

		    	T localSum = 0;
		    	fpga_tools::UnrolledLoop<kBlkFold>([&](auto k) {
		      		fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
		      			if(wordId == k && lj*pipe_size+t < columns){
		      				blk_W.template get<k*pipe_size+t>() = pipe_read.template get<t>(); 
		      				// PRINTF("%f ", pipe_read.template get<t>());
		      			}
		      		});
		    	});


		    	fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
	      			if(lj*pipe_size+t < columns){
	      				localSum +=  pipe_read.template get<t>(); 
	      				// PRINTF("%f ", pipe_read.template get<t>());
	      			}
	      		});

		    	sum += localSum;

		    	if(lj == kRowpipeBlk -1){
		    		mu = sum * 1.0f/(columns);
		    		// PRINTF("%f ", mu);

		    	}

		    	blk_R = blkRow[lj/kBlkFold];

		    	fpga_tools::UnrolledLoop<blockSize>([&](auto k) {
		    		blk_R.template get<k>() -= mu_old;
		    		// if(li > 0) PRINTF("%f ", blk_R.template get<k>());
		    	});

		    	// PRINTF("\n");
		    	if(li > 0){
		    		MatrixA[MatrixA_addr] = blk_R;
		    	}
		    	blkRow[lj] = blk_W;
		    	blkRow[lj/kBlkFold] = blk_W;

		    	if(lj == kRowpipeBlk -1){
		    		mu_old = mu;
		    	}
			}
			// PRINTF("\n");
		}
		// PRINTF("\n");

		// computing the eigen vectors
		// PRINTF("Covariance Matrix is: \n");
		pipe_tuple pipe_write;
		for (ac_int<kIBitSizeRows, false> li = 0; li < rows; li++) {
			for (ac_int<kIBitSizeRows, false> lj = 0; lj < rows; lj++) {
				T Dot = 0;
				// need get dot product of li row and lj row
				for (ac_int<kIBitSizeColumnBlks, false> lk = 0; lk < kRowBlocks; lk++) {
					int add1 = li*kRowBlocks + lk;
					int add2 = lj*kRowBlocks + lk;

					fpga_tools::UnrolledLoop<blockSize>([&](auto t) {
						if(lk*blockSize+t < columns){
							Dot += MatrixA[add1].template get<t>() * MatrixA[add2].template get<t>();
						}

					});
				}
				// PRINTF("%f ", (1.0f/(columns-1)) * Dot);

				fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
					if(t == lj % pipe_size){
						pipe_write.template get<t>() = (1.0f/(columns-1)) * Dot;
					}
				});
				
				// sending column by column
				if(lj % pipe_size == pipe_size -1 || lj == rows-1){
					AOut::write(pipe_write);
				}

			}
			// PRINTF("\n");
		}
		// PRINTF("\n");


  	}

 	};
};

}


#endif 