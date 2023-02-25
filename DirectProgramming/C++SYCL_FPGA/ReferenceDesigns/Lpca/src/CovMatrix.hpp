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
  	using row_tuple = fpga_tools::NTuple<TT, rows>;
  	using pipe_tuple = fpga_tools::NTuple<TT, pipe_size>;

  	constexpr int kColBlocks = (columns+rows-1)/rows;
  	constexpr int kRowBlocks = (rows+pipe_size-1)/pipe_size;



  	while(1){

  		// storing in a internal matrix 

  		// NO-FORMAT: Attribute
		row_tuple MatrixA[rows];
		row_tuple MatrixC[rows];
		TT Avg[rows];
  		pipe_tuple pipe_read;

  		for(int blk = 0; blk < kColBlocks; blk++){
  			// loading data onchip memory 
  			for(int i_ll = 0; i_ll < rows; i_ll++){
  				for(int j_ll = 0; j_ll < kRowBlocks; j_ll++){
  					pipe_read = AIn::read();
  					fpga_tools::UnrolledLoop<kRowBlocks>([&](auto k) {
          				fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
          					if(k == j_ll){
          						if constexpr (k*pipe_size+t < rows){
          							MatrixA[i_ll].template get<k*pipe_size+t> () = pipe_read.template get<t>();
          						}
          					}
          				});
          			});

  				}
  			}

  			row_tuple row1, row2, row_temp;
  			for(int i_ll = 0; i_ll < rows; i_ll++){
  				for(int j_ll = 0; j_ll < rows; j_ll++){
  					T sum = 0;
  					
  					row2 = MatrixA[j_ll];
  					if(j_ll == i_ll + 1){
  						row_temp = row2;
  					}

  					if(i_ll == 0 && j_ll == 0){
  						row1 = row2;
  					} else if(j_ll == 0){
  						row1 = row_temp;
  					}

  					fpga_tools::UnrolledLoop<rows>([&](auto t) {
  						sum += row1.template get<t>() * row2.template get<t>();
  					});

  					fpga_tools::UnrolledLoop<rows>([&](auto t) {
  						if(j_ll == t && blk == 0){
  							MatrixC[i_ll].template get<t> () = sum;
  						} else if(j_ll == t){
  							MatrixC[i_ll].template get<t> () += sum;
  						}
  					});

  					T colSum = 0;
  					fpga_tools::UnrolledLoop<rows>([&](auto t) {
  						colSum += row1.template get<t>() / columns;
  					});

					if(j_ll == 0  && blk == 0){
						Avg[i_ll] = colSum;
					} else if(j_ll == 0){
						Avg[i_ll] += colSum;
					}

  				}
  			}
  		}

  		row_tuple row_write;
  		pipe_tuple pipe_write;
  		for(int i_ll = 0; i_ll < rows; i_ll++){
  			for(int j_ll = 0; j_ll < rows; j_ll++){
  				T loadVal;
  				fpga_tools::UnrolledLoop<rows>([&](auto t) {
  					if(j_ll == t){
  						loadVal = MatrixC[i_ll].template get<t>();
  					}
  				});

  				T cov_i_j_tmp = loadVal - columns * Avg[i_ll] * Avg[j_ll];
  				T cov_i_j = (1.0f/(columns-1)) * cov_i_j_tmp;

  				fpga_tools::UnrolledLoop<rows>([&](auto t) {
  					if(j_ll == t){
  						row_write.template get<t>() = cov_i_j;
  					}
  				});

  				fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
  					if(t == j_ll % pipe_size){
  						pipe_write.template get<t> () = cov_i_j;
  					}
  				});

  				if(j_ll % pipe_size == pipe_size -1 || j_ll == rows-1){
  					AOut::write(pipe_write);
  				}

  			}
  		}



  	}

 	};
};

}


#endif 