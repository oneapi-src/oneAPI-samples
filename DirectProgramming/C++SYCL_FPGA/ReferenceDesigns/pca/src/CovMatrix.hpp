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
  	constexpr int kLoopItr = rows*kRowBlocks;

  	constexpr int kColBlockBitSize = fpga_tools::BitsForMaxValue<kColBlocks + 1>();
  	constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopItr + 1>();

    constexpr int maxRow = (rows > pipe_size) ? rows : pipe_size;
  	constexpr int kRowBitSize = fpga_tools::BitsForMaxValue<maxRow + 1>();



  	while(1){

  		// storing in a internal matrix 

    		// NO-FORMAT: Attribute
  		row_tuple MatrixC[rows], MatrixCW[rows];
  		TT Avg[rows], AvgW[rows];
    	pipe_tuple pipe_read;
      TT digValM[rows], avgVal;



  		for(ac_int<kColBlockBitSize, false> blk = 0; blk < kColBlocks; blk++){
  			// loading data onchip memory 
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


  			// [[intel::loop_coalesce(2)]]
        // [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        T colSum;
        row_tuple row1, row2, row_temp, rowSumL, rowSumW;
  			for(ac_int<kRowBitSize, false> i_ll = 0; i_ll < rows; i_ll++){
  				for(ac_int<kRowBitSize, false> j_ll = 0; j_ll < rows; j_ll++){
  					T sum = 0;
  					
  					if(j_ll == 0){
  						rowSumL = MatrixC[i_ll];
  					}

            if(j_ll == 0 && blk == 0){
              avgVal = 0;
            } else if(j_ll == 0){
              avgVal = Avg[i_ll];
            }
            

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


            TT newSum;
  					fpga_tools::UnrolledLoop<rows>([&](auto t) {
  						if(j_ll == t && blk == 0){
  							rowSumW.template get<t> () = sum;
                newSum = sum;
  						} else if(j_ll == t){
                newSum = rowSumL.template get<t> () + sum;
  							rowSumW.template get<t> () = rowSumL.template get<t> () + sum;
  						}
  					});


            if(i_ll == j_ll){
              digValM[i_ll] = newSum;
            }

            if(j_ll == 0){
              colSum = 0;
            }

            T Elem;
  					fpga_tools::UnrolledLoop<rows>([&](auto t) {
              if(t == j_ll){
                Elem = row1.template get<t>();
              }
  					});


            colSum += Elem/columns ;
            T tempVal =  avgVal + colSum; //colSum;

					  if(j_ll == rows - 1){
  						MatrixC[i_ll] = rowSumW;
  						Avg[i_ll] = tempVal;
  					}

  					if(blk == kColBlocks-1 && j_ll == rows - 1){
  						MatrixCW[i_ll] = rowSumW;
  						AvgW[i_ll] = tempVal;
  					}

  				}
  			}
  		}

  		// row_tuple row_write;
  		pipe_tuple pipe_write;
  		TT avg1, avg2, avg_temp;
      TT digVal1, digVal2, dig_temp; 
  		for(ac_int<kRowBitSize, false> i_ll = 0; i_ll < rows; i_ll++){
  			for(ac_int<kRowBitSize, false> j_ll = 0; j_ll < rows; j_ll++){
  				T loadVal;
  				row_tuple loadRow = MatrixCW[i_ll];
  				fpga_tools::UnrolledLoop<rows>([&](auto t) {
  					if(j_ll == t){
  						loadVal = loadRow.template get<t>();
  					}
  				});

          //---------------------------
  				avg2 = AvgW[j_ll];
          digVal2 = digValM[j_ll];
  				if(j_ll == i_ll + 1){
  					avg_temp = avg2;
            dig_temp = digVal2;
  				}

  				if(i_ll == 0 && j_ll == 0){
  					avg1 = avg2;
            digVal1 = digVal2;
  				} else if(j_ll == 0){
  					avg1 = avg_temp;
            digVal1 = dig_temp;
  				}
          //---------------------------
          
          T cov_i_i = digVal1 - columns * avg1 * avg1;
          T cov_j_j = digVal2 - columns * avg2 * avg2;


  				T cov_i_j_tmp = loadVal - columns * avg1 * avg2;
  				// T cov_i_j = (1.0f/(columns-1)) * cov_i_j_tmp;

          // PRINTF("%f ", cov_i_j);

          T cov_i_j = cov_i_j_tmp/sqrt(cov_i_i*cov_j_j);


  				fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            // pipe_write.template get<t> () = 5;
            // PRINTF("j_ll=%d pipe_size=%d val:%d\n", j_ll, pipe_size, j_ll % pipe_size);
  					if(t == j_ll % pipe_size){

  						pipe_write.template get<t> () = cov_i_j;
  					}
  				});

  				if(j_ll % pipe_size == pipe_size -1 || j_ll == rows-1){
  					AOut::write(pipe_write);
  				}

  			}
        // PRINTF("\n");
  		}



  	}

 	};
};

}


#endif 