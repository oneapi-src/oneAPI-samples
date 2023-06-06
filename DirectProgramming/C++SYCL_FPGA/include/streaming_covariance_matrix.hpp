#ifndef __STREAMING_COVARIANCE_MATRIX_HPP__
#define __STREAMING_COVARIANCE_MATRIX_HPP__

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

namespace fpga_linalg {
// This functor computes the columns x columns covariance matrix of a rows x
// columns input matrix A

// It uses the following formula:
// COV[i][j] = (T[i][j] - rows*mean[i]*mean[j]) /
//                                (sqrt(T[i][i] - rows*mean[i]*mean[i]) *
//                                   sqrt(T[j][j] - rows*mean[j]*mean[j]))
// Where T is transpose(A)*A and mean[k] is the mean of the column k of the A
// matrix

template <typename T,          // The datatype for the computation
          unsigned rows,       // Number of rows in the A matrices
          unsigned columns,    // Number of columns in the A matrices
          unsigned pipe_size,  // Number of elements read/write per pipe
                               // operation, the matrix is received through the
                               // pipe
                               // by blocks of size columns*columns.
          typename InputPipe,  // A matrix input pipe, receive pipe_size
                               // elements from the pipe with each read
          typename OutputPipe  // Q matrix output pipe, send pipe_size
                               // elements to the pipe with each write
          >
struct StreamingCovarianceMatrix {
  void operator()() const {
    static_assert(rows % columns == 0,
                  "The feature count must be  a multiple of the samples count."
                  "This can be artificially achieved by increasing the number"
                  "of samples with no data.");

    // Type used to store the matrices in the compute loop
    using row_tuple = fpga_tools::NTuple<T, columns>;

    // Number of matrix blocks to read from the pipe
    constexpr int block_count = rows / columns;

    // Break memories up to store 8 float numbers (32 bytes) per bank
    constexpr short kBankwidth = pipe_size * sizeof(T);
    constexpr unsigned short kNumBanks = columns / pipe_size;

    // When specifying numbanks for a memory, it must be a power of 2.
    // Unused banks will be automatically optimized away.
    constexpr short kNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

    // Copy a matrix from the pipe to a local memory
    // Number of pipe reads of pipe_size required to read a full column
    constexpr int kExtraIteration = (columns % pipe_size) != 0 ? 1 : 0;
    constexpr int kLoopIterationPerRow = columns / pipe_size + kExtraIteration;
    // Number of pipe reads of pipe_size to read all the matrices
    constexpr int kLoopIterations = kLoopIterationPerRow * columns;



    // We keep a replicate of the diagonal of T for improved memory access
    // pattern over T
    // [[intel::fpga_register]]       // NO-FORMAT: Attribute
      // [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
    T t_matrix_diagonal_replicate[columns];

    // Array to keep the means of all the A matrix columns
    //   [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
    row_tuple means_tuple;
        // Copy the mean and the T diagonal to structures that are
        // more easily accessible
      // [[intel::private_copies(128)]] // NO-FORMAT: Attribute
        T means[columns];

        // T means_consume[columns];


    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
    T t_matrix[columns * columns];


    int block = 0;

    while (1) {

      // // [[intel::ivdep(t_matrix_compute, columns*columns)]]      // NO-FORMAT: Attribute
      // for (int block = 0; block < block_count; block++) {
        // Read the first matrix block into the a_load local memory

        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
        row_tuple a_load[columns];

        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (int li = 0; li < kLoopIterations; li++) {
          fpga_tools::NTuple<T, pipe_size> pipe_read = InputPipe::read();

          int write_idx = li % kLoopIterationPerRow;
          int a_col_index = li / kLoopIterationPerRow;

          fpga_tools::UnrolledLoop<kLoopIterationPerRow>([&](auto k) {
            fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
              if (write_idx == k) {
                if constexpr (k * pipe_size + t < columns) {
                  a_load[a_col_index].template get<k * pipe_size + t>() =
                      pipe_read.template get<t>();
                }
              }

              // Delay data signals to create a vine-based data distribution
              // to lower signal fanout.
              pipe_read.template get<t>() =
                  sycl::ext::intel::fpga_reg(pipe_read.template get<t>());
            });

            write_idx = sycl::ext::intel::fpga_reg(write_idx);
          });
        }  // for:li

        // PRINTF("Cov submatrix read\n");
        // for (int row = 0; row < columns; row++) {
        //   fpga_tools::UnrolledLoop<columns>(
        //       [&](auto t) { PRINTF("%f ", a_load[row].template get<t>()); });
        //   PRINTF("\n");
        // }

        // We are going to reuse the same column of the matrix multiple
        // iterations in a row, so we keep it locally
        row_tuple current_base_column;
        row_tuple next_base_column;
        // Compute the block T matrix and the partial means


    T t_matrix_diagonal_replicate_partial[columns];
        T means_array[columns];

    // For the computation of COV
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
    T t_matrix_consume[columns][columns];
    // Matrix to hold the partial block results of At*A
    // During the compute of T
    [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
    T t_matrix_compute[columns * columns];
    
        int row=0;
        int column =0;
        // [[intel::ivdep]]      // NO-FORMAT: Attribute
        // [[intel::ivdep(t_matrix_compute)]]      // NO-FORMAT: Attribute
        // [[intel::ivdep(columns)]]      // NO-FORMAT: Attribute
        // [[intel::ivdep(t_matrix_diagonal_replicate)]]      // NO-FORMAT: Attribute
        // [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (int it = 0; it < columns*columns; it++){
        // for (int row = 0; row < columns; row++) {
        //   for (int column = 0; column < columns; column++) {
            // Load the current column of the block
            row_tuple current_column = a_load[column];

            // Keep the current column in the local cache for future reuse
            if (column == 0) {
              if (column == row) {
                current_base_column = current_column;
              } else {
                current_base_column = next_base_column;
              }
            } else if (column == (row + 1)) {
              next_base_column = current_column;
            }

            // Compute the partial T value and the partial mean
            T dot_product = 0;
            T mean = 0;
            fpga_tools::UnrolledLoop<columns>([&](auto t) {
              dot_product += current_column.template get<t>() *
                             current_base_column.template get<t>();
              mean += current_column.template get<t>();
            });


            t_matrix_compute[it] = dot_product;            

            // // Update the partial results matrix
            // T t_result = block == 0
            //                  ? dot_product
            //                  : dot_product + t_matrix_compute[it];
            // t_matrix_compute[it] = t_result;

            // // on the last block, copy the final T matrix to the second
            // // copy of T for the COV computation
            // if (block == (block_count - 1)) {
            //   t_matrix_consume[row][column] = t_result;
            // }

            // Adjust the mean as we only need to compute it once
            if (row != 0) {
              mean = 0;
            }
            mean /= rows;

            // T new_mean = means[column] + mean;
            if (row == 0){
              means_array[column] = mean;
            }

            if (row == column){
              t_matrix_diagonal_replicate_partial[column] = dot_product;
            }

            // if (block == block_count-1){
            //   means_consume[column] = new_mean;
            // }

            // if (row == column){
            //   T to_add = block == 0 ? dot_product : t_matrix_diagonal_replicate[row] + dot_product;
            //   t_matrix_diagonal_replicate[row] = to_add;
            //   if (block == block_count-1){
            //     t_matrix_diagonal_replicate_consume[row] = to_add;
            //   }
            // }

            // Feed the mean and diagonal accumulators
            // fpga_tools::UnrolledLoop<columns>([&](auto t) {
            //   // T mean_to_add = (t == column) ? mean : 0;
            //   // means_tuple.template get<t>() += mean_to_add;

            //   T diagonal_value_to_add =
            //       (row == column) && (row == t) ? dot_product : 0;
            //   t_matrix_diagonal_replicate_tuple.template get<t>() +=
            //       diagonal_value_to_add;
            // });

            // means[column] += mean;
        //   }  // end for:column
        // }    // end for:row

            if (column == columns-1){
              column=0;
              row++;
            }
            else{
              column++;
            }
          }
      // }      // end of for: block

        fpga_tools::UnrolledLoop<columns>([&](auto t) {
          // means[t] = means_tuple.template get<t>();
          T mean_to_add = block == 0 ? 0 : means[t]; 
          means[t] = means_array[t] + mean_to_add;
          T t_matrix_diagonal_replicate_to_add = block == 0 ? 0 : t_matrix_diagonal_replicate[t]; 
          t_matrix_diagonal_replicate[t] = t_matrix_diagonal_replicate_partial[t] + t_matrix_diagonal_replicate_to_add;
        });

        for (row = 0; row< columns; row++){
          fpga_tools::UnrolledLoop<columns>([&](auto column) {
            T t_matrix_to_add = block == 0 ? 0 : t_matrix[row * columns + column]; 
            T sum = t_matrix_compute[row * columns + column] + t_matrix_to_add;
            t_matrix[row * columns + column] = sum;
            t_matrix_consume[row][column] = sum;
          });        
        }

        // fpga_tools::UnrolledLoop<columns>([&](auto t) {
        //   // means[t] = means_tuple.template get<t>();
        //   t_matrix_diagonal_replicate[t] =
        //       t_matrix_diagonal_replicate_tuple.template get<t>();
        // });

        // t_matrix_consume now contains the full matrix product of the transpose
        // of A times A. mean now contains the mean of all the columns of A. We
        // now need to compose all of these results to get the covariance matrix
        [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        [[intel::private_copies(128)]]            // NO-FORMAT: Attribute
        T cov_matrix[columns][columns];

        // for (int row = 0; row < columns; row++) {
        //   for (int column = 0; column < columns; column++) {
        {
          int row = 0;
          int column = 0;
          for (int it = 0; it < columns*columns; it++){
              T numerator = t_matrix_consume[row][column] -
                            (rows * means[row] * means[column]);

              T denominator = std::sqrt((t_matrix_diagonal_replicate[row] -
                                         (rows * means[row] * means[row])) *
                                        (t_matrix_diagonal_replicate[column] -
                                         (rows * means[column] * means[column])));
              cov_matrix[row][column] = numerator / denominator;

              if (column == columns-1){
                column = 0;
                row++;
              }
              else{
                column++;
              }
          }
        }
        //   }  // end for:column
        // }    // end for:row

          // if (block == block_count-1){
          //   PRINTF("COV MATRIX\n");
          //   for (int row = 0; row < columns; row++) {
          //     for (int column = 0; column < columns; column++) {
          //       PRINTF("%f ", cov_matrix[row][column]);
          //     }  // end for:column
          //       PRINTF("\n");
          //   }    // end for:row
          // }

        // Write the standardized covariance matrix to the output pipe
        // [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (int li = 0; li < kLoopIterations; li++) {
          int column_iter = li % kLoopIterationPerRow;
          bool get[kLoopIterationPerRow];
          fpga_tools::UnrolledLoop<kLoopIterationPerRow>([&](auto k) {
            get[k] = column_iter == k;
            column_iter = sycl::ext::intel::fpga_reg(column_iter);
          });

          fpga_tools::NTuple<T, pipe_size> pipe_write;
          fpga_tools::UnrolledLoop<kLoopIterationPerRow>([&](auto t) {
            fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
              if constexpr (t * pipe_size + k < columns) {
                pipe_write.template get<k>() =
                    get[t]
                        // ? t_matrix_compute[(li / kLoopIterationPerRow) * columns + t * pipe_size + k]
                        ? cov_matrix[li / kLoopIterationPerRow][t * pipe_size + k]
                        : sycl::ext::intel::fpga_reg(
                              pipe_write.template get<k>());
              }
            });
          });

          if (block == block_count-1){
            OutputPipe::write(pipe_write);
          }
        }
      

      if (block == block_count-1){
        block = 0;
      }
      else{
        block++;
      }
    }  // end of while
  };  // end of operator()
};    // end of struct{}

}  // namespace fpga_linalg

#endif /* __STREAMING_COVARIANCE_MATRIX_HPP__ */
