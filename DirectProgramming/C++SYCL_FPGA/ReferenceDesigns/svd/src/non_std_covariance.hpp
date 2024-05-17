#ifndef __NON_STD_COVARIANCE_MATRIX_HPP__
#define __NON_STD_COVARIANCE_MATRIX_HPP__

namespace fpga_linalg {
// This functor computes the columns x columns covariance matrix of a rows x
// columns input matrix A

// The deign compute t_matrix which is transpose(A) * A 

template <typename T,          // The datatype for the computation
          unsigned rows,       // Number of Rows in the A matrices
          unsigned columns,    // Number of columns in the A matrices
          unsigned pipe_size,  // Number of elements read/write per pipe
                               // operation, the matrix is received through the
                               // pipe by blocks of size columns*columns.
          typename InputPipe,  // A matrix input pipe, receive pipe_size
                               // elements from the pipe with each read
          typename OutputPipe  // T matrix output pipe, send pipe_size
                               // elements to the pipe with each write
          >
struct StreamingNStdCovarianceMatrix {
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

    // Array to keep the T matrix
    [[intel::max_replicates(1)]]    // NO-FORMAT: Attribute
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    T t_matrix[columns * columns];

    // We keep count of the current block number
    int block = 0;

    while (1) {
      // Read the next matrix block into the a_load local memory

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      [[intel::private_copies(2)]]          // NO-FORMAT: Attribute
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

      // We are going to reuse the same column of the matrix multiple
      // iterations in a row, so we keep it locally
      row_tuple current_base_column;
      row_tuple next_base_column;
      // Compute the block T matrix and the partial means

      // Arrays to keep all the data of the current block being computed
      [[intel::max_replicates(1)]]    // NO-FORMAT: Attribute
      [[intel::private_copies(2)]]  // NO-FORMAT: Attribute
      T t_matrix_compute[columns * columns];

      int row = 0;
      int column = 0;
      for (int it = 0; it < columns * columns; it++) {
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
        // T mean = 0;
        fpga_tools::UnrolledLoop<columns>([&](auto t) {
          dot_product += current_column.template get<t>() *
                         current_base_column.template get<t>();
        });

        // Update the partial result T matrix
        t_matrix_compute[it] = dot_product;

        // Update the current row and column indexes
        if (column == columns - 1) {
          column = 0;
          row++;
        } else {
          column++;
        }
      }

      // For the computation of COV
      [[intel::max_replicates(1)]]    // NO-FORMAT: Attribute
      [[intel::private_copies(2)]]  // NO-FORMAT: Attribute
      T t_matrix_consume[columns][columns];

      // Update the global T matrix with the partial results and copy the result
      // to the t_matrix_consume array for better memory structure in the
      // computation of COV
      for (row = 0; row < columns; row++) {
        fpga_tools::UnrolledLoop<columns>([&](auto column) {
          T t_matrix_to_add = block == 0 ? 0 : t_matrix[row * columns + column];
          T sum = t_matrix_compute[row * columns + column] + t_matrix_to_add;
          t_matrix[row * columns + column] = sum;
          t_matrix_consume[row][column] = sum;
        });
      }


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
                      ? t_matrix_consume[li / kLoopIterationPerRow][t * pipe_size + k]
                      : sycl::ext::intel::fpga_reg(
                            pipe_write.template get<k>());
            }
          });
        });

        if (block == block_count - 1) {
          OutputPipe::write(pipe_write);
        }
      }

      if (block == block_count - 1) {
        block = 0;
      } else {
        block++;
      }
    }  // end of while
  };   // end of operator()
};     // end of struct{}

}  // namespace fpga_linalg

#endif /* __NON_STD_COVARIANCE_MATRIX_HPP__ */