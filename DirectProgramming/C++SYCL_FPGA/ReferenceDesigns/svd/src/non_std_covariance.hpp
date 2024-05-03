#ifndef __NON_STD_COVARIANCE_MATRIX_HPP__
#define __NON_STD_COVARIANCE_MATRIX_HPP__

namespace fpga_linalg {
// This functor computes the columns x columns covariance matrix of a rows x
// columns input matrix A

// The deign compute t_matrix which is transpose(A) * A 

template <typename T,          // The datatype for the computation
          unsigned kRows,       // Number of Rows in the A matrices
          unsigned kColumns,    // Number of columns in the A matrices
          unsigned kPipeSize,  // Number of elements read/write per pipe
                               // operation, the matrix is received through the
                               // pipe by blocks of size columns*columns.
          typename InputPipe,  // A matrix input pipe, receive kPipeSize
                               // elements from the pipe with each read
          typename OutputPipe  // T matrix output pipe, send kPipeSize
                               // elements to the pipe with each write
          >
struct StreamingNStdCovarianceMatrix {
  void operator()() const {
    static_assert(kRows % kColumns == 0,
                  "The feature count must be  a multiple of the samples count."
                  "This can be artificially achieved by increasing the number"
                  "of samples with no data.");

    // Type used to store the matrices in the compute loop
    using row_tuple = fpga_tools::NTuple<T, kColumns>;

    // Number of matrix blocks to read from the pipe
    constexpr int block_count = kRows / kColumns;

    // Break memories up to store 8 float numbers (32 bytes) per bank
    constexpr short kBankwidth = kPipeSize * sizeof(T);
    constexpr unsigned short kNumBanks = kColumns / kPipeSize;

    // When specifying numbanks for a memory, it must be a power of 2.
    // Unused banks will be automatically optimized away.
    constexpr short kNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

    // Copy a matrix from the pipe to a local memory
    // Number of pipe reads of kPipeSize required to read a full column
    constexpr int kExtraIteration = (kColumns % kPipeSize) != 0 ? 1 : 0;
    constexpr int kLoopIterationPerRow = kColumns / kPipeSize + kExtraIteration;
    // Number of pipe reads of kPipeSize to read all the matrices
    constexpr int kLoopIterations = kLoopIterationPerRow * kColumns;

    // Array to keep the T matrix
    [[intel::max_replicates(1)]]    // NO-FORMAT: Attribute
    [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
    T t_matrix[kColumns * kColumns];

    // We keep count of the current block number
    int block = 0;

    while (1) {
      // Read the next matrix block into the a_load local memory

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      [[intel::private_copies(2)]]          // NO-FORMAT: Attribute
      row_tuple a_load[kColumns];

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int li = 0; li < kLoopIterations; li++) {
        fpga_tools::NTuple<T, kPipeSize> pipe_read = InputPipe::read();

        int write_idx = li % kLoopIterationPerRow;
        int a_col_index = li / kLoopIterationPerRow;

        fpga_tools::UnrolledLoop<kLoopIterationPerRow>([&](auto k) {
          fpga_tools::UnrolledLoop<kPipeSize>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * kPipeSize + t < kColumns) {
                a_load[a_col_index].template get<k * kPipeSize + t>() =
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
      T t_matrix_compute[kColumns * kColumns];

      int row = 0;
      int column = 0;
      for (int it = 0; it < kColumns * kColumns; it++) {
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
        fpga_tools::UnrolledLoop<kColumns>([&](auto t) {
          dot_product += current_column.template get<t>() *
                         current_base_column.template get<t>();
        });

        // Update the partial result T matrix
        t_matrix_compute[it] = dot_product;

        // Update the current row and column indexes
        if (column == kColumns - 1) {
          column = 0;
          row++;
        } else {
          column++;
        }
      }

      // For the computation of COV
      [[intel::max_replicates(1)]]    // NO-FORMAT: Attribute
      [[intel::private_copies(2)]]  // NO-FORMAT: Attribute
      T t_matrix_consume[kColumns][kColumns];

      // Update the global T matrix with the partial results and copy the result
      // to the t_matrix_consume array for better memory structure in the
      // computation of COV
      for (row = 0; row < kColumns; row++) {
        fpga_tools::UnrolledLoop<kColumns>([&](auto column) {
          T t_matrix_to_add = block == 0 ? 0 : t_matrix[row * kColumns + column];
          T sum = t_matrix_compute[row * kColumns + column] + t_matrix_to_add;
          t_matrix[row * kColumns + column] = sum;
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

        fpga_tools::NTuple<T, kPipeSize> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterationPerRow>([&](auto t) {
          fpga_tools::UnrolledLoop<kPipeSize>([&](auto k) {
            if constexpr (t * kPipeSize + k < kColumns) {
              pipe_write.template get<k>() =
                  get[t]
                      ? t_matrix_consume[li / kLoopIterationPerRow][t * kPipeSize + k]
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
