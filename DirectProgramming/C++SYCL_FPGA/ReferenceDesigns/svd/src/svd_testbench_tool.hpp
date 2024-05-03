#ifndef __SVD_TESTBENCH_TOOL_HPP__
#define __SVD_TESTBENCH_TOOL_HPP__

#include <iostream>
#include <vector>

/*
* This file contains helper functions that is used in the host test bench
* of the SVD demonstration to check for correctness.
*/

#define EPSILON 2E-6

namespace svd_testbench_tool {  // not for kernel code

template <typename TT>
void SoftTranspose(std::vector<std::vector<TT>> &origin, unsigned rows,
                   unsigned cols, std::vector<std::vector<TT>> &transposed) {
  // just swap row and col
  for (unsigned row = 0; row < rows; row++) {
    for (unsigned col = 0; col < cols; col++) {
      transposed[col][row] = origin[row][col];
    }
  }
}

// for col major matrix
template <typename TT>
void SoftTranspose(std::vector<TT> &mat_A, unsigned rows, unsigned cols,
                   std::vector<TT> &mat_At) {
  for (int i = 0; i < (rows * cols); i++) {
    int cur_col = int(i / rows);
    int cur_row = i % rows;
    mat_At[cur_row * cols + cur_col] = mat_A[i];
  }
}

template <typename TT>
void SoftMatmult(std::vector<std::vector<TT>> &mat_A, unsigned rows_A,
                 unsigned cols_A, std::vector<std::vector<TT>> &mat_B,
                 unsigned rows_B, unsigned cols_B,
                 std::vector<std::vector<TT>> &mat_AB) {
  assert((cols_A == rows_B) && "Mat_Mult with illegal matrix sizes");
  // Initializing AB to 0s
  for (unsigned row = 0; row < rows_A; row++) {
    for (unsigned col = 0; col < cols_B; col++) {
      mat_AB[row][col] = 0.0;
    }
  }
  // Multiplying matrix A and B and storing in AB.
  for (unsigned row = 0; row < rows_A; row++) {
    for (unsigned col = 0; col < cols_B; col++) {
      for (unsigned item = 0; item < cols_A; item++) {
        mat_AB[row][col] += mat_A[row][item] * mat_B[item][col];
      }
    }
  }
}

// for col major matrix
template <typename TT>
void SoftMatmult(std::vector<TT> &mat_A, unsigned rows_A, unsigned cols_A,
                 std::vector<TT> &mat_B, unsigned rows_B, unsigned cols_B,
                 std::vector<TT> &mat_AB) {
  std::vector<std::vector<TT>> a_2d(rows_A, std::vector<TT>(cols_A, 0));
  std::vector<std::vector<TT>> b_2d(rows_B, std::vector<TT>(cols_B, 0));
  std::vector<std::vector<TT>> ab_2d(rows_A, std::vector<TT>(cols_B, 0));

  // turn A vertical
  for (int i = 0; i < (rows_A * cols_A); i++) {
    a_2d[i % rows_A][int(i / rows_A)] = mat_A[i];
  }

  // turn B vertical
  for (int i = 0; i < (rows_B * cols_B); i++) {
    b_2d[i % rows_B][int(i / rows_B)] = mat_B[i];
  }

  SoftMatmult<TT>(a_2d, rows_A, cols_A, b_2d, rows_B, cols_B, ab_2d);

  for (int c = 0; c < cols_B; c++) {
    for (int r = 0; r < rows_A; r++) {
      mat_AB[c * rows_A + r] = ab_2d[r][c];
    }
  }
}

template <typename T>
bool IsRankDeficient(std::vector<std::vector<T>> &input_matrix) {
  std::vector<std::vector<T>> tempMatrix = input_matrix;

  int numRows = tempMatrix.size();
  int numCols = tempMatrix[0].size();
  int minDim = std::min(numRows, numCols);
  // doing gaussian elimination on the local matrix
  for (int pivot = 0; pivot < minDim; ++pivot) {
    int maxRow = pivot;
    for (int row = pivot + 1; row < numRows; ++row) {
      if (std::abs(tempMatrix[row][pivot]) >
          std::abs(tempMatrix[maxRow][pivot])) {
        maxRow = row;
      }
    }

    if (tempMatrix[maxRow][pivot] == 0.0) {
      continue;
    }

    if (maxRow != pivot) {
      std::swap(tempMatrix[pivot], tempMatrix[maxRow]);
    }

    for (int row = pivot + 1; row < numRows; ++row) {
      T factor = tempMatrix[row][pivot] / tempMatrix[pivot][pivot];
      for (int col = pivot; col < numCols; ++col) {
        tempMatrix[row][col] -= factor * tempMatrix[pivot][col];
      }
    }
  }

  // check for zero row after elimination
  for (int row = 0; row < numRows; ++row) {
    bool allZeroes = true;
    for (int col = 0; col < numCols; ++col) {
      if (std::abs(tempMatrix[row][col]) > 1e-6) {
        allZeroes = false;
        break;
      }
    }
    if (allZeroes) {
      return true;  // If a row of zeros is found, matrix is rank-deficient
    }
  }
  return false;  // No row of zeros found, matrix is full rank
}

template <typename T>
T RandomValueInInterval(T min, T max) {
  return min +
         static_cast<T>(rand()) / (static_cast<T>(RAND_MAX) / (max - min));
}

template <typename T>
void GenMatrix(std::vector<std::vector<T>> &output_mat, int rows, int cols,
               T min, T max) {
  output_mat.resize(rows);
  for (int r = 0; r < rows; ++r) {
    output_mat[r].resize(cols);
    for (int c = 0; c < cols; ++c) {
      output_mat[r][c] = RandomValueInInterval<T>(min, max);
    }
  }
}

// return a sub matrix of a stream of matrices
template <typename T>
std::vector<T> subMatrix(std::vector<T> &og_matrix, int mat_idx, int rows,
                         int cols) {
  int mat_size = rows * cols;
  int start = mat_idx * mat_size;
  int end = (mat_idx + 1) * mat_size;
  std::vector<T> sub_matrix(og_matrix.begin() + start, og_matrix.begin() + end);
  return sub_matrix;
}

}  // namespace svd_testbench_tool

#endif /* __SVD_TESTBENCH_TOOL_HPP__ */