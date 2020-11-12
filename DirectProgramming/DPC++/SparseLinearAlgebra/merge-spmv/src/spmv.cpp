//==============================================================
// This sample provides a parallel implementation of a merge based sparse matrix
// and vector multiplication algorithm using DPC++. The input matrix is in
// compressed sparse row format.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>
#include <map>
#include <set>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

// n x n sparse matrix.
constexpr int n = 100 * 1000;

// Number of non zero values in sparse matrix.
constexpr int nonzero = 2 * 1000 * 1000;

// Maximum value of an element in the matrix.
constexpr int max_value = 100;

// Number of repetitions.
constexpr int repetitions = 16;

// Compressed Sparse Row (CSR) representation for sparse matrix.
//
// Example: The following 4 x 4 sparse matrix
//
//   a 0 0 0
//   b c 0 0
//   0 0 0 d
//   0 0 e f
//
// have 6 non zero elements in it:
//
//   Index  Row  Column  Value
//       0    0       0      a
//       1    1       0      b
//       2    1       1      c
//       3    2       3      d
//       4    3       2      e
//       5    3       3      f
//
// Its CSR representation is have three components:
// - Nonzero values: a, b, c, d, e, f
// - Column indices: 0, 0, 1, 3, 2, 3
// - Row offsets: 0, 1, 3, 4, 6
//
// Non zero values and their column indices directly correspond to the entries
// in the above table.
//
// Row offsets are offsets in the values array for the first non zero element of
// each row of the matrix.
//
//   Row  NonZeros  NonZeros_SeenBefore
//     0         1                    0
//     1         2                    1
//     2         1                    3
//     3         2                    4
//     -         -                    6
typedef struct {
  int *row_offsets;
  int *column_indices;
  float *values;
} CompressedSparseRow;

// Allocate unified shared memory for storing matrix and vectors so that they
// are accessible from both the CPU and the device (e.g., a GPU).
bool AllocateMemory(queue &q, int thread_count, CompressedSparseRow *matrix,
                    float **x, float **y_sequential, float **y_parallel,
                    int **carry_row, float **carry_value) {
  matrix->row_offsets = malloc_shared<int>(n + 1, q);
  matrix->column_indices = malloc_shared<int>(nonzero, q);
  matrix->values = malloc_shared<float>(nonzero, q);

  *x = malloc_shared<float>(n, q);
  *y_sequential = malloc_shared<float>(n, q);
  *y_parallel = malloc_shared<float>(n, q);

  *carry_row = malloc_shared<int>(thread_count, q);
  *carry_value = malloc_shared<float>(thread_count, q);

  return (matrix->row_offsets != nullptr) &&
         (matrix->column_indices != nullptr) && (matrix->values != nullptr) &&
         (*x != nullptr) && (*y_sequential != nullptr) &&
         (*y_parallel != nullptr) && (*carry_row != nullptr) &&
         (*carry_value != nullptr);
}

// Free allocated unified shared memory.
void FreeMemory(queue &q, CompressedSparseRow *matrix, float *x,
                float *y_sequential, float *y_parallel, int *carry_row,
                float *carry_value) {
  if (matrix->row_offsets != nullptr) free(matrix->row_offsets, q);
  if (matrix->column_indices != nullptr) free(matrix->column_indices, q);
  if (matrix->values != nullptr) free(matrix->values, q);

  if (x != nullptr) free(x, q);
  if (y_sequential != nullptr) free(y_sequential, q);
  if (y_parallel != nullptr) free(y_parallel, q);

  if (carry_row != nullptr) free(carry_row, q);
  if (carry_value != nullptr) free(carry_value, q);
}

// Initialize inputs: sparse matrix and vector.
void InitializeSparseMatrixAndVector(CompressedSparseRow *matrix, float *x) {
  map<int, set<int>> indices;

  // Randomly choose a set of elements (i.e., row and column pairs) of the
  // matrix. These elements will have non zero values.
  for (int k = 0; k < nonzero; k++) {
    int i = rand() % n;
    int j = rand() % n;

    if (indices.find(i) == indices.end()) {
      indices[i] = {j};

    } else if (indices[i].find(j) == indices[i].end()) {
      indices[i].insert(j);

    } else {
      k--;
    }
  }

  int offset = 0;

  // Randomly choose non zero values of the sparse matrix.
  for (int i = 0; i < n; i++) {
    matrix->row_offsets[i] = offset;

    if (indices.find(i) != indices.end()) {
      set<int> &cols = indices[i];

      for (auto it = cols.cbegin(); it != cols.cend(); ++it, ++offset) {
        matrix->column_indices[offset] = *it;
        matrix->values[offset] = rand() % max_value + 1;
      }
    }
  }

  matrix->row_offsets[n] = nonzero;

  // Initialize input vector.
  for (int i = 0; i < n; i++) {
    x[i] = 1;
  }
}

// A sequential implementation of merge based sparse matrix and vector
// multiplication algorithm.
//
// Both row offsets and values indices can be thought of as sorted arrays. The
// progression of the computation is similar to that of merging two sorted
// arrays at a conceptual level.
//
// When a row offset and an index of the values array are equal (denoted as '?'
// below), the algorithm starts computing the value of a new element of the
// result vector.
//
// The algorithm continues to accumulate for the same element of the result
// vector otherwise (denoted as '*' below).
//
// Row indices ->  0 1 2 3
// Row offsets ->  0 1 3 4 6
//
//                 ?         0  a
//                   ?       1  b
//                   *       2  c
//                     ?     3  d
//                       ?   4  e
//                       *   5  f
//
//                           ^  ^
//                           |  |
//                           |  Non zero values
//                           |
//                           Indices of values array
void MergeSparseMatrixVector(CompressedSparseRow *matrix, float *x, float *y) {
  int row_index = 0;
  int val_index = 0;

  y[row_index] = 0;

  while (val_index < nonzero) {
    if (val_index < matrix->row_offsets[row_index + 1]) {
      // Accumulate and move down.
      y[row_index] +=
          matrix->values[val_index] * x[matrix->column_indices[val_index]];
      val_index++;

    } else {
      // Move right.
      row_index++;
      y[row_index] = 0;
    }
  }

  for (row_index++; row_index < n; row_index++) {
    y[row_index] = 0;
  }
}

// Merge Coordinate.
typedef struct {
  int row_index;
  int val_index;
} MergeCoordinate;

// Given linear position on the merge path, find two dimensional merge
// coordinate (row index and value index pair) on the path.
MergeCoordinate MergePathBinarySearch(int diagonal, int *row_offsets) {
  // Diagonal search range (in row index space).
  int row_min = std::max(diagonal - nonzero, 0);
  int row_max = std::min(diagonal, n);

  // 2D binary search along the diagonal search range.
  while (row_min < row_max) {
    int pivot = (row_min + row_max) >> 1;

    if (row_offsets[pivot + 1] <= diagonal - pivot - 1) {
      // Keep top right half of diagonal range.
      row_min = pivot + 1;
    } else {
      // Keep bottom left half of diagonal range.
      row_max = pivot;
    }
  }

  MergeCoordinate coordinate;

  coordinate.row_index = std::min(row_min, n);
  coordinate.val_index = diagonal - row_min;

  return coordinate;
}

// The parallel implementation of spare matrix, vector multiplication algorithm
// uses this function as a subroutine. Each available thread calls this function
// with identical inputs, except the thread identifier (TID) is unique. Having a
// unique TID, each thread independently identifies its own, non overlapping
// share of the overall work. More importantly, each thread, except possibly the
// last one, handles the same amount of work. This implementation is an
// extension of the sequential implementation of the merge based sparse matrix,
// vector multiplication algorithm. It first identifies its scope of the merge
// and then performs only the amount of work that belongs this thread in the
// cohort of threads.
void MergeSparseMatrixVectorThread(int thread_count, int tid,
                                   CompressedSparseRow matrix, float *x,
                                   float *y, int *carry_row,
                                   float *carry_value) {
  int path_length = n + nonzero;  // Merge path length.
  int items_per_thread = (path_length + thread_count - 1) /
                         thread_count;  // Merge items per thread.

  // Find start and end merge path coordinates for this thread.
  int diagonal = std::min(items_per_thread * tid, path_length);
  int diagonal_end = std::min(diagonal + items_per_thread, path_length);

  MergeCoordinate path = MergePathBinarySearch(diagonal, matrix.row_offsets);
  MergeCoordinate path_end =
      MergePathBinarySearch(diagonal_end, matrix.row_offsets);

  // Consume items-per-thread merge items.
  float dot_product = 0;

  for (int i = 0; i < items_per_thread; i++) {
    if (path.val_index < matrix.row_offsets[path.row_index + 1]) {
      // Accumulate and move down.
      dot_product += matrix.values[path.val_index] *
                     x[matrix.column_indices[path.val_index]];
      path.val_index++;

    } else {
      // Output row total and move right.
      y[path.row_index] = dot_product;
      dot_product = 0;
      path.row_index++;
    }
  }

  // Save carry.
  carry_row[tid] = path_end.row_index;
  carry_value[tid] = dot_product;
}

// This is the parallel implementation of merge based sparse matrix and vector
// mutiplication algorithm. It works in three steps:
//   1. Initialize elements of the output vector to zero.
//   2. Multiply sparse matrix and vector.
//   3. Fix up rows of the output vector that spanned across multiple threads.
// First two steps are parallel. They utilize all available processors
// (threads). The last step performs a reduction. It could be parallel as well
// but is kept as sequential for the following reasons:
//   1. Number of operation in this step is proportional to the number of
//   processors (threads).
//   2. Number of available threads is not too high.
void MergeSparseMatrixVector(queue &q, int compute_units, int work_group_size,
                             CompressedSparseRow matrix, float *x, float *y,
                             int *carry_row, float *carry_value) {
  int thread_count = compute_units * work_group_size;

  // Initialize output vector.
  q.parallel_for<class InitializeVector>(
      nd_range<1>(compute_units * work_group_size, work_group_size),
      [=](nd_item<1> item) {
        auto global_id = item.get_global_id(0);
        auto items_per_thread = (n + thread_count - 1) / thread_count;
        auto start = global_id * items_per_thread;
        auto stop = start + items_per_thread;

        for (auto i = start; (i < stop) && (i < n); i++) {
          y[i] = 0;
        }
      });

  q.wait();

  // Multiply sparse matrix and vector.
  q.parallel_for<class MergeCsrMatrixVector>(
      nd_range<1>(compute_units * work_group_size, work_group_size),
      [=](nd_item<1> item) {
        auto global_id = item.get_global_id(0);
        MergeSparseMatrixVectorThread(thread_count, global_id, matrix, x, y,
                                      carry_row, carry_value);
      });

  q.wait();

  // Carry fix up for rows spanning multiple threads.
  for (int tid = 0; tid < thread_count - 1; tid++) {
    if (carry_row[tid] < n) {
      y[carry_row[tid]] += carry_value[tid];
    }
  }
}

// Check if two input vectors are equal.
bool VerifyVectorsAreEqual(float *u, float *v) {
  for (int i = 0; i < n; i++) {
    if (fabs(u[i] - v[i]) > 1E-06) {
      return false;
    }
  }

  return true;
}

int main() {
  // Sparse matrix.
  CompressedSparseRow matrix;

  // Input vector.
  float *x;

  // Vector: result of sparse matrix and vector multiplication.
  float *y_sequential;
  float *y_parallel;

  // Auxiliary storage for parallel computation.
  int *carry_row;
  float *carry_value;

  try {
    queue q{default_selector{}, dpc_common::exception_handler};
    auto device = q.get_device();

    cout << "Device: " << device.get_info<info::device::name>() << "\n";

    // Find max number of compute/execution units and max number of threads per
    // compute unit.
    auto compute_units = device.get_info<info::device::max_compute_units>();
    auto work_group_size = device.get_info<info::device::max_work_group_size>();

    cout << "Compute units: " << compute_units << "\n";
    cout << "Work group size: " << work_group_size << "\n";

    // Allocate memory.
    if (!AllocateMemory(q, compute_units * work_group_size, &matrix, &x,
                        &y_sequential, &y_parallel, &carry_row, &carry_value)) {
      cout << "Memory allocation failure.\n";
      FreeMemory(q, &matrix, x, y_sequential, y_parallel, carry_row,
                 carry_value);
      return -1;
    }

    // Initialize.
    InitializeSparseMatrixAndVector(&matrix, x);

    // Warm up the JIT.
    MergeSparseMatrixVector(q, compute_units, work_group_size, matrix, x,
                            y_parallel, carry_row, carry_value);

    // Time executions.
    double elapsed_s = 0;
    double elapsed_p = 0;
    int i;

    cout << "Repeating " << repetitions << " times to measure run time ...\n";

    for (i = 0; i < repetitions; i++) {
      cout << "Iteration: " << (i + 1) << "\n";

      // Sequential compute.
      dpc_common::TimeInterval timer_s;

      MergeSparseMatrixVector(&matrix, x, y_sequential);
      elapsed_s += timer_s.Elapsed();

      // Parallel compute.
      dpc_common::TimeInterval timer_p;

      MergeSparseMatrixVector(q, compute_units, work_group_size, matrix, x,
                              y_parallel, carry_row, carry_value);
      elapsed_p += timer_p.Elapsed();

      // Verify two results are equal.
      if (!VerifyVectorsAreEqual(y_sequential, y_parallel)) {
        cout << "Failed to correctly compute!\n";
        break;
      }
    }

    if (i == repetitions) {
      cout << "Successfully completed sparse matrix and vector "
              "multiplication!\n";

      elapsed_s /= repetitions;
      elapsed_p /= repetitions;

      cout << "Time sequential: " << elapsed_s << " sec\n";
      cout << "Time parallel: " << elapsed_p << " sec\n";
    }

    FreeMemory(q, &matrix, x, y_sequential, y_parallel, carry_row, carry_value);
  } catch (std::exception const &e) {
    cout << "An exception is caught while computing on device.\n";
    terminate();
  }

  return 0;
}
