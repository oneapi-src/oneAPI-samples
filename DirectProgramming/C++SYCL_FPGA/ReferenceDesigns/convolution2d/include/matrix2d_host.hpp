#pragma once
#include <vector>

/// @brief Abstraction of a 2D matrix. The individual pixel values may be
/// changed at runtime, but the dimensions are fixed. This class should only be
/// used in host code, since it depends on std::vector.
template <typename T>
class Matrix2d {
 public:
  Matrix2d(size_t rows, size_t cols)
      : mRows(rows), mCols(cols), mData(rows * cols) {}
  Matrix2d(const Matrix2d<T> &other)
      : mRows(other.GetRows()),
        mCols(other.GetCols()),
        mData(other.GetRows() * other.GetCols()) {
    std::copy(other.mData.begin(), other.mData.end(), mData.begin());
  }
  /// @brief Access an element of the `Matrix2d`
  /// @param row Matrix row
  /// @param col Matrix column
  /// @return Element of `Matrix2d`
  T &operator()(size_t row, size_t col) { return mData[row * mCols + col]; }

  /// @brief Access an element of the `Matrix2d`
  /// @param row Matrix row
  /// @param col Matrix column
  /// @return Element of `Matrix2d`
  T operator()(size_t row, size_t col) const {
    return mData[row * mCols + col];
  }

  /// @brief Access an element of the `Matrix2d` in row-major sequence. This
  /// means a 2x4 matrix would be accessed like this:
  /// @paragraph Schematic
  /// ```
  /// /             \ <br/>
  /// | 0, 1, 2, 3, | <br/>
  /// | 4, 5, 6, 7, | <br/>
  /// \             / <br/>
  /// ```
  /// @param idx Matrix element (row-major)
  /// @return Element of `Matrix2d`
  T &operator()(size_t idx) { return mData[idx]; }

  /// @brief Access an element of the `Matrix2d` in row-major sequence. This
  /// means a 2x4 matrix would be accessed like this:
  /// @paragraph Schematic
  /// ```
  /// /             \ <br/>
  /// | 0, 1, 2, 3, | <br/>
  /// | 4, 5, 6, 7, | <br/>
  /// \             / <br/>
  /// ```
  /// @param idx Matrix element (row-major)
  /// @return Element of `Matrix2d`
  T operator()(size_t idx) const { return mData[idx]; }

  /// Number of rows (image height)
  size_t GetRows() const { return mRows; }

  /// Number of columns (image width)
  size_t GetCols() const { return mCols; }

  /// @brief total number of elements in this Matrix2D
  /// @return Element of `Matrix2d`
  size_t Size() const { return mRows * mCols; }

 private:
  const size_t mRows;
  const size_t mCols;
  std::vector<T> mData;
};