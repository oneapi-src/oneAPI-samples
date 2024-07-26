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
  T &operator()(size_t row, size_t col) { return mData[row * mCols + col]; }
  T operator()(size_t row, size_t col) const {
    return mData[row * mCols + col];
  }

  T &operator()(size_t idx) { return mData[idx]; }
  T operator()(size_t idx) const { return mData[idx]; }

  /// Number of rows (image height)
  size_t GetRows() const { return mRows; }

  /// Number of columns (image width)
  size_t GetCols() const { return mCols; }

 private:
  size_t mRows;
  size_t mCols;
  std::vector<T> mData;
};