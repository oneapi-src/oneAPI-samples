//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// data_bundle.hpp

#ifndef __DATA_BUNDLE_HPP__
#define __DATA_BUNDLE_HPP__

namespace fpga_tools {

/////////////////////////////////////////////////////////////////////////////////
// C++ magic that lets us extract template parameters from DataBundle instances
/////////////////////////////////////////////////////////////////////////////////

/// @brief Extract template parameters from a DataBundle. Usage: `constexpr int
/// kPixelsInParallel =
/// fpga_tools::ExtractDataBundleType<DataBundleType>::kBundlePayloadCount;`
/// @tparam T The DataBundle whose parameters you wish to extract
template <typename T>
struct ExtractDataBundleType {
  typedef T BundlePayloadT;
  static constexpr int kBundlePayloadCount = 0;
};

/// @brief Extract template parameters from a DataBundle. Usage:
/// `constexpr int kPixelsInParallel =
/// fpga_tools::ExtractDataBundleType<DataBundleType>::kBundlePayloadCount;`
template <template <typename, int> typename DataBundle, typename BundlePayload,
          int kBundleCount>
struct ExtractDataBundleType<
    DataBundle<BundlePayload, kBundleCount>>  // specialization
{
  typedef BundlePayload BundlePayloadT;
  static constexpr int kBundlePayloadCount = kBundleCount;
};

/// @brief Utility class for handling vectorized data
///
/// @paragraph This class allows a user to use an array of POD types in their
/// SYCL code. It has been optimized for FPGA to ensure that compiler-inferred
/// loops are unrolled, and allow for data-parallel operation. Since the array
/// is fully unrolled, take care to keep your `DataBundle` objects to a
/// reasonable size, or you may experience routing congestion in the FPGA
/// fabric. A good rule of thumb is to ensure that you select `T` and
/// `kBundleSize` such that `sizeof(DataBundle<T, kBundleSize>)` is less than
/// 512 bytes.
///
/// @tparam T The type of elements in the `DataBundle`
///
/// @tparam kBundleSize the number of elements in the `DataBundle`
template <typename T, int kBundleSize>
struct DataBundle {
  T data_[kBundleSize];

  DataBundle() {}

  DataBundle(const T op) {
    // unroll in case this constructor is implemented in a oneAPI FPGA kernel
#pragma unroll
    for (int idx = 0; idx < kBundleSize; idx++) {
      data_[idx] = op;
    }
  }

  DataBundle(const DataBundle &op) {
    // unroll in case this copy constructor is implemented in a oneAPI FPGA
    // kernel
#pragma unroll
    for (int idx = 0; idx < kBundleSize; idx++) {
      data_[idx] = op.data_[idx];
    }
  }

  DataBundle &operator=(const DataBundle &other) {
    // unroll in case this assign operator is implemented in a oneAPI FPGA
    // kernel
#pragma unroll
    for (int idx = 0; idx < kBundleSize; idx++) {
      data_[idx] = other.data_[idx];
    }

    return *this;
  }

  bool operator==(const DataBundle &rhs) {
    bool is_equal = true;
    // unroll in case this comparison is implemented in a oneAPI FPGA kernel
#pragma unroll
    for (int b = 0; b < kBundleSize; b++) {
      is_equal &= (data_[b] == rhs.data_[b]);
    }

    return is_equal;
  }

  /// @brief Access a specific value in the bundle (like a C-array)
  /// @param i index in the `DataBundle` that you want to access
  /// @return the reference to the type
  T &operator[](int i) { return data_[i]; }

  /// @brief Get a pointer to underlying data.
  /// @note This may be useful for bulk copy operations in SYCL* host
  /// code, or for passing references to the underlying data to subroutines in
  /// SYCL* device code.
  /// @return A C-style pointer to the underlying data of the `DataBundle`.
  T *data() { return &data_[0]; }

  /// @brief Add a new element to the `DataBundle` at location `[kBundleSize -
  /// 1]`, and shift all members of the `DataBundle` to the left. The element at
  /// location `0` is removed.
  ///
  /// @details For a shift register with N columns, the first piece of data is
  /// inserted in location [N-1], and is read out of location [0].
  ///
  /// ```
  ///         i=0  1   2          \n
  ///        ┌───┬───┬───┐        \n
  /// out ◄─ │ r ◄─e ◄─g ◄─ input \n
  ///        └───┴───┴───┘        \n
  /// ```
  ///
  /// @param[in] in The value to insert
  void Shift(T &in) {
    T return_val = data_[0];
#pragma unroll
    for (int i = 0; i < (kBundleSize - 1); i++) {
      data_[i] = data_[i + 1];
    }
    data_[kBundleSize - 1] = in;

    return return_val;
  }

  /// @brief Add multiple copies of a new element to the `DataBundle` at
  /// locations `[kBundleSize - 1]`, `[kBundleSize - 2]`, etc. and shift all
  /// members of the `DataBundle` to the left. The elements at location `0`,
  /// `1`, etc. are removed.
  /// @tparam kShiftAmt The number of copies of the new element to insert
  /// @param[in] in The new element to insert
  template <int kShiftAmt>
  void ShiftSingleVal(T &in) {
#pragma unroll
    for (int i = 0; i < (kBundleSize - kShiftAmt); i++) {
      data_[i] = data_[i + kShiftAmt];
    }

#pragma unroll
    for (int i = 0; i < (kShiftAmt); i++) {
      data_[(kBundleSize - kShiftAmt) + i] = in;
    }
  }

  /// @brief  Add multiple new elements to the `DataBundle` at locations
  /// `[kBundleSize - 1]`, `[kBundleSize - 2]`, etc. and shift all members of
  /// the `DataBundle` to the left. The elements at location `0`, `1`, etc.
  /// are removed.
  /// @tparam kShiftAmt The number of new elements to insert
  /// @tparam kBundleSize2 (Optional) The number of new elements in the
  /// `DataBundle` containing the new elements. This number should be greater
  /// than or equal to `kShiftAmt`.
  /// @param[in] in A `DataBundle` holding new elements to shift in. If the size
  /// of this type is not `kShiftAmt`, you must set the `kBundleSize2` template
  /// parameter.
  template <int kShiftAmt, int kBundleSize2 = kShiftAmt>
  void ShiftMultiVals(DataBundle<T, kBundleSize2> &in) {
    static_assert(
        kBundleSize2 >= kShiftAmt,
        "kBundleSize2 should be greater than or equal to `kShiftAmt`.");

#pragma unroll
    for (int i = 0; i < (kBundleSize - kShiftAmt); i++) {
      data_[i] = data_[i + kShiftAmt];
    }

#pragma unroll
    for (int i = 0; i < (kShiftAmt); i++) {
      data_[(kBundleSize - kShiftAmt) + i] = in[i];
    }
  }
};

}  // namespace fpga_tools

#endif