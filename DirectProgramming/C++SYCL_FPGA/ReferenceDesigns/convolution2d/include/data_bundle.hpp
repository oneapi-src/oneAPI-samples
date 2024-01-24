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
/// fpga_tools::extractDataBundleType<DataBundleType>::BundlePayloadCount;`
/// @tparam T The DataBundle whose parameters you wish to extract
template <typename T>
struct extractDataBundleType {
  typedef T BundlePayloadT;
  static constexpr int BundlePayloadCount = 0;
};

/// @brief Extract template parameters from a DataBundle. Usage: `constexpr int
/// kPixelsInParallel =
/// fpga_tools::extractDataBundleType<DataBundleType>::BundlePayloadCount;`
/// @tparam T The DataBundle whose parameters you wish to extract
template <template <typename, int> typename DATA_BUNDLE,
          typename BUNDLE_PAYLOAD, int BUNDLE_COUNT>
struct extractDataBundleType<
    DATA_BUNDLE<BUNDLE_PAYLOAD, BUNDLE_COUNT>>  // specialization
{
  typedef BUNDLE_PAYLOAD BundlePayloadT;
  static constexpr int BundlePayloadCount = BUNDLE_COUNT;
};

/// @brief Utility class for handling vectorized data
///
/// @paragraph This class allows a user to use an array of POD types in their
/// SYCL code. It has been optimized for FPGA to ensure that compiler-inferred
/// loops are unrolled, and allow for data-parallel operation. Since the array
/// is fully unrolled, take care to keep your `DataBundle` objects to a
/// reasonable size, or you may experience routing congestion in the FPGA
/// fabric. A good rule of thumb is to ensure that you select `T` and
/// `BUNDLE_SIZE` such that `sizeof(DataBundle<T, BUNDLE_SIZE>)` is less than
/// 512 bytes.
///
/// @tparam T The type of elements in the `DataBundle`
///
/// @tparam BUNDLE_SIZE the number of elements in the `DataBundle`
template <typename T, int BUNDLE_SIZE>
struct DataBundle {
  T data_[BUNDLE_SIZE];

  DataBundle() {}

  DataBundle(const T op) {
    // unroll in case this constructor is implemented in a oneAPI FPGA kernel
#pragma unroll
    for (int idx = 0; idx < BUNDLE_SIZE; idx++) {
      data_[idx] = op;
    }
  }

  DataBundle(const DataBundle &op) {
    // unroll in case this copy constructor is implemented in a oneAPI FPGA
    // kernel
#pragma unroll
    for (int idx = 0; idx < BUNDLE_SIZE; idx++) {
      data_[idx] = op.data_[idx];
    }
  }

  DataBundle &operator=(const DataBundle &other) {
    // unroll in case this assign operator is implemented in a oneAPI FPGA
    // kernel
#pragma unroll
    for (int idx = 0; idx < BUNDLE_SIZE; idx++) {
      data_[idx] = other.data_[idx];
    }

    return *this;
  }

  bool operator==(const DataBundle &rhs) {
    bool isEqual = true;
    // unroll in case this comparison is implemented in a oneAPI FPGA kernel
#pragma unroll
    for (int b = 0; b < BUNDLE_SIZE; b++) {
      isEqual &= (data_[b] == rhs.data_[b]);
    }

    return isEqual;
  }

  /// @brief Access a specific value in the bundle (like a C-array)
  /// @param i index in the `DataBundle` that you want to access
  /// @return the reference to the type
  T &operator[](int i) { return data_[i]; }

  /// @brief get a pointer to underlying data.
  /// @note This may be useful for bulk copy operations in SYCL* host
  /// code, or for passing references to the underlying data to subroutines in
  /// SYCL* device code.
  /// @return A C-style pointer to the underlying data of the `DataBundle`.
  T *data() { return &data_[0]; }

  /// @brief Add a new element to the `DataBundle` at location `[BUNDLE_SIZE -
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
  /// @param in The value to insert
  ///
  /// @return The value that was removed
  void shift(T &in) {
    T retVal = data_[0];
#pragma unroll
    for (int i = 0; i < (BUNDLE_SIZE - 1); i++) {
      data_[i] = data_[i + 1];
    }
    data_[BUNDLE_SIZE - 1] = in;

    return retVal;
  }

  /// @brief Add multiple copies of a new element to the `DataBundle` at
  /// locations `[BUNDLE_SIZE - 1]`, `[BUNDLE_SIZE - 2]`, etc. and shift all
  /// members of the `DataBundle` to the left. The elements at location `0`,
  /// `1`, etc. are removed.
  /// @tparam SHIFT_AMT The number of copies of the new element to insert
  /// @param in The new element to insert
  template <int SHIFT_AMT>
  void shiftSingleVal(T &in) {
#pragma unroll
    for (int i = 0; i < (BUNDLE_SIZE - SHIFT_AMT); i++) {
      data_[i] = data_[i + SHIFT_AMT];
    }

#pragma unroll
    for (int i = 0; i < (SHIFT_AMT); i++) {
      data_[(BUNDLE_SIZE - SHIFT_AMT) + i] = in;
    }
  }

  /// @brief  Add multiple new elements to the `DataBundle` at locations
  /// `[BUNDLE_SIZE - 1]`, `[BUNDLE_SIZE - 2]`, etc. and shift all members of
  /// the `DataBundle` to the left. The elements at location `0`, `1`, etc.
  /// are removed.
  ///
  /// @tparam SHIFT_AMT The number of new elements to insert
  ///
  /// @tparam BUNDLE_SZ (Optional) The number of new elements in the
  /// `DataBundle` containing the new elements. This number should be greater
  /// than or equal to `SHIFT_AMT`.
  ///
  /// @param in A `DataBundle` holding new elements to shift in. If the size
  /// of this type is not `SHIFT_AMT`, you must set the `BUNDLE_SZ` template
  /// parameter.
  template <int SHIFT_AMT, int BUNDLE_SZ = SHIFT_AMT>
  void shiftMultiVals(DataBundle<T, BUNDLE_SZ> &in) {
#pragma unroll
    for (int i = 0; i < (BUNDLE_SIZE - SHIFT_AMT); i++) {
      data_[i] = data_[i + SHIFT_AMT];
    }

#pragma unroll
    for (int i = 0; i < (SHIFT_AMT); i++) {
      data_[(BUNDLE_SIZE - SHIFT_AMT) + i] = in[i];
    }
  }
};

}  // namespace fpga_tools

#endif