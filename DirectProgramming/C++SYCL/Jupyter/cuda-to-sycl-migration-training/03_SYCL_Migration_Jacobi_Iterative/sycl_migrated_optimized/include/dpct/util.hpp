//==---- util.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_UTIL_HPP__
#define __DPCT_UTIL_HPP__

#include <sycl/sycl.hpp>
#include <complex>
#include <type_traits>
#include <cassert>
#include <cstdint>


namespace dpct {

namespace detail {

template <typename tag, typename T> class generic_error_type {
public:
  generic_error_type() = default;
  generic_error_type(T value) : value{value} {}
  operator T() const { return value; }

private:
  T value;
};

} // namespace detail

using err0 = detail::generic_error_type<struct err0_tag, int>;
using err1 = detail::generic_error_type<struct err1_tag, int>;

/// shift_sub_group_left move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the left. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is caller's id adds \p delta. If calculated id is outside the logical
/// sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = sycl::shift_group_left(g, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
}

namespace experimental {

/// The logical-group is a logical collection of some work-items within a
/// work-group.
/// Note: Please make sure that the logical-group size is a power of 2 in the
/// range [1, current_sub_group_size].
class logical_group {
  sycl::nd_item<3> _item;
  sycl::group<3> _g;
  uint32_t _logical_group_size;
  uint32_t _group_linear_range_in_parent;

public:
  /// Dividing \p parent_group into several logical-groups.
  /// \param [in] item Current work-item.
  /// \param [in] parent_group The group to be divided.
  /// \param [in] size The logical-group size.
  logical_group(sycl::nd_item<3> item, sycl::group<3> parent_group,
                uint32_t size)
      : _item(item), _g(parent_group), _logical_group_size(size) {
    _group_linear_range_in_parent =
        (_g.get_local_linear_range() - 1) / _logical_group_size + 1;
  }

  /// Returns the index of the work-item within the logical-group.
  uint32_t get_local_linear_id() const {
    return _item.get_local_linear_id() % _logical_group_size;
  }

  /// Returns the number of work-items in the logical-group.
  uint32_t get_local_linear_range() const {
    if (_g.get_local_linear_range() % _logical_group_size == 0) {
      return _logical_group_size;
    }
    uint32_t last_item_group_id =
        _g.get_local_linear_range() / _logical_group_size;
    uint32_t first_of_last_group = last_item_group_id * _logical_group_size;
    if (_item.get_local_linear_id() >= first_of_last_group) {
      return _g.get_local_linear_range() - first_of_last_group;
    } else {
      return _logical_group_size;
    }
  }

};

} // namespace experimental

} // namespace dpct

#endif // __DPCT_UTIL_HPP__
