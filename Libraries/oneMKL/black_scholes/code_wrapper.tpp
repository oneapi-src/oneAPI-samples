//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*******************************************************************************
  !  Content:
  !      Wrapper utility for backward compatibility with get_cl_code in SYCL 1.2.1
  !******************************************************************************/

#pragma once
#include <utility>

template <typename T, typename = void>
struct has_member_code_meta : std::false_type {};

template <typename T>
struct has_member_code_meta<T, std::void_t<decltype( std::declval<T>().code() )> > : std::true_type {};

template <typename T, typename std::enable_if<has_member_code_meta<T>::value>::type* = nullptr >
auto code_wrapper (T x) {
    return x.code();
};
template <typename T, typename std::enable_if<!has_member_code_meta<T>::value>::type* = nullptr >
auto code_wrapper (T x) {
    return x.get_cl_code();
};
