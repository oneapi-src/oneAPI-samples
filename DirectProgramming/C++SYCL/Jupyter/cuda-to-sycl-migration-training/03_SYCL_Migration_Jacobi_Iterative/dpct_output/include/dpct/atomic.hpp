//==---- atomic.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_ATOMIC_HPP__
#define __DPCT_ATOMIC_HPP__

#include <sycl/sycl.hpp>


namespace dpct {

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_add(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_add(T *addr, T operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                    "atomics are: sycl::memory_order::relaxed, "
                    "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          typename T1, typename T2>
inline T1 atomic_fetch_add(T1 *addr, T2 operand,
                           sycl::memory_order memoryOrder) {
  atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
}

} // namespace dpct

#endif // __DPCT_ATOMIC_HPP__
