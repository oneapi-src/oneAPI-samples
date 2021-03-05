/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 * Copyright (c) 2019-2021 Intel Corporation (oneAPI modifications)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <CL/sycl.hpp>

// using MACRO to allocate memory inside kernel
#define NUM_3D_BOX_CORNERS_MACRO 8
#define NUM_2D_BOX_CORNERS_MACRO 4

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// Performs atomic fetch and add operation using SYCL
// Calls add on SYCL atomic object
inline int AtomicFetchAdd(int *addr, int operand) {
  sycl::atomic<int, sycl::access::address_space::global_space> obj(
      (sycl::multi_ptr<int, sycl::access::address_space::global_space>(addr)));
  return sycl::atomic_fetch_add(obj, operand, sycl::memory_order::relaxed);
}

// Returns the next power of 2 for a given number
uint32_t inline NextPower(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
