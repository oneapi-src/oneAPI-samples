/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef __ALLOCATOR_HELPER_HPP
#define __ALLOCATOR_HELPER_HPP

#include <stdlib.h>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "test_helper.hpp"

template <typename T, int align>
struct allocator_helper {
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef void* void_pointer;
    typedef const void* const_void_pointer;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    template <typename U>
    struct rebind {
        typedef allocator_helper<U, align> other;
    };

    allocator_helper() noexcept {}
    template <typename U, int align2>
    allocator_helper(allocator_helper<U, align2>& other) noexcept {}
    template <typename U, int align2>
    allocator_helper(allocator_helper<U, align2>&& other) noexcept {}

    T* allocate(size_t n) {
        void* mem = oneapi::mkl::aligned_alloc(align, n * sizeof(T));
        if (!mem)
            throw std::bad_alloc();

        return static_cast<T*>(mem);
    }

    void deallocate(T* p, size_t n) noexcept {
        oneapi::mkl::aligned_free(p);
    }

    constexpr size_t max_size() const noexcept {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    template <typename U, int align2>
    constexpr bool operator==(const allocator_helper<U, align2>) const noexcept {
        return true;
    }
    template <typename U, int align2>
    constexpr bool operator!=(const allocator_helper<U, align2>) const noexcept {
        return false;
    }

    typedef std::true_type is_always_equal;
};

#endif
