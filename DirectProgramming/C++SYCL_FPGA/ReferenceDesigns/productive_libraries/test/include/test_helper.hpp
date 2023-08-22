/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef _TEST_HELPER_HPP_
#define _TEST_HELPER_HPP_

#include <iostream>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl.hpp"

#ifdef _WIN64
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#define test_failed  0
#define test_passed  1
#define test_skipped 2

#define EXPECT_TRUEORSKIP(a)             \
    do {                                 \
        int res = a;                     \
        if (res == test_skipped)         \
            GTEST_SKIP();                \
        else                             \
            EXPECT_EQ(res, test_passed); \
    } while (0);

inline void print_error_code(sycl::exception const& e) {
#ifdef __HIPSYCL__
    std::cout << "Backend status: " << e.code() << std::endl;
#else
    std::cout << "OpenCL status: " << e.code() << std::endl;
#endif
}


class DeviceNamePrint {
public:
    std::string operator()(testing::TestParamInfo<sycl::device *> dev) const {
        std::string dev_name = dev.param->get_info<sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        if (dev_name.size() == 0)
            dev_name = dev_name.append("_");
        return dev_name;
    }
};

class LayoutDeviceNamePrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<sycl::device *, oneapi::mkl::layout>> dev) const {
        std::string layout_name = std::get<1>(dev.param) == oneapi::mkl::layout::col_major
                                      ? "Column_Major"
                                      : "Row_Major";
        std::string dev_name = std::get<0>(dev.param)->get_info<sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        std::string info_name = (layout_name.append("_")).append(dev_name);
        return info_name;
    }
};

/* to accommodate Windows and Linux differences between alligned_alloc and
   _aligned_malloc calls use oneapi::mkl::aligned_alloc and oneapi::mkl::aligned_free instead */
namespace oneapi {
namespace mkl {

static inline void *aligned_alloc(size_t align, size_t size) {
#ifdef _WIN64
    return ::_aligned_malloc(size, align);
#else
    return ::aligned_alloc(align, size);
#endif
}

static inline void aligned_free(void *p) {
#ifdef _WIN64
    ::_aligned_free(p);
#else
    ::free(p);
#endif
}

/* Support for Unified Shared Memory allocations for different backends */
static inline void *malloc_shared(size_t align, size_t size, sycl::device dev, sycl::context ctx) {
    (void)align;
#ifdef _WIN64
    return sycl::malloc_shared(size, dev, ctx);
#else
#if defined(ENABLE_CUBLAS_BACKEND) || defined(ENABLE_ROCBLAS_BACKEND)
    return sycl::aligned_alloc_shared(align, size, dev, ctx);
#endif
#if !defined(ENABLE_CUBLAS_BACKEND) && !defined(ENABLE_ROCBLAS_BACKEND)
    return sycl::malloc_shared(size, dev, ctx);
#endif
#endif
}

static inline void *malloc_device(size_t align, size_t size, sycl::device dev, sycl::context ctx) {
#ifdef _WIN64
    return sycl::malloc_device(size, dev, ctx);
#else
#if defined(ENABLE_CUBLAS_BACKEND) || defined(ENABLE_ROCBLAS_BACKEND)
    return sycl::aligned_alloc_device(align, size, dev, ctx);
#endif
#if !defined(ENABLE_CUBLAS_BACKEND) && !defined(ENABLE_ROCBLAS_BACKEND)
    return sycl::malloc_device(size, dev, ctx);
#endif
#endif
}

static inline void free_shared(void *p, sycl::context ctx) {
    sycl::free(p, ctx);
}

static inline void free_usm(void *p, sycl::context ctx) {
    sycl::free(p, ctx);
}

} // namespace mkl
} // namespace oneapi

#endif // _TEST_HELPER_HPP_
