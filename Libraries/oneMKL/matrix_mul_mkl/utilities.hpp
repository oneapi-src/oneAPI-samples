//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <algorithm>

template <typename T> const char *type_string()     { return "unknown type"; }
template <> const char *type_string<sycl::half>()   { return "half precision"; }
template <> const char *type_string<float>()        { return "single precision"; }
template <> const char *type_string<double>()       { return "double precision"; }

/* Choose inter-column padding for optimal performance */
template <typename T>
int nice_ld(int x)
{
    x = std::max(x, 1);
    x *= sizeof(T);
    x = (x + 511) & ~511;
    x += 256;
    x /= sizeof(T);
    return x;
}

/* Random number generation helpers */
template <typename T>
void generate_random_data(size_t elems, T *v)
{
#pragma omp parallel for
    for (size_t i = 0; i < elems; i++)
        v[i] = double(std::rand()) / RAND_MAX;
}

template <typename T>
void replicate_data(sycl::queue &Q, T *dst, size_t dst_elems, const T *src, size_t src_elems)
{
    while (dst_elems > 0) {
        auto copy_elems = std::min(dst_elems, src_elems);
        Q.copy(src,  dst, copy_elems);
        dst += copy_elems;
        dst_elems -= copy_elems;
    }
    Q.wait();
}

