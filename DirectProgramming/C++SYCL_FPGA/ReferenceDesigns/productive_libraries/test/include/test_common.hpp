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

#ifndef _TEST_COMMON_HPP__
#define _TEST_COMMON_HPP__

#include <algorithm>

#include <complex>
#include <stdexcept>
#include <type_traits>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#define MAX_NUM_PRINT 20

namespace std {
static sycl::half abs(sycl::half v) {
    if (v < sycl::half(0))
        return -v;
    else
        return v;
}
} // namespace std

// Complex helpers.
template <typename T>
struct complex_info {
    using real_type = T;
    static const bool is_complex = false;
};

template <typename T>
struct complex_info<std::complex<T>> {
    using real_type = T;
    static const bool is_complex = true;
};

template <typename T>
constexpr bool is_complex() {
    return complex_info<T>::is_complex;
}
template <typename T>
constexpr int num_components() {
    return is_complex<T>() ? 2 : 1;
}

// Matrix helpers.
template <typename T>
constexpr T inner_dimension(oneapi::mkl::transpose trans, T m, T n) {
    return (trans == oneapi::mkl::transpose::nontrans) ? m : n;
}
template <typename T>
constexpr T outer_dimension(oneapi::mkl::transpose trans, T m, T n) {
    return (trans == oneapi::mkl::transpose::nontrans) ? n : m;
}
template <typename T>
constexpr T matrix_size(oneapi::mkl::transpose trans, T m, T n, T ldm) {
    return outer_dimension(trans, m, n) * ldm;
}
template <typename T>
constexpr T matrix_size(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, T m, T n, T ldm) {
    return (layout == oneapi::mkl::layout::col_major) ? outer_dimension(trans, m, n) * ldm
                                                         : inner_dimension(trans, m, n) * ldm;
}

// SYCL buffer creation helper.
template <typename vec>
sycl::buffer<typename vec::value_type, 1> make_buffer(const vec &v) {
    sycl::buffer<typename vec::value_type, 1> buf(v.data(), sycl::range<1>(v.size()));
    return buf;
}

// Reference helpers.
template <typename T>
struct ref_type_info {
    using type = T;
};
template <>
struct ref_type_info<std::complex<float>> {
    using type = std::complex<float>;
};
template <>
struct ref_type_info<std::complex<double>> {
    using type = std::complex<double>;
};
template <>
struct ref_type_info<int8_t> {
    using type = int8_t;
};
template <>
struct ref_type_info<uint8_t> {
    using type = uint8_t;
};
template <>
struct ref_type_info<int32_t> {
    using type = int32_t;
};

// Random initialization.
template <typename fp>
static fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}
template <typename fp>
static std::complex<fp> rand_complex_scalar() {
    return std::complex<fp>(rand_scalar<fp>(), rand_scalar<fp>());
}
template <>
std::complex<float> rand_scalar() {
    return rand_complex_scalar<float>();
}
template <>
std::complex<double> rand_scalar() {
    return rand_complex_scalar<double>();
}
template <>
int8_t rand_scalar() {
    return std::rand() % 254 - 127;
}
template <>
int32_t rand_scalar() {
    return std::rand() % 256 - 128;
}
template <>
uint8_t rand_scalar() {
    return std::rand() % 128;
}

template <>
sycl::half rand_scalar() {
    return sycl::half(std::rand() % 32000) / sycl::half(32000) - sycl::half(0.5);
}

template <typename fp>
static fp rand_scalar(int mag) {
    fp tmp = fp(mag) + fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
    if (std::rand() % 2)
        return tmp;
    else
        return -tmp;
}
template <typename fp>
static std::complex<fp> rand_complex_scalar(int mag) {
    return std::complex<fp>(rand_scalar<fp>(mag), rand_scalar<fp>(mag));
}
template <>
std::complex<float> rand_scalar(int mag) {
    return rand_complex_scalar<float>(mag);
}
template <>
std::complex<double> rand_scalar(int mag) {
    return rand_complex_scalar<double>(mag);
}

template <typename fp>
void rand_vector(fp *v, int n, int inc) {
    int abs_inc = std::abs(inc);
    for (int i = 0; i < n; i++)
        v[i * abs_inc] = rand_scalar<fp>();
}

template <typename vec>
void rand_vector(vec &v, int n, int inc) {
    using fp = typename vec::value_type;
    int abs_inc = std::abs(inc);

    v.resize(n * abs_inc);

    for (int i = 0; i < n; i++)
        v[i * abs_inc] = rand_scalar<fp>();
}

template <typename fp>
oneapi::mkl::transpose rand_trans() {
    std::int64_t tmp;
    oneapi::mkl::transpose trans;
    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        trans = (oneapi::mkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            trans = oneapi::mkl::transpose::conjtrans;
        else
            trans = (oneapi::mkl::transpose)tmp;
    }
    return trans;
}

template <typename vec>
void print_matrix(vec &M, oneapi::mkl::transpose trans, int m, int n, int ld, char *name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (trans == oneapi::mkl::transpose::nontrans)
                std::cout << (double)M[i + j * ld] << " ";
            else
                std::cout << (double)M[j + i * ld] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename fp>
void copy_vector(fp *src, int n, int inc, fp *dest) {
    int abs_inc = std::abs(inc);
    for (int i = 0; i < n; i++)
        dest[i * abs_inc] = src[i * abs_inc];
}

template <typename vec_src, typename vec_dest>
void copy_matrix(vec_src &src, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m,
                 int n, int ld, vec_dest &dest) {
    using T_data = typename vec_dest::value_type;
    dest.resize(matrix_size(layout, trans, m, n, ld));
    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                dest[i + j * ld] = (T_data)src[i + j * ld];
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                dest[j + i * ld] = (T_data)src[j + i * ld];
    }
}

template <typename fp>
void copy_matrix(fp *src, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m, int n,
                 int ld, fp *dest) {
    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                dest[i + j * ld] = (fp)src[i + j * ld];
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                dest[j + i * ld] = (fp)src[j + i * ld];
    }
}

template <typename vec>
void rand_matrix(vec &M, oneapi::mkl::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    M.resize(matrix_size(trans, m, n, ld));

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

template <typename vec>
void rand_matrix(vec &M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m, int n,
                 int ld) {
    using fp = typename vec::value_type;

    M.resize(matrix_size(layout, trans, m, n, ld));

    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

template <typename fp>
void rand_matrix(fp *M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m, int n,
                 int ld) {
    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

// Create a complex matrix. If for debugging, every element's data reflects its location: element (i, j) equals (0.1*i, 0.1*j)
template <typename vec>
void rand_complex_matrix(vec &M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m, int n,
                         int ld, bool for_debugging = false) {
    using fp = typename vec::value_type;
    if (!(std::is_same_v<std::complex<float>, fp>) && !(std::is_same_v<std::complex<double>, fp>)) {
        throw std::invalid_argument("For debugging, currently data type must be std::complex<float> or std::complex<double>");
    }
    if (!for_debugging) {
        rand_matrix(M, layout, trans, m, n, ld);
        return;
    }

    M.resize(matrix_size(layout, trans, m, n, ld));

    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = (fp){(float)(0.1 * j), (float)(0.1 * i)};
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = (fp){(float)(0.1 * i), (float)(0.1 * j)};
    }
}

// Create a Hermitian matrix. If for debugging, every element's data reflects its location: element (i, j) in the upper triangle equals (0.i, 0.j),
// while the corresponding element (j, i) that is in the lower triangle equals (0.i, -0.j)
template <typename vec>
void rand_hermitian_matrix(vec &M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int n, int ld, bool for_debugging) {
    using fp = typename vec::value_type;
    if (!(std::is_same_v<std::complex<float>, fp>) && !(std::is_same_v<std::complex<double>, fp>)) {
        throw std::invalid_argument("Data type must be std::complex<float> or std::complex<double>");
    }

    if (!for_debugging) {
        rand_matrix(M, layout, trans, n, n, ld);
        // Set the diagonal data as real
        for (int i = 0; i < n; i++) {
            M[i + i * ld] = M[i + i * ld].real();
        }
        return;
    }

    M.resize(matrix_size(layout, trans, n, n, ld));

    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (j == i)  {
                    M[i + i * ld] = (fp){(float)(0.1 * i), 0.0};
                }
                else {
                    M[j + i * ld] = (fp){(float)(0.1 * i), (float)(0.1 * j)};  // Upper triangle element (i, j)
                    M[i + j * ld] = (fp){(float)(0.1 * i), (float)(-0.1 * j)}; // Lower triangle element (j, i)
                }
            }
        }
    }
    else {
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (j == i) {
                    M[i + i * ld] = (fp){(float)(0.1 * i), 0.0};
                }
                else {
                    M[j + i * ld] = (fp){(float)(0.1 * i), (float)(-0.1 * j)}; // Lower triangle element (j, i)
                    M[i + j * ld] = (fp){(float)(0.1 * i), (float)(0.1 * j)};  // Upper triangle element (i, j)
                }
            }
        }
    }
}

template <typename vec>
void rand_trsm_matrix(vec &M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m,
                      int n, int ld) {
    using fp = typename vec::value_type;

    M.resize(matrix_size(layout, trans, m, n, ld));

    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                if (i == j)
                    M[i + j * ld] = rand_scalar<fp>(10);
                else
                    M[i + j * ld] = rand_scalar<fp>();
            }
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (i == j)
                    M[j + i * ld] = rand_scalar<fp>(10);
                else
                    M[j + i * ld] = rand_scalar<fp>();
            }
    }
}

template <typename fp>
void rand_trsm_matrix(fp *M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m, int n,
                      int ld) {
    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::col_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                if (i == j)
                    M[i + j * ld] = rand_scalar<fp>(10);
                else
                    M[i + j * ld] = rand_scalar<fp>();
            }
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (i == j)
                    M[j + i * ld] = rand_scalar<fp>(10);
                else
                    M[j + i * ld] = rand_scalar<fp>();
            }
    }
}

template <typename vec>
void rand_tpsv_matrix(vec &M, oneapi::mkl::layout layout, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, int m) {
    using fp = typename vec::value_type;
    std::vector<fp> tmp;
    int start, end, i, j, k = 0;

    rand_trsm_matrix(tmp, layout, trans, m, m, m);
    M.resize((m * (m + 1)) / 2);

    for (j = 0; j < m; j++) {
        if (layout == oneapi::mkl::layout::col_major) {
            start = (upper_lower == oneapi::mkl::uplo::U) ? 0 : j;
            end = (upper_lower == oneapi::mkl::uplo::U) ? j : m - 1;
        }
        else {
            start = (upper_lower == oneapi::mkl::uplo::U) ? j : 0;
            end = (upper_lower == oneapi::mkl::uplo::U) ? m - 1 : j;
        }
        for (i = start; i <= end; i++) {
            M[k] = tmp[i + j * m];
            k++;
        }
    }
}

template <typename vec>
void rand_tbsv_matrix(vec &M, oneapi::mkl::layout layout, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, int m, int k, int ld) {
    using fp = typename vec::value_type;
    std::vector<fp> tmp;
    int i, j, n;

    rand_trsm_matrix(tmp, layout, trans, m, m, ld);
    M.resize(matrix_size(layout, trans, m, m, ld));

    if (((layout == oneapi::mkl::layout::col_major) && (upper_lower == oneapi::mkl::uplo::U)) ||
        ((layout == oneapi::mkl::layout::row_major) && (upper_lower == oneapi::mkl::uplo::L))) {
        for (j = 0; j < m; j++) {
            n = k - j;
            for (i = std::max(0, j - k); i <= j; i++) {
                M[(n + i) + j * ld] = tmp[i + j * ld];
            }
        }
    }
    else {
        for (j = 0; j < m; j++) {
            n = -j;
            for (i = j; i < std::min(m, j + k + 1); i++) {
                M[(n + i) + j * ld] = tmp[i + j * ld];
            }
        }
    }
}

// Correctness checking.
template <typename fp>
typename std::enable_if<!std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                              int error_mag) {
    using fp_real = typename complex_info<fp>::real_type;
    fp_real bound = (error_mag * num_components<fp>() * std::numeric_limits<fp_real>::epsilon());

    bool ok;

    fp_real aerr = std::abs(x - x_ref);
    fp_real rerr = aerr / std::abs(x_ref);
    ok = (rerr <= bound) || (aerr <= bound);
    if (!ok)
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
    return ok;
}

template <typename fp>
typename std::enable_if<std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                             int error_mag) {
    return (x == x_ref);
}

template <typename fp>
bool check_equal_ptr(sycl::queue queue, fp *x, fp x_ref, int error_mag) {
    fp x_host;
    queue.memcpy(&x_host, x, sizeof(fp)).wait();
    return check_equal(x_host, x_ref, error_mag);
}

template <typename fp>
bool check_equal_trsm(fp x, fp x_ref, int error_mag) {
    using fp_real = typename complex_info<fp>::real_type;
    fp_real bound = std::max(fp_real(5e-5), (error_mag * num_components<fp>() *
                                             std::numeric_limits<fp_real>::epsilon()));
    fp zero = fp(0);
    bool ok, check_rerr = (x_ref != zero);

    fp_real aerr = std::abs(x - x_ref);
    fp_real rerr = check_rerr ? aerr / std::abs(x_ref) : 0.0;
    ok = check_rerr ? ((rerr <= bound) || (aerr <= bound)) : (aerr <= bound);
    if (!ok)
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
    return ok;
}

template <typename fp>
bool check_equal(fp x, fp x_ref, int error_mag, std::ostream &out) {
    bool good = check_equal(x, x_ref, error_mag);

    if (!good) {
        out << "Difference in result: T2SP " << x << " vs. DPC++ " << x_ref << std::endl;
    }
    return good;
}

template <typename fp>
bool check_equal_ptr(sycl::queue queue, fp *x, fp x_ref, int error_mag, std::ostream &out) {
    fp x_host;
    queue.memcpy(&x_host, x, sizeof(fp)).wait();
    return check_equal(x_host, x_ref, error_mag, out);
}

template <typename fp>
bool check_equal_vector(const fp *v, const fp *v_ref, int n, int inc, int error_mag,
                        std::ostream &out) {
    int abs_inc = std::abs(inc), count = 0;
    bool good = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": T2SP " << v[i * abs_inc]
                      << " vs. DPC++ " << v_ref[i * abs_inc] << std::endl;
            good = false;
            count++;
            if (count > MAX_NUM_PRINT)
                return good;
        }
    }

    return good;
}

template <typename vec1, typename vec2>
bool check_equal_vector(vec1 &v, vec2 &v_ref, int n, int inc, int error_mag, std::ostream &out) {
    int abs_inc = std::abs(inc), count = 0;
    bool good = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": T2SP " << v[i * abs_inc]
                      << " vs. DPC++ " << v_ref[i * abs_inc] << std::endl;
            good = false;
            count++;
            if (count > MAX_NUM_PRINT)
                return good;
        }
    }

    return good;
}

template <typename vec1, typename vec2>
bool check_equal_trsv_vector(vec1 &v, vec2 &v_ref, int n, int inc, int error_mag,
                             std::ostream &out) {
    int abs_inc = std::abs(inc), count = 0;
    bool good = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal_trsm(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": T2SP " << v[i * abs_inc]
                      << " vs. DPC++ " << v_ref[i * abs_inc] << std::endl;
            good = false;
            count++;
            if (count > MAX_NUM_PRINT)
                return good;
        }
    }

    return good;
}

template <typename acc1, typename acc2>
bool check_equal_matrix(acc1 &M, acc2 &M_ref, oneapi::mkl::layout layout, int m, int n, int ld,
                        int error_mag, std::ostream &out) {
    bool good = true;
    int idx, count = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            idx = (layout == oneapi::mkl::layout::col_major) ? i + j * ld : j + i * ld;
            if (!check_equal(M[idx], M_ref[idx], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): T2SP " << M[idx]
                    << " vs. DPC++ " << M_ref[idx] << std::endl;
                good = false;
                count++;
                if (count > MAX_NUM_PRINT)
                    return good;
            }
        }
    }

    return good;
}

template <typename fp>
bool check_equal_matrix(const fp *M, const fp *M_ref, oneapi::mkl::layout layout, int m, int n,
                        int ld, int error_mag, std::ostream &out) {
    bool good = true;
    int idx, count = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            idx = (layout == oneapi::mkl::layout::col_major) ? i + j * ld : j + i * ld;
            if (!check_equal(M[idx], M_ref[idx], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): T2SP " << M[idx]
                    << " vs. DPC++ " << M_ref[idx] << std::endl;
                good = false;
                count++;
                if (count > MAX_NUM_PRINT)
                    return good;
            }
        }
    }

    return good;
}

template <typename acc1, typename acc2>
bool check_equal_matrix(acc1 &M, acc2 &M_ref, oneapi::mkl::layout layout,
                        oneapi::mkl::uplo upper_lower, int m, int n, int ld, int error_mag,
                        std::ostream &out) {
    bool good = true;
    int idx, count = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            idx = (layout == oneapi::mkl::layout::col_major) ? i + j * ld : j + i * ld;
            if (((upper_lower == oneapi::mkl::uplo::upper) && (j >= i)) ||
                ((upper_lower == oneapi::mkl::uplo::lower) && (j <= i))) {
                if (!check_equal(M[idx], M_ref[idx], error_mag)) {
                    out << "Difference in entry (" << i << ',' << j << "): T2SP " << M[idx]
                        << " vs. DPC++ " << M_ref[idx] << std::endl;
                    good = false;
                    count++;
                    if (count > MAX_NUM_PRINT)
                        return good;
                }
            }
        }
    }

    return good;
}

template <typename acc1, typename acc2>
bool check_equal_trsm_matrix(acc1 &M, acc2 &M_ref, oneapi::mkl::layout layout, int m, int n, int ld,
                             int error_mag, std::ostream &out) {
    bool good = true;
    int idx, count = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            idx = (layout == oneapi::mkl::layout::col_major) ? i + j * ld : j + i * ld;
            if (!check_equal_trsm(M[idx], M_ref[idx], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): T2SP " << M[idx]
                    << " vs. DPC++ " << M_ref[idx] << std::endl;
                good = false;
                count++;
                if (count > MAX_NUM_PRINT)
                    return good;
            }
        }
    }

    return good;
}

#endif /* header guard */
