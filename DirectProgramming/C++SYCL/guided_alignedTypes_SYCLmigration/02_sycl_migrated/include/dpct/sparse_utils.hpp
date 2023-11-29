//==---- sparse_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SPARSE_UTILS_HPP__
#define __DPCT_SPARSE_UTILS_HPP__

#include "lib_common_utils.hpp"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace sparse {
/// Describes properties of a sparse matrix.
/// The properties are matrix type, diag, uplo and index base.
class matrix_info {
public:
  /// Matrix types are:
  /// ge: General matrix
  /// sy: Symmetric matrix
  /// he: Hermitian matrix
  /// tr: Triangular matrix
  enum class matrix_type : int { ge = 0, sy, he, tr };

  auto get_matrix_type() const { return _matrix_type; }
  auto get_diag() const { return _diag; }
  auto get_uplo() const { return _uplo; }
  auto get_index_base() const { return _index_base; }
  void set_matrix_type(matrix_type mt) { _matrix_type = mt; }
  void set_diag(oneapi::mkl::diag d) { _diag = d; }
  void set_uplo(oneapi::mkl::uplo u) { _uplo = u; }
  void set_index_base(oneapi::mkl::index_base ib) { _index_base = ib; }

private:
  matrix_type _matrix_type = matrix_type::ge;
  oneapi::mkl::diag _diag = oneapi::mkl::diag::nonunit;
  oneapi::mkl::uplo _uplo = oneapi::mkl::uplo::upper;
  oneapi::mkl::index_base _index_base = oneapi::mkl::index_base::zero;
};

/// Computes a CSR format sparse matrix-dense vector product.
/// y = alpha * op(A) * x + beta * y
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] x Data of the vector x.
/// \param [in] beta Scaling factor for the vector x.
/// \param [in, out] y Data of the vector y.
template <typename T>
void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans, int num_rows,
           int num_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *x, const T *beta,
           T *y) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename dpct::DataType<T>::T2;
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
      new oneapi::mkl::sparse::matrix_handle_t;
  oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
  auto data_row_ptr = dpct::detail::get_memory<int>(row_ptr);
  auto data_col_ind = dpct::detail::get_memory<int>(col_ind);
  auto data_val = dpct::detail::get_memory<Ty>(val);
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, num_rows,
                                    num_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_x = dpct::detail::get_memory<Ty>(x);
  auto data_y = dpct::detail::get_memory<Ty>(y);
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::optimize_gemv(queue, trans, *sparse_matrix_handle);
    oneapi::mkl::sparse::gemv(queue, trans, alpha_value, *sparse_matrix_handle,
                              data_x, beta_value, data_y);
    break;
  }
  case matrix_info::matrix_type::sy: {
    oneapi::mkl::sparse::symv(queue, info->get_uplo(), alpha_value,
                              *sparse_matrix_handle, data_x, beta_value,
                              data_y);
    break;
  }
  case matrix_info::matrix_type::tr: {
    oneapi::mkl::sparse::optimize_trmv(queue, info->get_uplo(), trans,
                                       info->get_diag(), *sparse_matrix_handle);
    oneapi::mkl::sparse::trmv(queue, info->get_uplo(), trans, info->get_diag(),
                              alpha_value, *sparse_matrix_handle, data_x,
                              beta_value, data_y);
    break;
  }
  default:
    throw std::runtime_error(
        "the spmv does not support matrix_info::matrix_type::he");
  }

  sycl::event e =
      oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
#endif
}

/// Computes a CSR format sparse matrix-dense vector product.
/// y = alpha * op(A) * x + beta * y
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] alpha_type Data type of \p alpha .
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] val_type Data type of \p val .
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] x Data of the vector x.
/// \param [in] x_type Data type of \p x .
/// \param [in] beta Scaling factor for the vector x.
/// \param [in] beta_type Data type of \p beta .
/// \param [in, out] y Data of the vector y.
/// \param [in] y_type Data type of \p y .
inline void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans,
                  int num_rows, int num_cols, const void *alpha,
                  library_data_t alpha_type,
                  const std::shared_ptr<matrix_info> info, const void *val,
                  library_data_t val_type, const int *row_ptr,
                  const int *col_ind, const void *x, library_data_t x_type,
                  const void *beta, library_data_t beta_type, void *y,
                  library_data_t y_type) {
  std::uint64_t key = dpct::detail::get_type_combination_id(
      alpha_type, val_type, x_type, beta_type, y_type);
  switch (key) {
  case dpct::detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float): {
    csrmv<float>(queue, trans, num_rows, num_cols,
                 static_cast<const float *>(alpha), info,
                 static_cast<const float *>(val), row_ptr, col_ind,
                 static_cast<const float *>(x),
                 static_cast<const float *>(beta), static_cast<float *>(y));
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double): {
    csrmv<double>(queue, trans, num_rows, num_cols,
                  static_cast<const double *>(alpha), info,
                  static_cast<const double *>(val), row_ptr, col_ind,
                  static_cast<const double *>(x),
                  static_cast<const double *>(beta), static_cast<double *>(y));
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float): {
    csrmv<std::complex<float>>(
        queue, trans, num_rows, num_cols,
        static_cast<const std::complex<float> *>(alpha), info,
        static_cast<const std::complex<float> *>(val), row_ptr, col_ind,
        static_cast<const std::complex<float> *>(x),
        static_cast<const std::complex<float> *>(beta),
        static_cast<std::complex<float> *>(y));
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double): {
    csrmv<std::complex<double>>(
        queue, trans, num_rows, num_cols,
        static_cast<const std::complex<double> *>(alpha), info,
        static_cast<const std::complex<double> *>(val), row_ptr, col_ind,
        static_cast<const std::complex<double> *>(x),
        static_cast<const std::complex<double> *>(beta),
        static_cast<std::complex<double> *>(y));
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a CSR format sparse matrix-dense matrix product.
/// C = alpha * op(A) * B + beta * C
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] sparse_rows Number of rows of the matrix A.
/// \param [in] dense_cols Number of columns of the matrix B or C.
/// \param [in] sparse_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] b Data of the matrix B.
/// \param [in] ldb Leading dimension of the matrix B.
/// \param [in] beta Scaling factor for the matrix B.
/// \param [in, out] c Data of the matrix C.
/// \param [in] ldc Leading dimension of the matrix C.
template <typename T>
void csrmm(sycl::queue &queue, oneapi::mkl::transpose trans, int sparse_rows,
           int dense_cols, int sparse_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *b, int ldb,
           const T *beta, T *c, int ldc) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename dpct::DataType<T>::T2;
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
      new oneapi::mkl::sparse::matrix_handle_t;
  oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
  auto data_row_ptr = dpct::detail::get_memory<int>(row_ptr);
  auto data_col_ind = dpct::detail::get_memory<int>(col_ind);
  auto data_val = dpct::detail::get_memory<Ty>(val);
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, sparse_rows,
                                    sparse_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_b = dpct::detail::get_memory<Ty>(b);
  auto data_c = dpct::detail::get_memory<Ty>(c);
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, trans,
                              oneapi::mkl::transpose::nontrans, alpha_value,
                              *sparse_matrix_handle, data_b, dense_cols, ldb,
                              beta_value, data_c, ldc);
    break;
  }
  default:
    throw std::runtime_error(
        "the csrmm does not support matrix_info::matrix_type::sy, "
        "matrix_info::matrix_type::tr and matrix_info::matrix_type::he");
  }

  sycl::event e =
      oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
#endif
}

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Saving the optimization information for solving a system of linear
/// equations.
class optimize_info {
public:
  /// Constructor
  optimize_info() { oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle); }
  /// Destructor
  ~optimize_info() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }
  /// Add dependency for the destructor.
  /// \param [in] e The event which the destructor depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }

private:
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
};
#endif

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Performs internal optimizations for solving a system of linear equations for
/// a CSR format sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the sparse matrix.
/// \param [in] row_col Number of rows of the sparse matrix.
/// \param [in] info Matrix info of the sparse matrix.
/// \param [in] val An array containing the non-zero elements of the sparse matrix.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [out] optimize_info The result of the optimizations.
template <typename T>
void optimize_csrsv(sycl::queue &queue, oneapi::mkl::transpose trans,
                    int row_col, const std::shared_ptr<matrix_info> info,
                    const T *val, const int *row_ptr, const int *col_ind,
                    std::shared_ptr<optimize_info> optimize_info) {
  using Ty = typename dpct::DataType<T>::T2;
  auto data_row_ptr = dpct::detail::get_memory<int>(row_ptr);
  auto data_col_ind = dpct::detail::get_memory<int>(col_ind);
  auto data_val = dpct::detail::get_memory<Ty>(val);
  oneapi::mkl::sparse::set_csr_data(queue, optimize_info->get_matrix_handle(),
                                    row_col, row_col, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);
  if (info->get_matrix_type() != matrix_info::matrix_type::tr)
    return;
#ifndef DPCT_USM_LEVEL_NONE
  sycl::event e;
  e =
#endif
      oneapi::mkl::sparse::optimize_trsv(queue, info->get_uplo(), trans,
                                         info->get_diag(),
                                         optimize_info->get_matrix_handle());
#ifndef DPCT_USM_LEVEL_NONE
  optimize_info->add_dependency(e);
#endif
}
#endif

class sparse_matrix_desc;

using sparse_matrix_desc_t = std::shared_ptr<sparse_matrix_desc>;

/// Structure for describe a dense vector
class dense_vector_desc {
public:
  dense_vector_desc(std::int64_t ele_num, void *value,
                    library_data_t value_type)
      : _ele_num(ele_num), _value(value), _value_type(value_type) {}
  void get_desc(std::int64_t *ele_num, const void **value,
                library_data_t *value_type) const noexcept {
    *ele_num = _ele_num;
    *value = _value;
    *value_type = _value_type;
  }
  void get_desc(std::int64_t *ele_num, void **value,
                library_data_t *value_type) const noexcept {
    get_desc(ele_num, const_cast<const void **>(value), value_type);
  }
  void *get_value() const noexcept { return _value; }
  void set_value(void *value) { _value = value; }
  library_data_t get_value_type() const noexcept { return _value_type; }
  std::int64_t get_ele_num() const noexcept { return _ele_num; }

private:
  std::int64_t _ele_num;
  void *_value;
  library_data_t _value_type;
};

/// Structure for describe a dense matrix
class dense_matrix_desc {
public:
  dense_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                    std::int64_t leading_dim, void *value,
                    library_data_t value_type, oneapi::mkl::layout layout)
      : _row_num(row_num), _col_num(col_num), _leading_dim(leading_dim),
        _value(value), _value_type(value_type), _layout(layout) {}
  void get_desc(std::int64_t *row_num, std::int64_t *col_num,
                std::int64_t *leading_dim, void **value,
                library_data_t *value_type,
                oneapi::mkl::layout *layout) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *leading_dim = _leading_dim;
    *value = _value;
    *value_type = _value_type;
    *layout = _layout;
  }
  void *get_value() const noexcept { return _value; }
  void set_value(void *value) { _value = value; }
  std::int64_t get_col_num() const noexcept { return _col_num; }
  std::int64_t get_leading_dim() const noexcept { return _leading_dim; }
  oneapi::mkl::layout get_layout() const noexcept { return _layout; }

private:
  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _leading_dim;
  void *_value;
  library_data_t _value_type;
  oneapi::mkl::layout _layout;
};

/// Sparse matrix data format
enum matrix_format : int {
  csr = 1,
};

/// Sparse matrix attribute
enum matrix_attribute : int { uplo = 0, diag };

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Structure for describe a sparse matrix
class sparse_matrix_desc {
public:
  /// Constructor
  /// \param [out] desc The descriptor to be created
  /// \param [in] row_num Number of rows of the sparse matrix.
  /// \param [in] col_num Number of colums of the sparse matrix.
  /// \param [in] nnz Non-zero elements in the sparse matrix.
  /// \param [in] row_ptr An array of length \p row_num + 1. If the \p row_ptr is
  /// NULL, the sparse_matrix_desc will allocate internal memory for it. This
  /// internal memory can be gotten from get_shadow_row_ptr().
  /// \param [in] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [in] value An array containing the non-zero elements of the sparse matrix.
  /// \param [in] row_ptr_type Data type of the \p row_ptr .
  /// \param [in] col_ind_type Data type of the \p col_ind .
  /// \param [in] base Indicates how input arrays are indexed.
  /// \param [in] value_type Data type of the \p value .
  /// \param [in] data_format The matrix data format.
  sparse_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                     std::int64_t nnz, void *row_ptr, void *col_ind,
                     void *value, library_data_t row_ptr_type,
                     library_data_t col_ind_type, oneapi::mkl::index_base base,
                     library_data_t value_type, matrix_format data_format)
      : _row_num(row_num), _col_num(col_num), _nnz(nnz), _row_ptr(row_ptr),
        _col_ind(col_ind), _value(value), _row_ptr_type(row_ptr_type),
        _col_ind_type(col_ind_type), _base(base), _value_type(value_type),
        _data_format(data_format) {
    if (_data_format != matrix_format::csr) {
      throw std::runtime_error("the sparse matrix data format is unsupported");
    }
    oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle);
    set_data();
  }
  /// Destructor
  ~sparse_matrix_desc() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
#ifdef DPCT_USM_LEVEL_NONE
    if (_data_row_ptr_32)
      delete _data_row_ptr_32;
    if (_data_row_ptr_64)
      delete _data_row_ptr_64;
    if (_data_col_ind_32)
      delete _data_col_ind_32;
    if (_data_col_ind_64)
      delete _data_col_ind_64;
    if (_data_value_s)
      delete _data_value_s;
    if (_data_value_d)
      delete _data_value_d;
    if (_data_value_c)
      delete _data_value_c;
    if (_data_value_z)
      delete _data_value_z;
#endif
    if (_shadow_row_ptr)
      dpct::dpct_free(_shadow_row_ptr);
  }

  /// Add dependency for the destroy method.
  /// \param [in] e The event which the destroy method depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }
  /// Get the values saved in the descriptor
  /// \param [out] row_num Number of rows of the sparse matrix.
  /// \param [out] col_num Number of colums of the sparse matrix.
  /// \param [out] nnz Non-zero elements in the sparse matrix.
  /// \param [out] row_ptr An array of length \p row_num + 1.
  /// \param [out] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [out] value An array containing the non-zero elements of the sparse matrix.
  /// \param [out] row_ptr_type Data type of the \p row_ptr .
  /// \param [out] col_ind_type Data type of the \p col_ind .
  /// \param [out] base Indicates how input arrays are indexed.
  /// \param [out] value_type Data type of the \p value .
  void get_desc(int64_t *row_num, int64_t *col_num, int64_t *nnz,
                void **row_ptr, void **col_ind, void **value,
                library_data_t *row_ptr_type, library_data_t *col_ind_type,
                oneapi::mkl::index_base *base,
                library_data_t *value_type) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
    *row_ptr = _row_ptr;
    *col_ind = _col_ind;
    *value = _value;
    *row_ptr_type = _row_ptr_type;
    *col_ind_type = _col_ind_type;
    *base = _base;
    *value_type = _value_type;
  }
  /// Get the sparse matrix data format of this descriptor
  /// \param [out] format The matrix data format result
  void get_format(matrix_format *data_format) const noexcept {
    *data_format = _data_format;
  }
  /// Get the index base of this descriptor
  /// \param [out] base The index base result
  void get_base(oneapi::mkl::index_base *base) const noexcept { *base = _base; }
  /// Get the value pointer of this descriptor
  /// \param [out] value The value pointer result
  void get_value(void **value) const noexcept { *value = _value; }
  /// Set the value pointer of this descriptor
  /// \param [in] value The input value pointer
  void set_value(void *value) {
    if (!value) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_value(): The value "
          "pointer is NULL.");
    }
    if (_value && (_value != value)) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_value(): "
          "The _value pointer is not NULL. It cannot be reset.");
    }
    _value = value;
    set_data();
  }
  /// Get the size of the sparse matrix
  /// \param [out] row_num Number of rows of the sparse matrix.
  /// \param [out] col_num Number of colums of the sparse matrix.
  /// \param [out] nnz Non-zero elements in the sparse matrix.
  void get_size(int64_t *row_num, int64_t *col_num,
                int64_t *nnz) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
  }
  /// Set the sparse matrix attribute
  /// \param [in] attribute The attribute type
  /// \param [in] data The attribute value
  /// \param [in] data_size The data size of the attribute value
  void set_attribute(matrix_attribute attribute, const void *data,
                     size_t data_size) {
    if (attribute == matrix_attribute::diag) {
      const oneapi::mkl::diag *diag_ptr =
          reinterpret_cast<const oneapi::mkl::diag *>(data);
      if (*diag_ptr == oneapi::mkl::diag::unit) {
        _diag = oneapi::mkl::diag::unit;
      } else if (*diag_ptr == oneapi::mkl::diag::nonunit) {
        _diag = oneapi::mkl::diag::nonunit;
      } else {
        throw std::runtime_error("unsupported diag value");
      }
    } else if (attribute == matrix_attribute::uplo) {
      const oneapi::mkl::uplo *uplo_ptr =
          reinterpret_cast<const oneapi::mkl::uplo *>(data);
      if (*uplo_ptr == oneapi::mkl::uplo::upper) {
        _uplo = oneapi::mkl::uplo::upper;
      } else if (*uplo_ptr == oneapi::mkl::uplo::lower) {
        _uplo = oneapi::mkl::uplo::lower;
      } else {
        throw std::runtime_error("unsupported uplo value");
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  /// Get the sparse matrix attribute
  /// \param [out] attribute The attribute type
  /// \param [out] data The attribute value
  /// \param [out] data_size The data size of the attribute value
  void get_attribute(matrix_attribute attribute, void *data,
                     size_t data_size) const {
    if (attribute == matrix_attribute::diag) {
      oneapi::mkl::diag *diag_ptr = reinterpret_cast<oneapi::mkl::diag *>(data);
      if (_diag.has_value()) {
        *diag_ptr = _diag.value();
      } else {
        *diag_ptr = oneapi::mkl::diag::nonunit;
      }
    } else if (attribute == matrix_attribute::uplo) {
      oneapi::mkl::uplo *uplo_ptr = reinterpret_cast<oneapi::mkl::uplo *>(data);
      if (_uplo.has_value()) {
        *uplo_ptr = _uplo.value();
      } else {
        *uplo_ptr = oneapi::mkl::uplo::lower;
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  /// Set the pointers for describing the sparse matrix
  /// \param [in] row_ptr An array of length \p row_num + 1.
  /// \param [in] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [in] value An array containing the non-zero elements of the sparse matrix.
  void set_pointers(void *row_ptr, void *col_ind, void *value) {
    if (!row_ptr) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_pointers(): The "
          "row_ptr pointer is NULL.");
    }
    if (!col_ind) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_pointers(): The "
          "col_ind pointer is NULL.");
    }
    if (_row_ptr && (_row_ptr != row_ptr)) {
      throw std::runtime_error("dpct::sparse::sparse_matrix_desc::set_pointers("
                               "): The _row_ptr pointer is "
                               "not NULL. It cannot be reset.");
    }
    if (_col_ind && (_col_ind != col_ind)) {
      throw std::runtime_error("dpct::sparse::sparse_matrix_desc::set_pointers("
                               "): The _col_ind pointer is "
                               "not NULL. It cannot be reset.");
    }
    _row_ptr = row_ptr;
    _col_ind = col_ind;

    // The descriptor will be updated in the set_value function
    set_value(value);
  }

  /// Get the diag attribute
  /// \return diag value
  std::optional<oneapi::mkl::diag> get_diag() const noexcept { return _diag; }
  /// Get the uplo attribute
  /// \return uplo value
  std::optional<oneapi::mkl::uplo> get_uplo() const noexcept { return _uplo; }
  /// Set the number of non-zero elements
  /// \param nnz [in] The number of non-zero elements.
  void set_nnz(std::int64_t nnz) noexcept { _nnz = nnz; }
  /// Get the type of the value pointer.
  /// \return The type of the value pointer.
  library_data_t get_value_type() const noexcept { return _value_type; }
  /// Get the row_ptr.
  /// \return The row_ptr.
  void *get_row_ptr() const noexcept { return _row_ptr; }
  /// If the internal _row_ptr is NULL, the sparse_matrix_desc will allocate
  /// internal memory for it in the constructor. The internal memory can be gotten
  /// from this interface.
  /// \return The shadow row_ptr.
  void *get_shadow_row_ptr() const noexcept { return _shadow_row_ptr; }
  /// Get the type of the col_ind pointer.
  /// \return The type of the col_ind pointer.
  library_data_t get_col_ind_type() const noexcept { return _col_ind_type; }
  /// Get the row_num.
  /// \return The row_num.
  std::int64_t get_row_num() const noexcept { return _row_num; }

private:
#ifdef DPCT_USM_LEVEL_NONE
#define SET_DATA(INDEX_TYPE, INDEX_SUFFIX, VALUE_TYPE, VALUE_SUFFIX)           \
  do {                                                                         \
    void *row_ptr = nullptr;                                                   \
    if (_shadow_row_ptr) {                                                     \
      row_ptr = _shadow_row_ptr;                                               \
    } else if (_row_ptr) {                                                     \
      row_ptr = _row_ptr;                                                      \
    } else {                                                                   \
      row_ptr = _shadow_row_ptr = dpct::dpct_malloc(                           \
          sizeof(INDEX_TYPE) * (_row_num + 1), get_default_queue());           \
    }                                                                          \
    _data_row_ptr##INDEX_SUFFIX =                                              \
        new sycl::buffer<INDEX_TYPE>(dpct::get_buffer<INDEX_TYPE>(row_ptr));   \
    _data_col_ind##INDEX_SUFFIX =                                              \
        new sycl::buffer<INDEX_TYPE>(dpct::get_buffer<INDEX_TYPE>(_col_ind));  \
    _data_value##VALUE_SUFFIX =                                                \
        new sycl::buffer<VALUE_TYPE>(dpct::get_buffer<VALUE_TYPE>(_value));    \
    oneapi::mkl::sparse::set_csr_data(                                         \
        get_default_queue(), _matrix_handle, _row_num, _col_num, _base,        \
        *_data_row_ptr##INDEX_SUFFIX, *_data_col_ind##INDEX_SUFFIX,            \
        *_data_value##VALUE_SUFFIX);                                           \
    get_default_queue().wait();                                                \
  } while (0)
#else
#define SET_DATA(INDEX_TYPE, INDEX_SUFFIX, VALUE_TYPE, VALUE_SUFFIX)           \
  do {                                                                         \
    void *row_ptr = nullptr;                                                   \
    if (_shadow_row_ptr) {                                                     \
      row_ptr = _shadow_row_ptr;                                               \
    } else if (_row_ptr) {                                                     \
      row_ptr = _row_ptr;                                                      \
    } else {                                                                   \
      row_ptr = _shadow_row_ptr = dpct::dpct_malloc(                           \
          sizeof(INDEX_TYPE) * (_row_num + 1), get_default_queue());           \
    }                                                                          \
    oneapi::mkl::sparse::set_csr_data(                                         \
        get_default_queue(), _matrix_handle, _row_num, _col_num, _base,        \
        reinterpret_cast<INDEX_TYPE *>(row_ptr),                               \
        reinterpret_cast<INDEX_TYPE *>(_col_ind),                              \
        reinterpret_cast<VALUE_TYPE *>(_value));                               \
    get_default_queue().wait();                                                \
  } while (0)
#endif

  void set_data() {
    std::uint64_t key = dpct::detail::get_type_combination_id(
        _row_ptr_type, _col_ind_type, _value_type);
    switch (key) {
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::real_float): {
      SET_DATA(std::int32_t, _32, float, _s);
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::real_double): {
      SET_DATA(std::int32_t, _32, double, _d);
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::complex_float): {
      SET_DATA(std::int32_t, _32, std::complex<float>, _c);
      break;
    }
    case dpct::detail::get_type_combination_id(
        library_data_t::real_int32, library_data_t::real_int32,
        library_data_t::complex_double): {
      SET_DATA(std::int32_t, _32, std::complex<double>, _z);
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::real_float): {
      SET_DATA(std::int64_t, _64, float, _s);
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::real_double): {
      SET_DATA(std::int64_t, _64, double, _d);
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::complex_float): {
      SET_DATA(std::int64_t, _64, std::complex<float>, _c);
      break;
    }
    case dpct::detail::get_type_combination_id(
        library_data_t::real_int64, library_data_t::real_int64,
        library_data_t::complex_double): {
      SET_DATA(std::int64_t, _64, std::complex<double>, _z);
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
    }
  }
#undef SET_DATA

  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _nnz;
  void *_row_ptr;
  void *_col_ind;
  void *_value;
  library_data_t _row_ptr_type;
  library_data_t _col_ind_type;
  oneapi::mkl::index_base _base;
  library_data_t _value_type;
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
  matrix_format _data_format;
  std::optional<oneapi::mkl::uplo> _uplo;
  std::optional<oneapi::mkl::diag> _diag;
  void *_shadow_row_ptr = nullptr;
#ifdef DPCT_USM_LEVEL_NONE
  sycl::buffer<std::int32_t> *_data_row_ptr_32 = nullptr;
  sycl::buffer<std::int64_t> *_data_row_ptr_64 = nullptr;
  sycl::buffer<std::int32_t> *_data_col_ind_32 = nullptr;
  sycl::buffer<std::int64_t> *_data_col_ind_64 = nullptr;
  sycl::buffer<float> *_data_value_s = nullptr;
  sycl::buffer<double> *_data_value_d = nullptr;
  sycl::buffer<std::complex<float>> *_data_value_c = nullptr;
  sycl::buffer<std::complex<double>> *_data_value_z = nullptr;
#endif
};

namespace detail {
#ifdef DPCT_USM_LEVEL_NONE
#define SPARSE_CALL(X)                                                         \
  do {                                                                         \
    X;                                                                         \
  } while (0)
#else
#define SPARSE_CALL(X)                                                         \
  do {                                                                         \
    sycl::event e = X;                                                         \
    a->add_dependency(e);                                                      \
  } while (0)
#endif

template <typename Ty>
inline void spmv_impl(sycl::queue queue, oneapi::mkl::transpose trans,
                      const void *alpha, sparse_matrix_desc_t a,
                      std::shared_ptr<dense_vector_desc> x, const void *beta,
                      std::shared_ptr<dense_vector_desc> y) {
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);
  auto data_x = dpct::detail::get_memory<Ty>(x->get_value());
  auto data_y = dpct::detail::get_memory<Ty>(y->get_value());
  if (a->get_diag().has_value() && a->get_uplo().has_value()) {
    oneapi::mkl::sparse::optimize_trmv(queue, a->get_uplo().value(), trans,
                                       a->get_diag().value(),
                                       a->get_matrix_handle());
    SPARSE_CALL(oneapi::mkl::sparse::trmv(
        queue, a->get_uplo().value(), trans, a->get_diag().value(), alpha_value,
        a->get_matrix_handle(), data_x, beta_value, data_y));
  } else {
    oneapi::mkl::sparse::optimize_gemv(queue, trans, a->get_matrix_handle());
    SPARSE_CALL(oneapi::mkl::sparse::gemv(queue, trans, alpha_value,
                                          a->get_matrix_handle(), data_x,
                                          beta_value, data_y));
  }
}

template <typename Ty>
inline void spmm_impl(sycl::queue queue, oneapi::mkl::transpose trans_a,
                      oneapi::mkl::transpose trans_b, const void *alpha,
                      sparse_matrix_desc_t a,
                      std::shared_ptr<dense_matrix_desc> b, const void *beta,
                      std::shared_ptr<dense_matrix_desc> c) {
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);
  auto data_b = dpct::detail::get_memory<Ty>(b->get_value());
  auto data_c = dpct::detail::get_memory<Ty>(c->get_value());
  SPARSE_CALL(oneapi::mkl::sparse::gemm(
      queue, b->get_layout(), trans_a, trans_b, alpha_value,
      a->get_matrix_handle(), data_b, b->get_col_num(), b->get_leading_dim(),
      beta_value, data_c, c->get_leading_dim()));
}
#undef SPARSE_CALL
} // namespace detail

/// Computes a sparse matrix-dense vector product: y = alpha * op(a) * x + beta * y.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans Specifies operation on input matrix.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] x Specifies the dense vector x.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] y Specifies the dense vector y.
/// \param [in] data_type Specifies the data type of \param a, \param x and \param y .
inline void spmv(sycl::queue queue, oneapi::mkl::transpose trans,
                 const void *alpha, sparse_matrix_desc_t a,
                 std::shared_ptr<dense_vector_desc> x, const void *beta,
                 std::shared_ptr<dense_vector_desc> y,
                 library_data_t data_type) {
  switch (data_type) {
  case library_data_t::real_float: {
    detail::spmv_impl<float>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  case library_data_t::real_double: {
    detail::spmv_impl<double>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  case library_data_t::complex_float: {
    detail::spmv_impl<std::complex<float>>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  case library_data_t::complex_double: {
    detail::spmv_impl<std::complex<double>>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a sparse matrix-dense matrix product: c = alpha * op(a) * op(b) + beta * c.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the dense matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the dense matrix c.
/// \param [in] data_type Specifies the data type of \param a, \param b and \param c .
inline void spmm(sycl::queue queue, oneapi::mkl::transpose trans_a,
                 oneapi::mkl::transpose trans_b, const void *alpha,
                 sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
                 const void *beta, std::shared_ptr<dense_matrix_desc> c,
                 library_data_t data_type) {
  if (b->get_layout() != c->get_layout())
    throw std::runtime_error("the layout of b and c are different");

  switch (data_type) {
  case library_data_t::real_float: {
    detail::spmm_impl<float>(queue, trans_a, trans_b, alpha, a, b, beta, c);
    break;
  }
  case library_data_t::real_double: {
    detail::spmm_impl<double>(queue, trans_a, trans_b, alpha, a, b, beta, c);
    break;
  }
  case library_data_t::complex_float: {
    detail::spmm_impl<std::complex<float>>(queue, trans_a, trans_b, alpha, a, b,
                                           beta, c);
    break;
  }
  case library_data_t::complex_double: {
    detail::spmm_impl<std::complex<double>>(queue, trans_a, trans_b, alpha, a,
                                            b, beta, c);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

namespace detail {
template <typename T, bool is_host_memory, typename host_memory_t = void>
struct temp_memory {
  static_assert(!is_host_memory || !std::is_same_v<host_memory_t, void>,
                "host_memory_t cannot be void when the input parameter ptr "
                "points to host memory");
  temp_memory(sycl::queue queue, void *ptr)
      : _queue(queue)
#ifdef DPCT_USM_LEVEL_NONE
        ,
        _buffer(is_host_memory ? sycl::buffer<T, 1>(sycl::range<1>(1))
                               : sycl::buffer<T, 1>(dpct::get_buffer<T>(ptr)))
#endif
  {
    if constexpr (is_host_memory) {
      _original_host_ptr = static_cast<host_memory_t *>(ptr);
#ifdef DPCT_USM_LEVEL_NONE
      auto _buffer_acc = _buffer.get_host_access(sycl::write_only);
      _buffer_acc[0] = static_cast<T>(*_original_host_ptr);
#else
      _memory_ptr = sycl::malloc_host<T>(1, _queue);
      *_memory_ptr = static_cast<T>(*_original_host_ptr);
#endif
    } else {
#ifndef DPCT_USM_LEVEL_NONE
      _memory_ptr = static_cast<T *>(ptr);
#endif
    }
  }

  ~temp_memory() {
    if constexpr (is_host_memory) {
#ifdef DPCT_USM_LEVEL_NONE
      auto _buffer_acc = _buffer.get_host_access(sycl::read_only);
      *_original_host_ptr = static_cast<host_memory_t>(_buffer_acc[0]);
#else
      _queue.wait();
      *_original_host_ptr = *_memory_ptr;
      sycl::free(_memory_ptr, _queue);
#endif
    }
  }
  auto get_memory_ptr() {
#ifdef DPCT_USM_LEVEL_NONE
    return &_buffer;
#else
    return _memory_ptr;
#endif
  }

private:
  sycl::queue _queue;
  host_memory_t *_original_host_ptr = nullptr;
#ifdef DPCT_USM_LEVEL_NONE
  sycl::buffer<T, 1> _buffer;
#else
  T *_memory_ptr;
#endif
};
} // namespace detail

/// Do initial estimation of work and load balancing of computing a sparse
/// matrix-sparse matrix product.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the sparse matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the sparse matrix c.
/// \param [in] matmat_descr Describes the sparse matrix-sparse matrix operation
/// to be executed.
/// \param [in, out] size_temp_buffer Specifies the size of workspace.
/// \param [in] temp_buffer Specifies the memory of the workspace.
inline void
spgemm_work_estimation(sycl::queue queue, oneapi::mkl::transpose trans_a,
                       oneapi::mkl::transpose trans_b, const void *alpha,
                       sparse_matrix_desc_t a, sparse_matrix_desc_t b,
                       const void *beta, sparse_matrix_desc_t c,
                       oneapi::mkl::sparse::matmat_descr_t matmat_descr,
                       size_t *size_temp_buffer, void *temp_buffer) {
  if (temp_buffer) {
    detail::temp_memory<std::int64_t, true, size_t> size_memory(
        queue, size_temp_buffer);
    detail::temp_memory<std::uint8_t, false> work_memory(queue, temp_buffer);
    oneapi::mkl::sparse::matmat(
        queue, a->get_matrix_handle(), b->get_matrix_handle(),
        c->get_matrix_handle(),
        oneapi::mkl::sparse::matmat_request::work_estimation, matmat_descr,
        size_memory.get_memory_ptr(), work_memory.get_memory_ptr()
#ifndef DPCT_USM_LEVEL_NONE
        , {}
#endif
    );
  } else {
    oneapi::mkl::sparse::set_matmat_data(
        matmat_descr, oneapi::mkl::sparse::matrix_view_descr::general, trans_a,
        oneapi::mkl::sparse::matrix_view_descr::general, trans_b,
        oneapi::mkl::sparse::matrix_view_descr::general);
    detail::temp_memory<std::int64_t, true, size_t> size_memory(
        queue, size_temp_buffer);
    oneapi::mkl::sparse::matmat(
        queue, a->get_matrix_handle(), b->get_matrix_handle(),
        c->get_matrix_handle(),
        oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size,
        matmat_descr, size_memory.get_memory_ptr(), nullptr
#ifndef DPCT_USM_LEVEL_NONE
        , {}
#endif
    );
  }
}

/// Do internal products for computing the C matrix of computing a sparse
/// matrix-sparse matrix product.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the sparse matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the sparse matrix c.
/// \param [in] matmat_descr Describes the sparse matrix-sparse matrix operation
/// to be executed.
/// \param [in, out] size_temp_buffer Specifies the size of workspace.
/// \param [in] temp_buffer Specifies the memory of the workspace.
inline void spgemm_compute(sycl::queue queue, oneapi::mkl::transpose trans_a,
                           oneapi::mkl::transpose trans_b, const void *alpha,
                           sparse_matrix_desc_t a, sparse_matrix_desc_t b,
                           const void *beta, sparse_matrix_desc_t c,
                           oneapi::mkl::sparse::matmat_descr_t matmat_descr,
                           size_t *size_temp_buffer, void *temp_buffer) {
  if (temp_buffer) {
    std::int64_t nnz_value = 0;
    {
      detail::temp_memory<std::int64_t, true, size_t> size_memory(
          queue, size_temp_buffer);
      detail::temp_memory<std::uint8_t, false> work_memory(queue, temp_buffer);
      detail::temp_memory<std::int64_t, true, std::int64_t> nnz_memory(
          queue, &nnz_value);
      oneapi::mkl::sparse::matmat(
          queue, a->get_matrix_handle(), b->get_matrix_handle(),
          c->get_matrix_handle(), oneapi::mkl::sparse::matmat_request::compute,
          matmat_descr, size_memory.get_memory_ptr(),
          work_memory.get_memory_ptr()
#ifndef DPCT_USM_LEVEL_NONE
          , {}
#endif
      );
      oneapi::mkl::sparse::matmat(
          queue, a->get_matrix_handle(), b->get_matrix_handle(),
          c->get_matrix_handle(), oneapi::mkl::sparse::matmat_request::get_nnz,
          matmat_descr, nnz_memory.get_memory_ptr(), nullptr
#ifndef DPCT_USM_LEVEL_NONE
          , {}
#endif
      );
    }
    c->set_nnz(nnz_value);
  } else {
    detail::temp_memory<std::int64_t, true, size_t> size_memory(
        queue, size_temp_buffer);
    oneapi::mkl::sparse::matmat(
        queue, a->get_matrix_handle(), b->get_matrix_handle(),
        c->get_matrix_handle(),
        oneapi::mkl::sparse::matmat_request::get_compute_buf_size, matmat_descr,
        size_memory.get_memory_ptr(), nullptr
#ifndef DPCT_USM_LEVEL_NONE
        , {}
#endif
    );
  }
}

/// Do any remaining internal products and accumulation and transfer into final
/// C matrix arrays of computing a sparse matrix-sparse matrix product.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the sparse matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the sparse matrix c.
/// \param [in] matmat_descr Describes the sparse matrix-sparse matrix operation
/// to be executed.
inline void spgemm_finalize(sycl::queue queue, oneapi::mkl::transpose trans_a,
                            oneapi::mkl::transpose trans_b, const void *alpha,
                            sparse_matrix_desc_t a, sparse_matrix_desc_t b,
                            const void *beta, sparse_matrix_desc_t c,
                            oneapi::mkl::sparse::matmat_descr_t matmat_descr) {
  oneapi::mkl::sparse::matmat(queue, a->get_matrix_handle(),
                              b->get_matrix_handle(), c->get_matrix_handle(),
                              oneapi::mkl::sparse::matmat_request::finalize,
                              matmat_descr, nullptr, nullptr
#ifdef DPCT_USM_LEVEL_NONE
  );
#else
  , {}).wait();
#endif
  if (c->get_shadow_row_ptr()) {
    switch (c->get_col_ind_type()) {
    case library_data_t::real_int32: {
      dpct::dpct_memcpy(c->get_row_ptr(), c->get_shadow_row_ptr(),
                        sizeof(std::int32_t) * (c->get_row_num() + 1));
      break;
    }
    case library_data_t::real_int64: {
      dpct::dpct_memcpy(c->get_row_ptr(), c->get_shadow_row_ptr(),
                        sizeof(std::int64_t) * (c->get_row_num() + 1));
      break;
    }
    default:
      throw std::runtime_error("dpct::sparse::spgemm_finalize(): The data type "
                               "of the col_ind in matrix c is unsupported.");
    }
  }
}

namespace detail {
template <typename T>
inline void spsv_case(sycl::queue queue, oneapi::mkl::uplo uplo,
                      oneapi::mkl::diag diag, oneapi::mkl::transpose trans_a,
                      const void *alpha, sparse_matrix_desc_t a,
                      std::shared_ptr<dense_vector_desc> x,
                      std::shared_ptr<dense_vector_desc> y) {
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const T *>(alpha), queue);
  T *new_x_ptr = nullptr;
  if (alpha_value != T(1.0f)) {
    new_x_ptr = (T *)dpct::dpct_malloc(x->get_ele_num() * sizeof(T));
    dpct::dpct_memcpy(new_x_ptr, x->get_value(), x->get_ele_num() * sizeof(T));
    auto data_new_x = dpct::detail::get_memory<T>(new_x_ptr);
    oneapi::mkl::blas::column_major::scal(queue, x->get_ele_num(), alpha_value,
                                          data_new_x, 1);
  } else {
    new_x_ptr = static_cast<T *>(x->get_value());
  }
  auto data_new_x = dpct::detail::get_memory<T>(new_x_ptr);
  auto data_y = dpct::detail::get_memory<T>(y->get_value());
  oneapi::mkl::sparse::trsv(queue, uplo, trans_a, diag, a->get_matrix_handle(),
                            data_new_x, data_y);
  if (alpha_value != T(1.0f)) {
    queue.wait();
    dpct::dpct_free(new_x_ptr);
  }
}
} // namespace detail

/// Performs internal optimizations for spsv by analyzing the provided matrix
/// structure and operation parameters.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] a Specifies the sparse matrix a.
inline void spsv_optimize(sycl::queue queue, oneapi::mkl::transpose trans_a,
                          sparse_matrix_desc_t a) {
  if (!a->get_uplo() || !a->get_diag()) {
    throw std::runtime_error(
        "dpct::sparse::spsv_optimize(): oneapi::mkl::sparse::optimize_trsv "
        "needs uplo and diag attributes to be specified.");
  }
  oneapi::mkl::sparse::optimize_trsv(
      queue, a->get_uplo().value(), oneapi::mkl::transpose::nontrans,
      a->get_diag().value(), a->get_matrix_handle());
}

/// Solves a system of linear equations for a sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] x Specifies the dense vector x.
/// \param [in, out] y Specifies the dense vector y.
/// \param [in] data_type Specifies the data type of \param a, \param x and
/// \param y .
inline void spsv(sycl::queue queue, oneapi::mkl::transpose trans_a,
                 const void *alpha, sparse_matrix_desc_t a,
                 std::shared_ptr<dense_vector_desc> x,
                 std::shared_ptr<dense_vector_desc> y,
                 library_data_t data_type) {
  if (!a->get_uplo() || !a->get_diag()) {
    throw std::runtime_error(
        "dpct::sparse::spsv(): oneapi::mkl::sparse::trsv needs uplo and diag "
        "attributes to be specified.");
  }
  oneapi::mkl::uplo uplo = a->get_uplo().value();
  oneapi::mkl::diag diag = a->get_diag().value();

  std::uint64_t key = dpct::detail::get_type_combination_id(
      a->get_value_type(), x->get_value_type(), y->get_value_type());
  switch (key) {
  case dpct::detail::get_type_combination_id(library_data_t::real_float,
                                             library_data_t::real_float,
                                             library_data_t::real_float): {
    detail::spsv_case<float>(queue, uplo, diag, trans_a, alpha, a, x, y);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::real_double,
                                             library_data_t::real_double,
                                             library_data_t::real_double): {
    detail::spsv_case<double>(queue, uplo, diag, trans_a, alpha, a, x, y);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                             library_data_t::complex_float,
                                             library_data_t::complex_float): {
    detail::spsv_case<std::complex<float>>(queue, uplo, diag, trans_a, alpha, a,
                                           x, y);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::complex_double,
                                             library_data_t::complex_double,
                                             library_data_t::complex_double): {
    detail::spsv_case<std::complex<double>>(queue, uplo, diag, trans_a, alpha,
                                            a, x, y);
    break;
  }
  default:
    throw std::runtime_error("dpct::sparse::spsv(): The combination of the "
                             "value types of a, x and y is unsupported.");
  }
}
#endif
} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
