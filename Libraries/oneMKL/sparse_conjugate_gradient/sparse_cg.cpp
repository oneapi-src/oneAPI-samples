//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This sample demonstrates use of oneAPI Math Kernel Library (oneMKL)
*       sparse BLAS API to solve a system of linear equations (Ax=b).
*
*       It uses the preconditioned conjugate gradient method with a symmetric
*       Gauss-Seidel preconditioner:
*
*       Compute r_0 = b - Ax_0
*       w_0 = B^{-1}*r_0 and p_0 = w_0
*       while not converged
*           {
*                   alpha_k = (r_k , w_k )/(Ap_k , p_k )
*                   x_{k+1} = x_k + alpha_k*p_k
*                   r_{k+1} = r_k - alpha_k*A*p_k
*                   w_{k+1} = B^{-1}*r_{k+1}
*                   beta_k = (r_{k+1}, w_{k+1})/(r_k , w_k )
*                   p_{k+1} = w_{k+1} + beta_k*p_k
*           }
*
*       where A = -L+D-L^t; B = (D-L)*D^{-1}*(D-L^t).
*
*       The supported floating point data types for gemm matrix data are:
*           float
*           double
*
*/

// stl includes
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <vector>

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

#include <CL/sycl.hpp>

#include "utils.hpp"

using namespace oneapi;


template <typename fp, typename intType>
static void diagonal_mv(sycl::queue main_queue,
                        const intType nrows,
                        sycl::buffer<fp, 1> &d_buffer,
                        sycl::buffer<fp, 1> &t_buffer)
{
    main_queue.submit([&](sycl::handler &cgh) {
        auto d = (d_buffer).template get_access<sycl::access::mode::write>(cgh);
        auto t = (t_buffer).template get_access<sycl::access::mode::read_write>(cgh);
        auto diagonalMVKernel = [=](sycl::item<1> item) {
            const int row = item.get_id(0);
            t[row] *= d[row];
        };
        cgh.parallel_for(sycl::range<1>(nrows), diagonalMVKernel);
    });
}

template <typename fp, typename intType>
void run_sparse_cg_example(const sycl::device &dev)
{
    // Matrix data size
    intType size  = 4;
    intType nrows = size * size * size;

    // Input matrix in CSR format
    std::vector<intType> ia;
    std::vector<intType> ja;
    std::vector<fp> a;

    ia.resize(nrows + 1);
    ja.resize(27 * nrows);
    a.resize(27 * nrows);

    generate_sparse_matrix<fp, intType>(size, ia, ja, a);

    // Vectors x and y
    std::vector<fp> x;
    std::vector<fp> b;
    x.resize(nrows);
    b.resize(nrows);

    // Init right hand side and vector x
    for (int i = 0; i < nrows; i++) {
        b[i] = 1;
        x[i] = 0;
    }

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL "
                             "exception during sparse CG:\n"
                          << e.what() << std::endl;
            }
        }
    };

    //
    // Execute CG
    //

    // create execution queue and buffers of matrix data
    sycl::queue main_queue(dev, exception_handler);

    sycl::buffer<intType, 1> ia_buffer(ia.data(), nrows + 1);
    sycl::buffer<intType, 1> ja_buffer(ja.data(), ia[nrows]);
    sycl::buffer<fp, 1> a_buffer(a.data(), ia[nrows]);
    sycl::buffer<fp, 1> x_buffer(x);
    sycl::buffer<fp, 1> b_buffer(b);
    sycl::buffer<fp, 1> r_buffer(nrows);
    sycl::buffer<fp, 1> w_buffer(nrows);
    sycl::buffer<fp, 1> p_buffer(nrows);
    sycl::buffer<fp, 1> t_buffer(nrows);
    sycl::buffer<fp, 1> y_buffer(nrows);
    sycl::buffer<fp, 1> d_buffer(nrows);
    sycl::buffer<fp, 1> temp_buffer(1);

    // create and initialize handle for a Sparse Matrix in CSR format
    mkl::sparse::matrix_handle_t handle;

    try {
        mkl::sparse::init_matrix_handle(&handle);

        mkl::sparse::set_csr_data(handle, nrows, nrows, mkl::index_base::zero,
                                          ia_buffer, ja_buffer, a_buffer);

        mkl::sparse::set_matrix_property(handle, mkl::sparse::property::symmetric);
        mkl::sparse::set_matrix_property(handle, mkl::sparse::property::sorted);

        mkl::sparse::optimize_trsv(main_queue, mkl::uplo::lower,
                                           mkl::transpose::nontrans,
                                           mkl::diag::nonunit, handle);
        mkl::sparse::optimize_trsv(main_queue, mkl::uplo::upper,
                                           mkl::transpose::nontrans,
                                           mkl::diag::nonunit, handle);
        mkl::sparse::optimize_gemv(main_queue, mkl::transpose::nontrans, handle);

        main_queue.submit([&](sycl::handler &cgh) {
            auto ia = (ia_buffer).template get_access<sycl::access::mode::read>(cgh);
            auto ja = (ja_buffer).template get_access<sycl::access::mode::read>(cgh);
            auto a  = (a_buffer).template get_access<sycl::access::mode::read>(cgh);
            auto d  = (d_buffer).template get_access<sycl::access::mode::write>(cgh);
            auto extractDiagonalKernel = [=](sycl::item<1> item) {
                const int row = item.get_id(0);
                for (intType i = ia[row]; i < ia[row + 1]; i++) {
                    if (ja[i] == row) {
                        d[row] = a[i];
                        break;
                    }
                }
            };
            cgh.parallel_for(sycl::range<1>(nrows), extractDiagonalKernel);
        });

        // initial residual equal to RHS cause of zero initial vector
        mkl::blas::copy(main_queue, nrows, b_buffer, 1, r_buffer, 1);

        // Calculation B^{-1}r_0
        {
            mkl::sparse::trsv(main_queue, mkl::uplo::lower,
                                      mkl::transpose::nontrans, mkl::diag::nonunit,
                                      handle, r_buffer, t_buffer);
            diagonal_mv<fp, intType>(main_queue, nrows, d_buffer, t_buffer);
            mkl::sparse::trsv(main_queue, mkl::uplo::upper,
                                      mkl::transpose::nontrans, mkl::diag::nonunit,
                                      handle, t_buffer, w_buffer);
        }

        mkl::blas::copy(main_queue, nrows, w_buffer, 1, p_buffer, 1);

        // Calculate initial norm of correction
        fp initial_norm_of_correction = 0;
        mkl::blas::nrm2(main_queue, nrows, w_buffer, 1, temp_buffer);
        {
            auto temp_accessor = temp_buffer.template get_access<sycl::access::mode::read>();
            initial_norm_of_correction = temp_accessor[0];
        }
        fp norm_of_correction = initial_norm_of_correction;

        // Start of main PCG algorithm
        std::int32_t k = 0;
        fp alpha, beta, temp;

        mkl::blas::dot(main_queue, nrows, r_buffer, 1, w_buffer, 1, temp_buffer);
        {
            auto temp_accessor = temp_buffer.template get_access<sycl::access::mode::read>();
            temp               = temp_accessor[0];
        }

        while (norm_of_correction / initial_norm_of_correction > 1.e-3 && k < 100) {
            // Calculate A*p
            mkl::sparse::gemv(main_queue, mkl::transpose::nontrans, 1.0, handle,
                                      p_buffer, 0.0, t_buffer);

            // Calculate alpha_k
            mkl::blas::dot(main_queue, nrows, p_buffer, 1, t_buffer, 1, temp_buffer);
            {
                auto temp_accessor =
                        temp_buffer.template get_access<sycl::access::mode::read>();
                alpha = temp / temp_accessor[0];
            }

            // Calculate x_k = x_k + alpha*p_k
            mkl::blas::axpy(main_queue, nrows, alpha, p_buffer, 1, x_buffer, 1);
            // Calculate r_k = r_k - alpha*A*p_k
            mkl::sparse::gemv(main_queue, mkl::transpose::nontrans, -alpha, handle,
                                      p_buffer, 1.0, r_buffer);

            // Calculate w_k = B^{-1}r_k
            {
                mkl::sparse::trsv(main_queue, mkl::uplo::lower,
                                          mkl::transpose::nontrans,
                                          mkl::diag::nonunit, handle, r_buffer, t_buffer);
                diagonal_mv<fp, intType>(main_queue, nrows, d_buffer, t_buffer);
                mkl::sparse::trsv(main_queue, mkl::uplo::upper,
                                          mkl::transpose::nontrans,
                                          mkl::diag::nonunit, handle, t_buffer, w_buffer);
            }

            // Calculate current norm of correction
            mkl::blas::nrm2(main_queue, nrows, w_buffer, 1, temp_buffer);
            {
                auto temp_accessor = temp_buffer.template get_access<sycl::access::mode::read>();
                norm_of_correction = temp_accessor[0];
            }
            std::cout << "\t\trelative norm of residual on " << ++k
                      << " iteration: " << norm_of_correction / initial_norm_of_correction
                      << std::endl;
            if (norm_of_correction <= 1.e-3)
                break;

            // Calculate beta_k
            mkl::blas::dot(main_queue, nrows, r_buffer, 1, w_buffer, 1, temp_buffer);
            {
                auto temp_accessor = temp_buffer.template get_access<sycl::access::mode::read>();
                beta = temp_accessor[0] / temp;
                temp = temp_accessor[0];
            }

            // Calculate p_k = w_k+beta*p_k
            mkl::blas::axpy(main_queue, nrows, beta, p_buffer, 1, w_buffer, 1);
            mkl::blas::copy(main_queue, nrows, w_buffer, 1, p_buffer, 1);
        }

        std::cout << "\n\t\tPreconditioned CG process has successfully converged, and\n"
                  << "\t\tthe following solution has been obtained:\n\n";

        auto result = x_buffer.template get_access<sycl::access::mode::read>();
        for (std::int32_t i = 0; i < 4; i++) {
            std::cout << "\t\tx[" << i << "] = " << result[i] << std::endl;
        }
        std::cout << "\t\t..." << std::endl;
    }
    catch (std::exception const &e) {
        std::cout << "\t\tCaught exception:\n" << e.what() << std::endl;
    }
    
    mkl::sparse::release_matrix_handle(&handle);
}

//
// Description of example setup, apis used and supported floating point type
// precisions
//
void print_banner()
{
    std::cout << "###############################################################"
                 "#########\n"
                 "# Sparse Conjugate Gradient Solver\n"
                 "# \n"
                 "# Uses the preconditioned conjugate gradient algorithm to\n"
                 "# iteratively solve the symmetric linear system\n"
                 "# \n"
                 "#     A * x = b\n"
                 "# \n"
                 "# where A is a symmetric sparse matrix in CSR format, and\n"
                 "#       x and b are dense vectors.\n"
                 "# \n"
                 "# Uses the symmetric Gauss-Seidel preconditioner.\n"
                 "# \n"
                 "###############################################################"
                 "#########\n\n";
}

int main(int argc, char **argv)
{
    print_banner();

    sycl::device my_dev{sycl::default_selector{}};

    std::cout << "Running tests on " << my_dev.get_info<sycl::info::device::name>() << ".\n";

    std::cout << "\tRunning with single precision real data type:" << std::endl;
    run_sparse_cg_example<float, std::int32_t>(my_dev);

    if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
        std::cout << "\tRunning with double precision real data type:" << std::endl;
        run_sparse_cg_example<double, std::int32_t>(my_dev);
    }
}
