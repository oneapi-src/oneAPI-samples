//==============================================================
// Copyright © 2024 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*       This example demonstrates use of oneAPI Math Kernel Library (oneMKL)
*       SPARSE BLAS and BLAS USM APIs to solve a system of linear equations (Ax=b)
*       by preconditioned Conjugate Gradient (PCG) method with the Symmetric
*       Gauss-Seidel preconditioner:
*
*       Solve A*x = b
*
*       x_0 initial guess
*       r_0 = b - A*x_0
*       k = 0
*       while (||r_k|| / ||r_0|| > relTol and k < maxIter )
*     
*           solve M*z_k = r_k for z_k
*           if (k == 0)
*               p_1 = z_0
*           else
*               beta_k = dot(r_k, z_k) / dot(r_{k-1}, z_{k-1})
*               p_{k+1} = z_k + beta_k * p_k
*           end if
*           Ap_{k+1} = A*p_{k+1}
*           alpha_{k+1} = (r_k, z_k) / (p_{k+1}, Ap_{k+1})
*     
*           x_{k+1} = x_k + alpha_{k+1} * p_{k+1}
*           r_{k+1} = r_k - alpha_{k+1} * Ap_{k+1}
*           if (||r_k|| < absTol) break with convergence
*           
*           k=k+1
*       end
*   
*       where A = L+D+L^T is in CSR format and the preconditioner
*       is M = (D+L)*D^{-1}*(D+L^T).
*
*       Note that:
*       
*         x is the solution
*         r is the residual
*         z is the preconditioned residual
*         p is the search direction
*
*       and we are using ||r||_2 for stopping criteria and alpha/beta scalars are 
*       provided as constants from host side.
*
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
#include <iomanip>
#include <iterator>
#include <limits>
#include <list>
#include <vector>

#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "utils.hpp"

using namespace oneapi;

template <typename dataType, typename intType>
class extractDiagonalClass;

template <typename dataType, typename intType>
class modifyDiagonalClass;

template <typename dataType, typename intType>
class diagonalMVClass;

template <typename dataType, typename intType>
class noPreconClass;

template <typename dataType, typename intType>
class jacobiPreconClass;


//
// extract diagonal from matrix
//
template <typename dataType, typename intType>
sycl::event extract_diagonal(sycl::queue q,
                             const intType n,
                             const intType *ia_d,
                             const intType *ja_d,
                             const dataType *a_d,
                                   dataType *d_d,
                                   dataType *invd_d,
                             const std::vector<sycl::event> &deps = {})
{
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        auto kernel = [=](sycl::item<1> item) {
            const int row = item.get_id(0);
            for (intType i = ia_d[row]; i < ia_d[row + 1]; i++) {
                if (ja_d[i] == row) {
                    dataType diagVal = a_d[i];
                    d_d[row] = diagVal;
                    invd_d[row] = dataType(1.0) / diagVal;
                    break;
                }
            }
        };
        cgh.parallel_for<class extractDiagonalClass<dataType, intType>>(sycl::range<1>(n), kernel);
    });
}

//
// Modify diagonal value in matrix
//
template <typename dataType, typename intType>
sycl::event modify_diagonal(sycl::queue q,
                            const dataType new_diagVal,
                            const intType n,
                            const intType *ia_d,
                            const intType *ja_d,
                                  dataType *a_d, // to be modified
                                  dataType *d_d, // to be modified
                                  dataType *invd_d, // to be modified
                            const std::vector<sycl::event> &deps = {})
{
    assert(new_diagVal != dataType(0.0) );
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        auto kernel = [=](sycl::item<1> item) {
            const int row = item.get_id(0);
            for (intType i = ia_d[row]; i < ia_d[row + 1]; i++) {
                if (ja_d[i] == row) {
                    a_d[i] = new_diagVal;
                    d_d[row] = new_diagVal;
                    invd_d[row] = dataType(1.0) / new_diagVal;
                    break;
                }
            }
        };
        cgh.parallel_for<class modifyDiagonalClass<dataType, intType>>(sycl::range<1>(n), kernel);
    });
}


//
// Scale by diagonal
//
// t = D * t
//
template <typename dataType, typename intType>
sycl::event diagonal_mv(sycl::queue q,
                        const intType n,
                        const dataType *d,
                              dataType *t,
                        const std::vector<sycl::event> &deps = {})
{
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        auto kernel = [=](sycl::item<1> item) {
            const int row = item.get_id(0);
            t[row] *= d[row];
        };
        cgh.parallel_for<class diagonalMVClass<dataType, intType>>(sycl::range<1>(n), kernel);
    });
}


//
// No Preconditioner
//
// solve M z = r   where M = Identity 
// z = r;
//
template <typename dataType, typename intType>
sycl::event precon_none(sycl::queue q,
                        const intType n,
                        const dataType *r,
                              dataType *z,
                        const std::vector<sycl::event> &deps = {})
{
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        auto kernel = [=](sycl::item<1> item) {
            const int row = item.get_id(0);
            z[row] = r[row];
        };
        cgh.parallel_for<class noPreconClass<dataType, intType>>(sycl::range<1>(n), kernel);
    });
}


//
// Jacobi Preconditioner
//
// solve M z = r   where M = D = diag(a_00, a_11, a_22, ...)
//
// z = inv(D) * r;
//
template <typename dataType, typename intType>
sycl::event precon_jacobi(sycl::queue q,
                          const intType n,
                          oneapi::mkl::sparse::matrix_handle_t A,
                          const dataType *invd,
                          const dataType *r,
                                dataType *z, // output
                          const std::vector<sycl::event> &deps = {})
{
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        auto kernel = [=](sycl::item<1> item) {
            const int row = item.get_id(0);
            z[row] = invd[row] * r[row];
        };
        cgh.parallel_for<class jacobiPreconClass<dataType, intType>>(sycl::range<1>(n), kernel);
    });
}


//
// Gauss-Seidel Preconditioner
//
// solve M z = r   where M = (L+D)*inv(D)*(D+U)
//
// t = inv(D+L) * r;   // forward triangular solve
// t = D*t             // diagonal mv
// z = inv(D+U) * t    // backward triangular solve
//
template <typename dataType, typename intType>
sycl::event precon_gauss_seidel(sycl::queue q,
                                const intType n,
                                oneapi::mkl::sparse::matrix_handle_t A,
                                const dataType *d,
                                const dataType *r,
                                      dataType *t, // temporary workspace
                                      dataType *z, // output
                                const std::vector<sycl::event> &deps = {})
{

    auto ev_trsvL = oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
            oneapi::mkl::diag::nonunit, dataType(1.0) /* alpha */, A, r, t, deps);
    auto ev_diagmv = diagonal_mv<dataType, intType>(q, n, d, t, {ev_trsvL});
    auto ev_trsvU = oneapi::mkl::sparse::trsv(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
            oneapi::mkl::diag::nonunit, dataType(1.0) /* alpha */, A, t, z, {ev_diagmv});

    return ev_trsvU;
}



template <typename dataType, typename intType>
int run_sparse_pcg_example(const sycl::device &dev)
{
    
    int good = 0;

    // Matrix data size
    const intType size  = 16;
    const intType n = size * size * size; // A is n x n
    
    const intType nnzUB = 27 * n; // upper bound of nnz from 27 point stencil

    // PCG settings
    const intType maxIter = 500;
    const dataType relTol = 1.0e-5;
    const dataType absTol = 5.0e-4;

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

    // create queue around device
    sycl::queue q(dev, exception_handler);


    // Input matrix in CSR format
    intType *ia_h = sycl::malloc_host<intType>(n+1, q);
    intType *ja_h = sycl::malloc_host<intType>(nnzUB, q);
    dataType *a_h = sycl::malloc_host<dataType>(nnzUB, q);
    dataType *x_h = sycl::malloc_host<dataType>(n, q);
    dataType *b_h = sycl::malloc_host<dataType>(n, q);

    if (!ia_h || !ja_h || !a_h || !x_h || !b_h ) {
        throw std::runtime_error("Failed to allocate host side USM memory");
    }

    // 
    // Generate a 27 point stencil for 3D laplacian using size elements in each dimension
    //
    generate_sparse_matrix<dataType, intType>(size, ia_h, ja_h, a_h);

    // Init right hand side and vector x
    for (int i = 0; i < n; i++) {
        b_h[i] = set_fp_value(dataType(1.0), dataType(0.0)); // rhs b = 1
        x_h[i] = set_fp_value(dataType(0.0), dataType(0.0)); // initial guess x0 = 0
    }

    //
    // Execute Preconditioned Conjugate Gradient Algorithm
    //
    // solve  A x = b  starting with initial guess x = x0
    //

    std::cout << "\n\t\tsparse PCG parameters:\n";

    std::cout << "\t\t\tA size: (" << n << ", " << n << ")" << std::endl;
    std::cout << "\t\t\tPreconditioner = Symmetric Gauss-Seidel" << std::endl;
    std::cout << "\t\t\tmax iterations = " << maxIter << std::endl;
    std::cout << "\t\t\trelative tolerance limit = " << relTol << std::endl;
    std::cout << "\t\t\tabsolute tolerance limit = " << absTol << std::endl;


    const intType nnz = ia_h[n]; // assumes zero indexing

    // create arrays for help
    intType *ia_d    = sycl::malloc_device<intType>(n+1, q);  // matrix rowptr
    intType *ja_d    = sycl::malloc_device<intType>(nnz, q);  // matrix columns
    dataType *a_d    = sycl::malloc_device<dataType>(nnz, q); // matrix values
    dataType *x_d    = sycl::malloc_device<dataType>(n, q);   // solution
    dataType *b_d    = sycl::malloc_device<dataType>(n, q);   // right hand side
    dataType *r_d    = sycl::malloc_device<dataType>(n, q);   // residual
    dataType *z_d    = sycl::malloc_device<dataType>(n, q);   // preconditioned residual
    dataType *p_d    = sycl::malloc_device<dataType>(n, q);   // search direction
    dataType *t_d    = sycl::malloc_device<dataType>(n, q);   // helper array
    dataType *d_d    = sycl::malloc_device<dataType>(n, q);   // matrix diagonals
    dataType *invd_d = sycl::malloc_device<dataType>(n, q);   // matrix reciprocal of diagonals

    const intType width = 8; // width * sizeof(dataType) >= cacheline size (64 Bytes)
    dataType *temp_d = sycl::malloc_device<dataType>(3*width, q);
    dataType *temp_h = sycl::malloc_host<dataType>(3*width, q);

    if ( !ia_d || !ja_d || !a_d || !x_d || !b_d || !z_d || !p_d || !t_d || !d_d || !invd_d || !temp_d || !temp_h) {
        throw std::runtime_error("Failed to allocate device side USM memory");
    }

    // device side aliases scattered by width elements each
    dataType *normr_h  = temp_h;
    dataType *rtz_h    = temp_h+1*width;
    dataType *pAp_h    = temp_h+2*width;
    dataType *normr_d  = temp_d;
    dataType *rtz_d    = temp_d+1*width;
    dataType *pAp_d    = temp_d+2*width;

    // copy data from host to device arrays
    q.copy(ia_h, ia_d, n+1).wait();
    q.copy(ja_h, ja_d, nnz).wait();
    q.copy(a_h, a_d, nnz).wait();
    q.copy(x_h, x_d, n).wait();
    q.copy(b_h, b_d, n).wait();

    extract_diagonal<dataType, intType>(q,n, ia_d, ja_d, a_d, d_d, invd_d, {}).wait();

    // make the matrix diagonally dominant
    modify_diagonal<dataType, intType>(q, dataType(52.0), n, ia_d, ja_d, a_d, d_d, invd_d, {}).wait();

    // create and initialize handle for a Sparse Matrix in CSR format
    oneapi::mkl::sparse::matrix_handle_t A = nullptr;

    try {
        // setup optimizations and properties we know about A matrix
        oneapi::mkl::sparse::init_matrix_handle(&A);

#if (INTEL_MKL_VERSION < 20250300)
        auto ev_set = oneapi::mkl::sparse::set_csr_data(q, A, n, n,
                oneapi::mkl::index_base::zero, ia_d, ja_d, a_d, {});
#else
        auto ev_set = oneapi::mkl::sparse::set_csr_data(q, A, n, n, nnz,
                oneapi::mkl::index_base::zero, ia_d, ja_d, a_d, {});
#endif

        oneapi::mkl::sparse::set_matrix_property(A, oneapi::mkl::sparse::property::symmetric);
        oneapi::mkl::sparse::set_matrix_property(A, oneapi::mkl::sparse::property::sorted);

        auto ev_optSvL = oneapi::mkl::sparse::optimize_trsv(q,
                oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
                oneapi::mkl::diag::nonunit, A, {ev_set});
        auto ev_optSvU = oneapi::mkl::sparse::optimize_trsv(q,
                oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
                oneapi::mkl::diag::nonunit, A, {ev_optSvL});
        auto ev_optGemv = oneapi::mkl::sparse::optimize_gemv(q,
                oneapi::mkl::transpose::nontrans, A, {ev_optSvU});
        // done setting up optimizations for A matrix

        // initial residual r_0 = b - A * x_0
        auto ev_r = oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0, A,
                                      x_d, 0.0, r_d, {ev_optGemv}); // r := A * x

        ev_r = oneapi::mkl::blas::axpby(q, n, 1.0, b_d, 1, -1.0, r_d, 1, {ev_r}); // r := 1 * b + -1 * r

        auto ev_normr = oneapi::mkl::blas::nrm2(q, n, r_d, 1, normr_d, {ev_r});
        dataType oldrTz = 0.0, rTz = 0.0, pAp = 0.0, normr = 0.0, normr_0 = 0.0;
        {
            q.copy(normr_d, normr_h, 12, {ev_normr}).wait();
            normr = std::sqrt(normr_h[0]);
            normr_0 = normr;
        }

        sycl::event ev_z, ev_rtz, ev_p, ev_Ap, ev_pAp, ev_x;

        std::int32_t k = 0;
        while ( normr / normr_0 > relTol && k < maxIter) {

            // Calculation z_k = M^{-1}r_k
            //ev_z = precon_none<dataType,intType>(q, n, r_d, z_d, {ev_r});
            //ev_z = precon_jacobi<dataType, intType>(q, n, A, invd_d, r_d, z_d, {ev_r});
            ev_z = precon_gauss_seidel<dataType, intType>(q, n, A, d_d, r_d, t_d, z_d, {ev_r});
 
            if (k == 0 ) {
                ev_rtz = oneapi::mkl::blas::dot(q, n, r_d, 1, z_d, 1, rtz_d, {ev_r, ev_z});
                {
                    q.copy(rtz_d, rtz_h, 1, {ev_rtz}).wait(); // synch point
                    rTz = rtz_h[0];
                }

                // copy D2D: p_1 = z_0
                ev_p = oneapi::mkl::blas::copy(q, n, z_d, 1, p_d, 1, {ev_z, ev_rtz});
            }
            else {
                // beta_{k+1} = dot(r_k, z_k) / dot(r_{k-1}, z_{k-1})
                ev_rtz = oneapi::mkl::blas::dot(q, n, r_d, 1, z_d, 1, rtz_d, {ev_r, ev_z});
                {
                    q.copy(rtz_d, rtz_h, 1, {ev_rtz}).wait(); // synch point
                    oldrTz = rTz;
                    rTz = rtz_h[0];
                }

                // Calculate p_{k+1} = z_{k+1} + beta_{k+1} * p_k
                ev_p = oneapi::mkl::blas::axpby(q, n, 1.0, z_d, 1, rTz / oldrTz, p_d, 1, {ev_rtz});

            }

            // Calculate Ap_{k+1} = A*p_{k+1} 
            ev_Ap = oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans,
                    1.0, A, p_d, 0.0, t_d, {ev_p});

            // alpha_{k+1} = dot(r_k, z_k) / dot(p_{k+1}, Ap_{k+1})
            ev_pAp = oneapi::mkl::blas::dot(q, n, p_d, 1, t_d, 1, pAp_d, {ev_Ap}); 
            {
                q.copy(pAp_d, pAp_h, 1, {ev_pAp}).wait(); // synch point
                pAp = pAp_h[0];
            }

            // Calculate x_{k+1} = x_k + alpha_{k+1}*p_{k+1}
            ev_x = oneapi::mkl::blas::axpy(q, n, rTz / pAp, p_d, 1, x_d, 1, {});

            // Calculate r_{k+1} = r_k - alpha_{k+1}*Ap_{k+1} (note that t = A*p_{k+1} right now so it can be reused here)
            ev_r = oneapi::mkl::blas::axpy(q, n, -rTz / pAp, t_d, 1, r_d, 1, {});

            // temp_d = ||r_{k+1}||^2
            ev_normr = oneapi::mkl::blas::nrm2(q, n, z_d, 1, normr_d, {ev_r});
            {
                q.copy(normr_d, normr_h, 1, {ev_normr}).wait(); // synch point
                normr = std::sqrt(normr_h[0]);
            }
            
            k++; // increment k counter
            std::cout << "\t\t\t\trelative norm of residual on " << std::setw(4) << k  // output in 1 base indexing
                      << " iteration: " << normr / normr_0 << std::endl;
            if (normr <= absTol) {
                std::cout << "\t\t\t\tabsolute norm of residual on " << std::setw(4) << k // output in 1-based indexing
                    << " iteration: " <<  normr << std::endl;
                break;
            }
            
        } // while normr / normr_0 > relTol && k < maxIter

        if (normr < absTol) {
            std::cout << "" << std::endl;
            std::cout << "\t\tPreconditioned CG process has successfully converged in absolute error in " << std::setw(4) << k << " steps with" << std::endl;
            good = 1;
        }
        else if (k <= maxIter && normr / normr_0 <= relTol) {
            std::cout << "" << std::endl;
            std::cout << "\t\tPreconditioned CG process has successfully converged in relative error in " << std::setw(4) << k << " steps with" << std::endl;
            good = 1;
        } else {
            std::cout << "" << std::endl;
            std::cout << "\t\tPreconditioned CG process has not converged after " << k << " steps with" << std::endl;
            good = 0;
        }

        std::cout << "\t\t relative error ||r||_2 / ||r_0||_2 = " << normr / normr_0 << (normr / normr_0 < relTol ? " < " : " > ") << relTol << std::endl;
        std::cout << "\t\t absolute error ||r||_2             = " << normr << (normr < absTol ? " < " : " > ") << absTol << std::endl;
        std::cout << "" << std::endl;

        oneapi::mkl::sparse::release_matrix_handle(q, &A, {}).wait();
    }
    catch (sycl::exception const &e) {
        std::cout << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;

        q.wait();
        oneapi::mkl::sparse::release_matrix_handle(q, &A).wait();
        return 1;
    }
    catch (std::exception const &e) {
        std::cout << "\t\tCaught std exception:\n" << e.what() << std::endl;

        q.wait();
        oneapi::mkl::sparse::release_matrix_handle(q, &A).wait();
        return 1;
    }

    q.wait();

    //  clean up USM memory allocations
    sycl::free(ia_h, q);
    sycl::free(ja_h, q);
    sycl::free(a_h, q);
    sycl::free(x_h, q);
    sycl::free(b_h, q);
    sycl::free(ia_d, q);
    sycl::free(ja_d, q);
    sycl::free(a_d, q);
    sycl::free(x_d, q);
    sycl::free(b_d, q);
    sycl::free(r_d, q);
    sycl::free(z_d, q);
    sycl::free(p_d, q);
    sycl::free(t_d, q);
    sycl::free(d_d, q);
    sycl::free(invd_d, q);
    sycl::free(temp_d, q);
    sycl::free(temp_h, q);

    return good ? 0 : 1;
}



//
// Description of example setup, apis used and supported floating point type
// precisions
//
void print_banner()
{

    std::cout << "###############################################################"
                 "#########\n"
                 "# Sparse Preconditioned Conjugate Gradient Solver with USM\n"
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
                 "# alpha and beta constants in PCG algorithm are host side.\n"
                 "# \n"
                 "###############################################################"
                 "#########\n\n";
}

int main(int argc, char **argv)
{
    print_banner();

    sycl::device my_dev{sycl::default_selector_v};

    std::cout << "Running tests on " << my_dev.get_info<sycl::info::device::name>() << ".\n";

    std::cout << "\tRunning with single precision real data type:" << std::endl;
    run_sparse_pcg_example<float, std::int32_t>(my_dev);

    if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
        std::cout << "\tRunning with double precision real data type:" << std::endl;
        run_sparse_pcg_example<double, std::int32_t>(my_dev);
    }
}
