//==============================================================
// Matrix Multiplication: oneMKL
//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <CL/sycl.hpp>
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions

using namespace sycl;

void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N, size_t M) {
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << "\n";
    
    //# Create buffers for matrices
    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);

    //# scalar multipliers for oneMKL
    float alpha = 1.f, beta = 1.f;

    //# transpose status of matrices for oneMKL
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

    //# Submit MKL library call to execute on device
    oneapi::mkl::blas::gemm(q, transA, transB, N, N, N, alpha, b, N, a, N, beta, c, N);
    c.get_access<access::mode::read>();
}
