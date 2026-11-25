/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::queue *cublasH = NULL;
    dpct::queue_ptr stream = &q_ct1;

    const int m = 2;
    const int n = 2;
    const int lda = m;

    /*
     *   AP = | 1.0 3.0 |
     *        | 3.0 4.0 |
     *   x  = | 5.0 6.0 |
     */

    std::vector<data_type> AP = {1.0, 3.0, 3.0, 4.0};
    const std::vector<data_type> x = {5.0, 6.0};
    const data_type alpha = 1.0;
    const int incx = 1;

    data_type *d_AP = nullptr;
    data_type *d_x = nullptr;

    oneapi::mkl::uplo uplo = oneapi::mkl::uplo::upper;

    printf("AP\n");
    print_matrix(m, n, AP.data(), lda);
    printf("=====\n");

    printf("x\n");
    print_vector(x.size(), x.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    /*
    DPCT1003:101: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUBLAS_CHECK((cublasH = &q_ct1, 0));

    CUDA_CHECK((stream = dev_ct1.create_queue(), 0));
    /*
    DPCT1003:102: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUBLAS_CHECK((cublasH = stream, 0));

    /* step 2: copy data to device */
    CUDA_CHECK((d_AP = (data_type *)sycl::malloc_device(
                    sizeof(data_type) * AP.size(), q_ct1),
                0));
    CUDA_CHECK((d_x = (data_type *)sycl::malloc_device(
                    sizeof(data_type) * x.size(), q_ct1),
                0));

    CUDA_CHECK(
        (stream->memcpy(d_AP, AP.data(), sizeof(data_type) * AP.size()), 0));
    CUDA_CHECK(
        (stream->memcpy(d_x, x.data(), sizeof(data_type) * x.size()), 0));

    /* step 3: compute */
    /*
    DPCT1003:103: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUBLAS_CHECK((oneapi::mkl::blas::column_major::spr(*cublasH, uplo, n, alpha,
                                                       d_x, incx, d_AP),
                  0));

    /* step 4: copy data to host */
    CUDA_CHECK(
        (stream->memcpy(AP.data(), d_AP, sizeof(data_type) * AP.size()), 0));

    CUDA_CHECK((stream->wait(), 0));

    /*
     *   AP = | 26.0 39.0 |
     *        | 33.0  4.0 |
     */

    printf("AP\n");
    print_matrix(m, n, AP.data(), lda);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK((sycl::free(d_AP, q_ct1), 0));
    CUDA_CHECK((sycl::free(d_x, q_ct1), 0));

    /*
    DPCT1003:104: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUBLAS_CHECK((cublasH = nullptr, 0));

    CUDA_CHECK((dev_ct1.destroy_queue(stream), 0));

    CUDA_CHECK((dev_ct1.reset(), 0));

    return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
