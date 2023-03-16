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

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "cublas_utils.h"

using data_type = sycl::double2;

int main(int argc, char* argv[]) try {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  sycl::queue* cublasH = NULL;
  dpct::queue_ptr stream = &q_ct1;

  /*
   *   A = | 1.1 + 1.2j | 2.3 + 2.4j | 3.5 + 3.6j | 4.7 + 4.8j |
   *   B = | 5.1 + 5.2j | 6.3 + 6.4j | 7.5 + 7.6j | 8.7 + 8.8j |
   */

  const std::vector<data_type> A = {
      {1.1, 1.2}, {2.3, 2.4}, {3.5, 3.6}, {4.7, 4.8}};
  const std::vector<data_type> B = {
      {5.1, 5.2}, {6.3, 6.4}, {7.5, 7.6}, {8.7, 8.8}};
  const int incx = 1;
  const int incy = 1;

  data_type result = {0.0, 0.0};

  data_type* d_A = nullptr;
  data_type* d_B = nullptr;

  printf("A\n");
  print_vector(A.size(), A.data());
  printf("=====\n");

  printf("B\n");
  print_vector(B.size(), B.data());
  printf("=====\n");

  /* step 1: create cublas handle, bind a stream */
  cublasH = &q_ct1;

  stream = dev_ct1.create_queue();
  cublasH = stream;

  /* step 2: copy data to device */
  d_A = (data_type*)sycl::malloc_device(sizeof(data_type) * A.size(), q_ct1);
  d_B = (data_type*)sycl::malloc_device(sizeof(data_type) * B.size(), q_ct1);

  stream->memcpy(d_A, A.data(), sizeof(data_type) * A.size());
  stream->memcpy(d_B, B.data(), sizeof(data_type) * B.size());

  /* step 3: compute */
  [&]() {
    sycl::double2* res_temp_ptr_ct6 = &result;
    if (sycl::get_pointer_type(&result, cublasH->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&result, cublasH->get_context()) !=
            sycl::usm::alloc::shared) {
      res_temp_ptr_ct6 =
          sycl::malloc_shared<sycl::double2>(1, dpct::get_default_queue());
    }
    oneapi::mkl::blas::column_major::dotc(
        *cublasH, A.size(), (std::complex<double>*)d_A, incx,
        (std::complex<double>*)d_B, incy,
        (std::complex<double>*)res_temp_ptr_ct6);
    if (sycl::get_pointer_type(&result, cublasH->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(&result, cublasH->get_context()) !=
            sycl::usm::alloc::shared) {
      cublasH->wait();
      result = *res_temp_ptr_ct6;
      sycl::free(res_temp_ptr_ct6, dpct::get_default_queue());
    }
    return 0;
  }();

  stream->wait();

  /*
   *   result = 178.44+-1.60j
   */

  printf("Result\n");
  printf("%0.2f+%0.2fj\n", result.x(), result.y());
  printf("=====\n");

  /* free resources */
  sycl::free(d_A, q_ct1);
  sycl::free(d_B, q_ct1);

  cublasH = nullptr;

  dev_ct1.destroy_queue(stream);

  dev_ct1.reset();

  return EXIT_SUCCESS;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
