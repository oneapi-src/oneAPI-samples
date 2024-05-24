/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Multi-GPU sample using OpenMP for threading on the CPU side
 * needs a compiler that supports OpenMP 2.0
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <helper_cuda.h>
#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe

using namespace std;

// a simple kernel that simply increments each array element by b
void kernelAddConstant(int *g_a, const int b, const sycl::nd_item<3> &item_ct1) {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  g_a[idx] += b;
}

// a predicate that checks whether each array element is set to its index plus b
int correctResult(int *data, const int n, const int b) {
  for (int i = 0; i < n; i++)
    if (data[i] != i + b) return 0;

  return 1;
}

int main(int argc, char *argv[]) {
  int num_gpus = 0;  // number of CUDA GPUs

  printf("%s Starting...\n\n", argv[0]);

  /////////////////////////////////////////////////////////////////
  // determine the number of CUDA capable GPUs
  //
  num_gpus = dpct::dev_mgr::instance().device_count();

  if (num_gpus < 1) {
    printf("no CUDA capable devices were detected\n");
    return 1;
  }

  /////////////////////////////////////////////////////////////////
  // display CPU and GPU configuration
  //
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of CUDA devices:\t%d\n", num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    dpct::device_info dprop;
    dpct::get_device_info(dprop, dpct::dev_mgr::instance().get_device(i));
    printf("   %d: %s\n", i, dprop.get_name());
  }

  printf("---------------------------\n");

  /////////////////////////////////////////////////////////////////
  // initialize data
  //
  unsigned int n = num_gpus * 8192;
  unsigned int nbytes = n * sizeof(int);
  int *a = 0;  // pointer to data on the CPU
  int b = 3;   // value by which the array is incremented
  a = (int *)malloc(nbytes);

  if (0 == a) {
    printf("couldn't allocate CPU memory\n");
    return 1;
  }

  for (unsigned int i = 0; i < n; i++) a[i] = i;

  ////////////////////////////////////////////////////////////////
  // run as many CPU threads as there are CUDA devices
  //   each CPU thread controls a different device, processing its
  //   portion of the data.  It's possible to use more CPU threads
  //   than there are CUDA devices, in which case several CPU
  //   threads will be allocating resources and launching kernels
  //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
  //   Recall that all variables declared inside an "omp parallel" scope are
  //   local to each CPU thread
  //
  omp_set_num_threads(
      num_gpus);  // create as many CPU threads as there are CUDA devices
// omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there
// are CUDA devices
#pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    // set and check the CUDA device for this CPU thread
    int gpu_id = -1;
    /*
    DPCT1093:15: The "cpu_thread_id %
        num_gpus" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(
        cpu_thread_id %
        num_gpus))); // "% num_gpus" allows more CPU threads than GPU devices
    checkCudaErrors(DPCT_CHECK_ERROR(
        gpu_id = dpct::dev_mgr::instance().current_device_id()));
    printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id,
           num_cpu_threads, gpu_id);

    int *d_a =
        0;  // pointer to memory on the device associated with this CPU thread
    int *sub_a =
        a +
        cpu_thread_id * n /
            num_cpu_threads;  // pointer to this CPU thread's portion of data
    unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
    sycl::range<3> gpu_threads(1, 1, 128); // 128 threads per block
    sycl::range<3> gpu_blocks(1, 1, n / (gpu_threads[2] * num_cpu_threads));

    checkCudaErrors(
        DPCT_CHECK_ERROR(d_a = (int *)sycl::malloc_device(
                             nbytes_per_kernel, dpct::get_in_order_queue())));
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memset(d_a, 0, nbytes_per_kernel).wait()));
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(d_a, sub_a, nbytes_per_kernel)
                                         .wait()));
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(gpu_blocks * gpu_threads, gpu_threads),
        [=](sycl::nd_item<3> item_ct1) {
          kernelAddConstant(d_a, b, item_ct1);
        });

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(sub_a, d_a, nbytes_per_kernel)
                                         .wait()));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_a, dpct::get_in_order_queue())));
  }
  printf("---------------------------\n");

  /*
  DPCT1010:16: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  if (0 != 0)
    /*
    DPCT1009:17: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced by a placeholder string. You need to
    rewrite this code.
    */
    /*
    DPCT1010:18: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    printf("%s\n", "<Placeholder string>");

  ////////////////////////////////////////////////////////////////
  // check the result
  //
  bool bResult = correctResult(a, n, b);

  if (a) free(a);  // free CPU memory

  exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
