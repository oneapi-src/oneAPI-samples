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

//
// This sample demonstrates the use of streams for concurrent execution. It also
// illustrates how to introduce dependencies between CUDA streams with the
// cudaStreamWaitEvent function.
//

// Devices of compute capability 2.0 or higher can overlap the kernels
//
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
void clock_block(clock_t *d_o, clock_t clock_count) {
  /*
  DPCT1008:16: clock function is not defined in SYCL. This is a
  hardware-specific feature. Consult with your hardware vendor to find a
  replacement.
  */
  unsigned int start_clock = (unsigned int)clock();

  clock_t clock_offset = 0;

  while (clock_offset < clock_count) {
    /*
    DPCT1008:17: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    unsigned int end_clock = (unsigned int)clock();

    // The code below should work like
    // this (thanks to modular arithmetics):
    //
    // clock_offset = (clock_t) (end_clock > start_clock ?
    //                           end_clock - start_clock :
    //                           end_clock + (0xffffffffu - start_clock));
    //
    // Indeed, let m = 2^32 then
    // end - start = end + m - start (mod m).

    clock_offset = (clock_t)(end_clock - start_clock);
  }

  d_o[0] = clock_offset;
}

// Single warp reduction kernel
void sum(clock_t *d_clocks, int N, const sycl::nd_item<3> &item_ct1,
         clock_t *s_clocks) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();

  clock_t my_sum = 0;

  for (int i = item_ct1.get_local_id(2); i < N;
       i += item_ct1.get_local_range(2)) {
    my_sum += d_clocks[i];
  }

  s_clocks[item_ct1.get_local_id(2)] = my_sum;
  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for (int i = 16; i > 0; i /= 2) {
    if (item_ct1.get_local_id(2) < i) {
      s_clocks[item_ct1.get_local_id(2)] +=
          s_clocks[item_ct1.get_local_id(2) + i];
    }

    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
  int nkernels = 8;             // number of concurrent kernels
  int nstreams = nkernels + 1;  // use one more stream than concurrent kernel
  int nbytes = nkernels * sizeof(clock_t);  // number of data bytes
  float kernel_time = 10;                   // time the kernel should run in ms
  float elapsed_time;                       // timing variables
  int cuda_device = 0;

  printf("[%s] - Starting...\n", argv[0]);

  // get number of kernels if overridden on the command line
  if (checkCmdLineFlag(argc, (const char **)argv, "nkernels")) {
    nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
    nstreams = nkernels + 1;
  }

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  cuda_device = findCudaDevice(argc, (const char **)argv);

  dpct::device_info deviceProp;
  checkCudaErrors(DPCT_CHECK_ERROR(
      cuda_device = dpct::dev_mgr::instance().current_device_id()));

  checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(
      deviceProp, dpct::dev_mgr::instance().get_device(cuda_device))));

  /*
  DPCT1051:18: SYCL does not support a device property functionally compatible
  with concurrentKernels. It was migrated to true. You may need to adjust the
  value of true for the specific device.
  */
  if ((true == 0)) {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         /*
         DPCT1005:19: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         deviceProp.get_major_version(), deviceProp.get_minor_version(),
         deviceProp.get_max_compute_units());

  // allocate host memory
  clock_t *a = 0;  // pointer to the array data in host memory
  checkCudaErrors(DPCT_CHECK_ERROR(
      a = (clock_t *)sycl::malloc_host(nbytes, dpct::get_in_order_queue())));

  // allocate device memory
  clock_t *d_a = 0;  // pointers to data and init value in the device memory
  checkCudaErrors(DPCT_CHECK_ERROR(d_a = (clock_t *)sycl::malloc_device(
                                       nbytes, dpct::get_in_order_queue())));

  // allocate and initialize an array of stream handles
  dpct::queue_ptr *streams =
      (dpct::queue_ptr *)malloc(nstreams * sizeof(dpct::queue_ptr));

  for (int i = 0; i < nstreams; i++) {
    checkCudaErrors(DPCT_CHECK_ERROR(
        (streams[i]) = dpct::get_current_device().create_queue()));
  }

  // create CUDA event handles
  dpct::event_ptr start_event, stop_event;
  checkCudaErrors(DPCT_CHECK_ERROR(start_event = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop_event = new sycl::event()));

  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  dpct::event_ptr *kernelEvent;
  kernelEvent = (dpct::event_ptr *)malloc(nkernels * sizeof(dpct::event_ptr));

  for (int i = 0; i < nkernels; i++) {
    checkCudaErrors(DPCT_CHECK_ERROR(kernelEvent[i] = new sycl::event()));
  }

  //////////////////////////////////////////////////////////////////////
  // time execution with nkernels streams
  clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
  // the kernel takes more time than the channel reset time on arm archs, so to
  // prevent hangs reduce time_clocks.
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
  clock_t time_clocks =
      (clock_t)(kernel_time * deviceProp.get_max_clock_frequency());
#endif

  dpct::sync_barrier(start_event, &dpct::get_in_order_queue());

  // queue nkernels in separate streams and record when they are done
  for (int i = 0; i < nkernels; ++i) {
    streams[i]->submit([&](sycl::handler &cgh) {
      clock_t *d_a_i_ct0 = &d_a[i];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            clock_block(d_a_i_ct0, time_clocks);
          });
    });
    total_clocks += time_clocks;
    /*
    DPCT1024:20: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        *kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier()));

    // make the last stream wait for the kernel event to be recorded
    checkCudaErrors(DPCT_CHECK_ERROR(
        streams[nstreams - 1]->ext_oneapi_submit_barrier({*kernelEvent[i]})));
  }

  // queue a sum kernel and a copy back to host in the last stream.
  // the commands in this stream get dispatched as soon as all the kernel events
  // have been recorded
  streams[nstreams - 1]->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<clock_t, 1> s_clocks_acc_ct1(sycl::range<1>(32), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          sum(d_a, nkernels, item_ct1,
              s_clocks_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
  checkCudaErrors(
      DPCT_CHECK_ERROR(streams[nstreams - 1]->memcpy(a, d_a, sizeof(clock_t))));

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // in this sample we just wait until the GPU is done
  /*
  DPCT1024:21: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::sync_barrier(stop_event, &dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(stop_event->wait_and_throw()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      elapsed_time = (stop_event->get_profiling_info<
                          sycl::info::event_profiling::command_end>() -
                      start_event->get_profiling_info<
                          sycl::info::event_profiling::command_start>()) /
                     1000000.0f));

  printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
         nkernels * kernel_time / 1000.0f);
  printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
         nkernels, kernel_time / 1000.0f);
  printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

  bool bTestResult = (a[0] > total_clocks);

  // release resources
  for (int i = 0; i < nkernels; i++) {
    dpct::get_current_device().destroy_queue(streams[i]);
    dpct::destroy_event(kernelEvent[i]);
  }

  free(streams);
  free(kernelEvent);

  dpct::destroy_event(start_event);
  dpct::destroy_event(stop_event);
  sycl::free(a, dpct::get_in_order_queue());
  dpct::dpct_free(d_a, dpct::get_in_order_queue());

  if (!bTestResult) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
