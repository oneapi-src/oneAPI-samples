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
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>

#include <CL/sycl.hpp>
#include <chrono>
#include <dpct/dpct.hpp>

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
void clock_block(clock_t *d_o, clock_t clock_count, sycl::nd_item<3> item_ct1) {
  // int i = 0;
  for (int i = item_ct1.get_local_id(2); i < 500000;
       i += item_ct1.get_local_range(2)) {
    d_o[0] = d_o[0] + i;
  }
}

// Single warp reduction kernel
void sum(clock_t *d_clocks, int N, sycl::nd_item<3> item_ct1,
         clock_t *s_clocks) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();

  clock_t my_sum = 0;

  for (int i = item_ct1.get_local_id(2); i < N;
       i += item_ct1.get_local_range(2)) {
    my_sum += d_clocks[i];
  }

  s_clocks[item_ct1.get_local_id(2)] = my_sum;
  cg::sync(cta);

  for (int i = 16; i > 0; i /= 2) {
    if (item_ct1.get_local_id(2) < i) {
      s_clocks[item_ct1.get_local_id(2)] +=
          s_clocks[item_ct1.get_local_id(2) + i];
    }

    cg::sync(cta);
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  checkCudaErrors(cuda_device = dpct::dev_mgr::instance().current_device_id());

  /*
  DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::dev_mgr::instance()
                       .get_device(cuda_device)
                       .get_device_info(deviceProp),
                   0));

  /*
  DPCT1051:4: SYCL does not support the device property that would be
  functionally compatible with concurrentKernels. It was migrated to true. You
  may need to rewrite the code.
  */
  if ((true == 0)) {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         /*
         DPCT1005:5: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         deviceProp.get_major_version(), deviceProp.get_minor_version(),
         deviceProp.get_max_compute_units());

  // allocate host memory
  clock_t *a = 0;  // pointer to the array data in host memory
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((a = (clock_t *)sycl::malloc_host(nbytes, q_ct1), 0));

  // allocate device memory
  clock_t *d_a = 0;  // pointers to data and init value in the device memory
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_a = (clock_t *)sycl::malloc_device(nbytes, q_ct1), 0));

  // allocate and initialize an array of stream handles
  sycl::queue **streams =
      (sycl::queue **)malloc(nstreams * sizeof(sycl::queue *));

  for (int i = 0; i < nstreams; i++) {
    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    checkCudaErrors(((streams[i]) = dev_ct1.create_queue(), 0));
  }

  // create CUDA event handles
  sycl::event start_event, stop_event;
  std::chrono::time_point<std::chrono::steady_clock> start_event_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_event_ct1;
  /*
  DPCT1027:9: The call to cudaEventCreate was replaced with 0 because this call
  is redundant in SYCL.
  */
  checkCudaErrors(0);
  /*
  DPCT1027:10: The call to cudaEventCreate was replaced with 0 because this call
  is redundant in SYCL.
  */
  checkCudaErrors(0);
  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  sycl::event *kernelEvent;
  std::chrono::time_point<std::chrono::steady_clock> kernelEvent_ct1_i;
  kernelEvent = new sycl::event[nkernels];

  for (int i = 0; i < nkernels; i++) {
    checkCudaErrors(
        /*
        DPCT1027:11: The call to cudaEventCreateWithFlags was replaced with 0
        because this call is redundant in SYCL.
        */
        0);
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

  /*
  DPCT1012:0: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  sycl::event stop_event_streams_nstreams_1;
  start_event_ct1 = std::chrono::steady_clock::now();
  start_event = q_ct1.ext_oneapi_submit_barrier();

  // queue nkernels in separate streams and record when they are done
  for (int i = 0; i < nkernels; ++i) {
    streams[i]->submit([&](sycl::handler &cgh) {
      auto d_a_i_ct0 = &d_a[i];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            clock_block(d_a_i_ct0, time_clocks, item_ct1);
          });
    });
    total_clocks += time_clocks;
    /*
    DPCT1012:12: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024:13: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    kernelEvent_ct1_i = std::chrono::steady_clock::now();
    checkCudaErrors(
        (kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier(), 0));

    // make the last stream wait for the kernel event to be recorded
    checkCudaErrors(
        /*
        DPCT1003:14: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        (kernelEvent[i] =
             streams[nstreams - 1]->ext_oneapi_submit_barrier({kernelEvent[i]}),
         0));
  }
  // queue a sum kernel and a copy back to host in the last stream.
  // the commands in this stream get dispatched as soon as all the kernel events
  // have been recorded
  streams[nstreams - 1]->submit([&](sycl::handler &cgh) {
    sycl::accessor<clock_t, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        s_clocks_acc_ct1(sycl::range<1>(32), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          sum(d_a, nkernels, item_ct1, s_clocks_acc_ct1.get_pointer());
        });
  });
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stop_event_streams_nstreams_1 =
                       streams[nstreams - 1]->memcpy(a, d_a, sizeof(clock_t)),
                   0));

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // in this sample we just wait until the GPU is done
  /*
  DPCT1012:16: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:17: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  dpct::get_current_device().queues_wait_and_throw();
  stop_event_streams_nstreams_1.wait();
  stop_event_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors((stop_event = q_ct1.ext_oneapi_submit_barrier(), 0));
  checkCudaErrors(0);
  /*
  DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((elapsed_time = std::chrono::duration<float, std::milli>(
                                      stop_event_ct1 - start_event_ct1)
                                      .count(),
                   0));

  printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
         nkernels * kernel_time / 1000.0f);
  printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
         nkernels, kernel_time / 1000.0f);
  printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

  bool bTestResult = (a[0] > total_clocks);

  // release resources
  for (int i = 0; i < nkernels; i++) {
    dev_ct1.destroy_queue(streams[i]);
    /*
    DPCT1026:19: The call to cudaEventDestroy was removed because this call is
    redundant in SYCL.
    */
  }

  free(streams);
  delete[] kernelEvent;

  /*
  DPCT1026:1: The call to cudaEventDestroy was removed because this call is
  redundant in SYCL.
  */
  /*
  DPCT1026:2: The call to cudaEventDestroy was removed because this call is
  redundant in SYCL.
  */
  sycl::free(a, q_ct1);
  sycl::free(d_a, q_ct1);

  if (!bTestResult) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
