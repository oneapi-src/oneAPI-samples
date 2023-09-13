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

#include <chrono>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

// This is a kernel that does no real work but runs at least for a specified
// number of clocks
void clock_block(clock_t *d_o, clock_t clock_count) {
  for (int i = 0; i < 500000; i++) {
    d_o[0] = d_o[0] + i;
  }
}

// Single warp reduction kernel
void sum(clock_t *d_clocks, int N, const sycl::nd_item<3> &item_ct1,
         clock_t *s_clocks) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();

  clock_t my_sum = 0;

  for (int i = item_ct1.get_local_id(2); i < N;
       i += item_ct1.get_local_range(2)) {
    my_sum += d_clocks[i];
  }

  s_clocks[item_ct1.get_local_id(2)] = my_sum;
  item_ct1.barrier();

  for (int i = 16; i > 0; i /= 2) {
    if (item_ct1.get_local_id(2) < i) {
      s_clocks[item_ct1.get_local_id(2)] +=
          s_clocks[item_ct1.get_local_id(2) + i];
    }

    item_ct1.barrier();
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
  sycl::queue q_ct1 = sycl::queue(sycl::default_selector_v);
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
  std::cout << "Device: "
            << q_ct1.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  std::cout
      << "> Detected Compute SM "
      << q_ct1.get_device().get_info<sycl::info::device::version>()
      << " hardware with "
      << q_ct1.get_device().get_info<sycl::info::device::max_compute_units>()
      << " multi-processors" << std::endl;

  if ((true == 0)) {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }

  // allocate host memory
  clock_t *a = 0;  // pointer to the array data in host memory

  a = (clock_t *)sycl::malloc_host(nbytes, q_ct1);

  // allocate device memory
  clock_t *d_a = 0;  // pointers to data and init value in the device memory

  d_a = (clock_t *)sycl::malloc_device(nbytes, q_ct1);

  // allocate and initialize an array of stream handles
  sycl::queue **streams =
      (sycl::queue **)malloc(nstreams * sizeof(sycl::queue *));

  for (int i = 0; i < nstreams; i++) {
    streams[i] = (sycl::queue *)malloc(nstreams * sizeof(sycl::queue));
    *streams[i] =
        sycl::queue(sycl::default_selector_v, dpct::exception_handler);
  }

  // create CUDA event handles
  dpct::event_ptr start_event, stop_event;
  std::chrono::time_point<std::chrono::steady_clock> start_event_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_event_ct1;
  start_event = new sycl::event();
  stop_event = new sycl::event();

  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  dpct::event_ptr *kernelEvent;
  std::chrono::time_point<std::chrono::steady_clock> kernelEvent_ct1_i;
  kernelEvent = (dpct::event_ptr *)malloc(nkernels * sizeof(dpct::event_ptr));

  for (int i = 0; i < nkernels; i++) {
    kernelEvent[i] = new sycl::event();
  }

  //////////////////////////////////////////////////////////////////////
  // time execution with nkernels streams
  clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
  // the kernel takes more time than the channel reset time on arm archs, so to
  // prevent hangs reduce time_clocks.
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
  clock_t time_clocks = (clock_t)(
      kernel_time *
      q_ct1.get_device().get_info<sycl::info::device::max_clock_frequency>());
#endif

  sycl::event stop_event_streams_nstreams_1;
  start_event_ct1 = std::chrono::steady_clock::now();
  *start_event = q_ct1.ext_oneapi_submit_barrier();

  // queue nkernels in separate streams and record when they are done
  for (int i = 0; i < nkernels; ++i) {
    streams[i]->submit([&](sycl::handler &cgh) {
      auto d_a_i_ct0 = &d_a[i];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            clock_block(d_a_i_ct0, time_clocks);
          });
    });
    total_clocks += time_clocks;

    kernelEvent_ct1_i = std::chrono::steady_clock::now();

    *kernelEvent[i] = streams[i]->ext_oneapi_submit_barrier();

    // make the last stream wait for the kernel event to be recorded

    streams[nstreams - 1]->ext_oneapi_submit_barrier({*kernelEvent[i]});
  }

  // queue a sum kernel and a copy back to host in the last stream.
  // the commands in this stream get dispatched as soon as all the kernel events
  // have been recorded
  streams[nstreams - 1]->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<clock_t, 1> s_clocks_acc_ct1(sycl::range<1>(32), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
          sum(d_a, nkernels, item_ct1, s_clocks_acc_ct1.get_pointer());
        });
  });

  stop_event_streams_nstreams_1 =
      streams[nstreams - 1]->memcpy(a, d_a, sizeof(clock_t));

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // in this sample we just wait until the GPU is done

  q_ct1.wait_and_throw();
  stop_event_streams_nstreams_1.wait();
  stop_event_ct1 = std::chrono::steady_clock::now();

  *stop_event = q_ct1.ext_oneapi_submit_barrier();

  elapsed_time =
      std::chrono::duration<float, std::milli>(stop_event_ct1 - start_event_ct1)
          .count();

  printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
         nkernels * kernel_time / 1000.0f);
  printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
         nkernels, kernel_time / 1000.0f);
  printf("Measured time for sample = %.5fs\n", elapsed_time / 1000.0f);

  bool bTestResult = (a[0] > total_clocks);

  // release resources
  for (int i = 0; i < nkernels; i++) {
    free(streams[i]);
  }

  free(streams);
  free(kernelEvent);

  dpct::destroy_event(start_event);
  dpct::destroy_event(stop_event);
  sycl::free(a, q_ct1);
  sycl::free(d_a, q_ct1);

  if (!bTestResult) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
