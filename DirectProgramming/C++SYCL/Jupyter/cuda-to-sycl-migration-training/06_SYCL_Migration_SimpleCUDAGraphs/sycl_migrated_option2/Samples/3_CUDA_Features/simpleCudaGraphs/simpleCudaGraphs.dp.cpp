//=========================================================
// Modifications Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
//=========================================================

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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <helper_cuda.h>
#include <vector>
#include <chrono>
#include <chrono>

using Time = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;
using float_ms = std::chrono::duration<float, ms::period>;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize, const sycl::nd_item<3> &item_ct1,
                       double *tmp) {

  sycl::group<3> cta = item_ct1.get_group();
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  double beta = temp_sum;
  double temp;

  for (int i = tile32.get_local_linear_range() / 2; i > 0;
       i >>= 1) {
    if (tile32.get_local_linear_id() < i) {
      temp = tmp[item_ct1.get_local_linear_id() + i];
      beta += temp;
      tmp[item_ct1.get_local_linear_id()] = beta;
    }
  }

  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() == 0 &&
      item_ct1.get_group(2) < outputSize) {
    beta = 0.0;
    for (int i = 0; i < item_ct1.get_group().get_local_linear_range();
         i += tile32.get_local_linear_range()) {
      beta += tmp[i];
    }
    outputVec[item_ct1.get_group(2)] = beta;
  }
}

void reduceFinal(double *inputVec, double *result,
                            size_t inputSize, const sycl::nd_item<3> &item_ct1,
                            double *tmp) {

  sycl::group<3> cta = item_ct1.get_group();
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  // do reduction in shared mem
  if ((item_ct1.get_local_range(2) >= 512) &&
      (item_ct1.get_local_linear_id() < 256)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 256];
  }

  item_ct1.barrier();

  if ((item_ct1.get_local_range(2) >= 256) &&
      (item_ct1.get_local_linear_id() < 128)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 128];
  }

  item_ct1.barrier();

  if ((item_ct1.get_local_range(2) >= 128) &&
      (item_ct1.get_local_linear_id() < 64)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 64];
  }

  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (item_ct1.get_local_range(2) >= 64) temp_sum +=
        tmp[item_ct1.get_local_linear_id() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      temp_sum +=
          sycl::shift_group_left(tile32, temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (item_ct1.get_local_linear_id() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void myHostNodeCallback(void *data) {
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void syclGraphManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks) try {
   namespace sycl_ext = sycl::ext::oneapi::experimental;
  double result_h = 0.0;
  //use default sycl queue, which is out of order
  sycl::queue q = sycl::queue{sycl::default_selector_v}; 
  if(!q.get_device().has(sycl::aspect::fp64)){
      printf("Double precision isn't supported : %s \nExit\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }
  
  sycl_ext::command_graph graph(q.get_context(), q.get_device());  
  auto nodecpy = graph.add([&](sycl::handler& h){
      h.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);
  }); 
  
  auto nodememset1 = graph.add([&](sycl::handler& h){
      h.fill(outputVec_d, 0, numOfBlocks);
  });

  auto nodememset2 = graph.add([&](sycl::handler& h){
      h.fill(result_d, 0, 1);
  });

  auto nodek1 = graph.add([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
               tmp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
      });
  },  sycl_ext::property::node::depends_on(nodecpy, nodememset1));
   
  
  auto nodek2 = graph.add([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduceFinal(outputVec_d, result_d, numOfBlocks, item_ct1,
                    tmp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
      });
  }, sycl_ext::property::node::depends_on(nodek1, nodememset2));
  auto nodecpy1 = graph.add([&](sycl::handler &cgh) {
      cgh.memcpy(&result_h, result_d, sizeof(double));  
  }, sycl_ext::property::node::depends_on(nodek2));
  auto exec_graph = graph.finalize();
  
  sycl::queue qexec = sycl::queue{sycl::default_selector_v, 
       {sycl::ext::intel::property::queue::no_immediate_command_list()}};
      
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    qexec.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait(); 
    printf("Final reduced sum = %lf\n", result_h);
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << " Exception caught at :" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::cerr << "Exiting..." << std::endl;
  exit(0);
}

void syclGraphCaptureQueue(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) try {
                                      
  namespace sycl_ext = sycl::ext::oneapi::experimental;
  double result_h = 0.0;
  //use default sycl queue, which is out of order
  sycl::queue q = sycl::queue{sycl::default_selector_v}; 
  if(!q.get_device().has(sycl::aspect::fp64)){
      printf("Double precision isn't supported : %s \nExit\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());
      exit(0);
  }

  sycl_ext::command_graph graph(q.get_context(), q.get_device());
  graph.begin_recording(q);
  
  sycl::event ememcpy = q.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);
  sycl::event ememset = q.fill(outputVec_d, 0, numOfBlocks);
  sycl::event ememset1 = q.fill(result_d, 0, 1);
  
  sycl::event ek1 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({ememcpy, ememset});
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
               tmp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
      });
  });
  
  sycl::event ek2 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({ek1, ememset1});
    sycl::local_accessor<double, 1> tmp_acc_ct1(
      sycl::range<1>(THREADS_PER_BLOCK), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        reduceFinal(outputVec_d, result_d, numOfBlocks, item_ct1,
                    tmp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
      });
  });
  
  q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(ek2);
      cgh.memcpy(&result_h, result_d, sizeof(double));  
  });
  graph.end_recording();
  auto exec_graph = graph.finalize();
  
  sycl::queue qexec = sycl::queue{sycl::default_selector_v, 
      {sycl::ext::intel::property::queue::no_immediate_command_list()}};

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    qexec.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait(); 
    printf("Final reduced sum = %lf\n", result_h);
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at :" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::cerr << "Exiting..." << std::endl;
  exit(0);
}

int main(int argc, char **argv) {
  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  sycl::device dev = dpct::get_default_queue().get_device();

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;

  inputVec_h = sycl::malloc_host<float>(size, dpct::get_default_queue());
  inputVec_d = sycl::malloc_device<float>(size, dpct::get_default_queue());
  outputVec_d = sycl::malloc_device<double>(maxBlocks, dpct::get_default_queue());
  result_d = sycl::malloc_device<double>(1, dpct::get_default_queue());

  init_input(inputVec_h, size);

  auto startTimer1 = Time::now();
  syclGraphManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                   maxBlocks);
  auto stopTimer1 = Time::now();
  auto Timer_duration1 =
      std::chrono::duration_cast<float_ms>(stopTimer1 - startTimer1).count();
  printf("Elapsed Time of SYCL Graphs Manual : %f (ms)\n", Timer_duration1);

  auto startTimer2 = Time::now();
  syclGraphCaptureQueue(inputVec_h, inputVec_d, outputVec_d, result_d,
                               size, maxBlocks);
  auto stopTimer2 = Time::now();
  auto Timer_duration2 =
      std::chrono::duration_cast<float_ms>(stopTimer2 - startTimer2).count();
  printf("Elapsed Time SYCL Streamcapture : %f (ms)\n", Timer_duration2);

  sycl::free(inputVec_d, dpct::get_default_queue());
  sycl::free(outputVec_d, dpct::get_default_queue());
  sycl::free(result_d, dpct::get_default_queue());
  sycl::free(inputVec_h, dpct::get_default_queue());
  return EXIT_SUCCESS;
}
