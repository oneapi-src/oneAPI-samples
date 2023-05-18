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
#include <cooperative_groups.h>
#include <helper_cuda.h>
#include <vector>
#include <chrono>

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize, sycl::nd_item<3> item_ct1,
                       double *tmp) {

  auto cta = item_ct1.get_group();
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  double beta = temp_sum;
  double temp;

  for (int i = item_ct1.get_sub_group().get_local_linear_range() / 2; i > 0;
       i >>= 1) {
    if (item_ct1.get_sub_group().get_local_linear_id() < i) {
      temp = tmp[item_ct1.get_local_linear_id() + i];
      beta += temp;
      tmp[item_ct1.get_local_linear_id()] = beta;
    }
    /*
    DPCT1065:2: Consider replacing sycl::sub_group::barrier() with
    sycl::sub_group::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.get_sub_group().barrier();
  }
  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() == 0 &&
      item_ct1.get_group(2) < outputSize) {
    beta = 0.0;
    /*
    DPCT1007:3: Migration of size is not supported.
    */
    for (int i = 0; i < cta.size();
         i += item_ct1.get_sub_group().get_local_linear_range()) {
      beta += tmp[i];
    }
    outputVec[item_ct1.get_group(2)] = beta;
  }
}

void reduceFinal(double *inputVec, double *result,
                            size_t inputSize, sycl::nd_item<3> item_ct1,
                            double *tmp) {

  auto cta = item_ct1.get_group();
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

  /*
  DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  // do reduction in shared mem
  if ((item_ct1.get_local_range(2) >= 512) &&
      (item_ct1.get_local_linear_id() < 256)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 256];
  }

  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if ((item_ct1.get_local_range(2) >= 256) &&
      (item_ct1.get_local_linear_id() < 128)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 128];
  }

  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if ((item_ct1.get_local_range(2) >= 128) &&
      (item_ct1.get_local_linear_id() < 64)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 64];
  }

  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (item_ct1.get_local_range(2) >= 64) temp_sum +=
        tmp[item_ct1.get_local_linear_id() + 32];
    // Reduce final warp using shuffle
    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      /*
      DPCT1007:8: Migration of shfl_down is not supported.
      */
      temp_sum += tile32.shfl_down(temp_sum, offset);
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
  // checkCudaErrors(tmp->status);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void cudaGraphsManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks) {
  dpct::queue_ptr streamForGraph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
  double result_h = 0.0;

  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (streamForGraph = dpct::get_current_device().create_queue(), 0));

  cudaKernelNodeParams kernelNodeParams = { 0sycl ::range<3>(1, 1, 1);
  dpct::pitched_data memcpyParams_from_data_ct1, memcpyParams_to_data_ct1;
  sycl::id<3> memcpyParams_from_pos_ct1(0, 0, 0),
      memcpyParams_to_pos_ct1(0, 0, 0);
  sycl::range<3> memcpyParams_size_ct1(1, 1, 1);
  dpct::memcpy_direction memcpyParams_direction_ct1;
  cudaMemsetParams memsetParams = {0};

  memcpyParams_from_data_ct1 = NULL->to_pitched_data();
  memcpyParams_from_pos_ct1 = sycl::id<3>(0, 0, 0);
  memcpyParams_from_data_ct1 =
      dpct::pitched_data(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams_to_data_ct1 = NULL->to_pitched_data();
  memcpyParams_to_pos_ct1 = sycl::id<3>(0, 0, 0);
  memcpyParams_to_data_ct1 =
      dpct::pitched_data(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams_size_ct1 = sycl::range<3>(sizeof(float) * inputSize, 1, 1);
  memcpyParams_direction_ct1 = dpct::host_to_device;

  memsetParams.dst = (void *)outputVec_d;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(float);  // elementSize can be max 4 bytes
  memsetParams.width = numOfBlocks * 2;
  memsetParams.height = 1;

  /*
  DPCT1007:32: Migration of cudaGraphCreate is not supported.
  */
  checkCudaErrors(cudaGraphCreate(&graph, 0));
  /*
  DPCT1007:33: Migration of cudaGraphAddMemcpyNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
  /*
  DPCT1007:34: Migration of cudaGraphAddMemsetNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

  nodeDependencies.push_back(memsetNode);
  nodeDependencies.push_back(memcpyNode);

  void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize,
                         &numOfBlocks};

  kernelNodeParams.func = (void *)reduce;
  kernelNodeParams.gridDim = sycl::range<3>(1, 1, numOfBlocks);
  kernelNodeParams.blockDim = sycl::range<3>(1, 1, THREADS_PER_BLOCK);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = (void **)kernelArgs;
  kernelNodeParams.extra = NULL;

  /*
  DPCT1007:35: Migration of cudaGraphAddKernelNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = result_d;
  memsetParams.value = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;
  /*
  DPCT1007:36: Migration of cudaGraphAddMemsetNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

  nodeDependencies.push_back(memsetNode);

  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = (void *)reduceFinal;
  kernelNodeParams.gridDim = sycl::range<3>(1, 1, 1);
  kernelNodeParams.blockDim = sycl::range<3>(1, 1, THREADS_PER_BLOCK);
  kernelNodeParams.sharedMemBytes = 0;
  void *kernelArgs2[3] = {(void *)&outputVec_d, (void *)&result_d,
                          &numOfBlocks};
  kernelNodeParams.kernelParams = kernelArgs2;
  kernelNodeParams.extra = NULL;

  /*
  DPCT1007:37: Migration of cudaGraphAddKernelNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  memset(&memcpyParams, 0, sizeof(memcpyParams));

  memcpyParams_from_data_ct1 = NULL->to_pitched_data();
  memcpyParams_from_pos_ct1 = sycl::id<3>(0, 0, 0);
  memcpyParams_from_data_ct1 =
      dpct::pitched_data(result_d, sizeof(double), 1, 1);
  memcpyParams_to_data_ct1 = NULL->to_pitched_data();
  memcpyParams_to_pos_ct1 = sycl::id<3>(0, 0, 0);
  memcpyParams_to_data_ct1 =
      dpct::pitched_data(&result_h, sizeof(double), 1, 1);
  memcpyParams_size_ct1 = sycl::range<3>(sizeof(double), 1, 1);
  memcpyParams_direction_ct1 = dpct::device_to_host;
  /*
  DPCT1007:38: Migration of cudaGraphAddMemcpyNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);

  cudaGraphNode_t hostNode;
  cudaHostNodeParams hostParams = {0};
  hostParams.fn = myHostNodeCallback;
  callBackData_t hostFnData;
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsManual";
  hostParams.userData = &hostFnData;

  /*
  DPCT1007:39: Migration of cudaGraphAddHostNode is not supported.
  */
  checkCudaErrors(cudaGraphAddHostNode(&hostNode, graph,
                                       nodeDependencies.data(),
                                       nodeDependencies.size(), &hostParams));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  /*
  DPCT1007:40: Migration of cudaGraphGetNodes is not supported.
  */
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

  cudaGraphExec_t graphExec;
  /*
  DPCT1007:41: Migration of cudaGraphInstantiate is not supported.
  */
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  /*
  DPCT1007:42: Migration of cudaGraphClone is not supported.
  */
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  /*
  DPCT1007:43: Migration of cudaGraphInstantiate is not supported.
  */
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    /*
    DPCT1007:44: Migration of cudaGraphLaunch is not supported.
    */
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }

  /*
  DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((streamForGraph->wait(), 0));

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    /*
    DPCT1007:46: Migration of cudaGraphLaunch is not supported.
    */
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }
  /*
  DPCT1003:47: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((streamForGraph->wait(), 0));

  /*
  DPCT1007:48: Migration of cudaGraphExecDestroy is not supported.
  */
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  /*
  DPCT1007:49: Migration of cudaGraphExecDestroy is not supported.
  */
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  /*
  DPCT1007:50: Migration of cudaGraphDestroy is not supported.
  */
  checkCudaErrors(cudaGraphDestroy(graph));
  /*
  DPCT1007:51: Migration of cudaGraphDestroy is not supported.
  */
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  /*
  DPCT1003:52: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (dpct::get_current_device().destroy_queue(streamForGraph), 0));
}

void cudaGraphsUsingStreamCapture(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
  dpct::queue_ptr stream1, stream2, stream3, streamForGraph;
  dpct::event_ptr forkStreamEvent, memsetEvent1, memsetEvent2;
  std::chrono::time_point<std::chrono::steady_clock> forkStreamEvent_ct1;
  std::chrono::time_point<std::chrono::steady_clock> memsetEvent1_ct1;
  std::chrono::time_point<std::chrono::steady_clock> memsetEvent2_ct1;
  cudaGraph_t graph;
  double result_h = 0.0;

  /*
  DPCT1003:53: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream1 = dpct::get_current_device().create_queue(), 0));
  /*
  DPCT1003:54: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream2 = dpct::get_current_device().create_queue(), 0));
  /*
  DPCT1003:55: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream3 = dpct::get_current_device().create_queue(), 0));
  /*
  DPCT1003:56: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (streamForGraph = dpct::get_current_device().create_queue(), 0));

  /*
  DPCT1003:57: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((forkStreamEvent = new sycl::event(), 0));
  /*
  DPCT1003:58: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((memsetEvent1 = new sycl::event(), 0));
  /*
  DPCT1003:59: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((memsetEvent2 = new sycl::event(), 0));

  /*
  DPCT1027:60: The call to cudaStreamBeginCapture was replaced with 0 because
  SYCL currently does not support capture operations on queues.
  */
  checkCudaErrors(0);

  /*
  DPCT1012:61: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:62: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  forkStreamEvent_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors((*forkStreamEvent = stream1->ext_oneapi_submit_barrier(), 0));
  /*
  DPCT1003:63: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream2->ext_oneapi_submit_barrier({*forkStreamEvent}), 0));
  /*
  DPCT1003:64: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream3->ext_oneapi_submit_barrier({*forkStreamEvent}), 0));

  /*
  DPCT1003:65: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (stream1->memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize), 0));

  /*
  DPCT1003:66: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (stream2->memset(outputVec_d, 0, sizeof(double) * numOfBlocks), 0));

  /*
  DPCT1012:67: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:68: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  memsetEvent1_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors((*memsetEvent1 = stream2->ext_oneapi_submit_barrier(), 0));

  /*
  DPCT1003:69: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream3->memset(result_d, 0, sizeof(double)), 0));
  /*
  DPCT1012:70: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:71: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  memsetEvent2_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors((*memsetEvent2 = stream3->ext_oneapi_submit_barrier(), 0));

  /*
  DPCT1003:72: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream1->ext_oneapi_submit_barrier({*memsetEvent1}), 0));

  /*
  DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream1->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(sycl::range<1>(512 /*512*/),
                                                cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                              sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                          sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item_ct1,
                 tmp_acc_ct1.get_pointer());
        });
  });

  /*
  DPCT1003:73: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream1->ext_oneapi_submit_barrier({*memsetEvent2}), 0));

  /*
  DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream1->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> tmp_acc_ct1(sycl::range<1>(512 /*512*/),
                                                cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                                       sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                     [=](sycl::nd_item<3> item_ct1)
                         [[intel::reqd_sub_group_size(32)]] {
                           reduceFinal(outputVec_d, result_d, numOfBlocks,
                                       item_ct1, tmp_acc_ct1.get_pointer());
                         });
  });
  /*
  DPCT1003:74: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stream1->memcpy(&result_h, result_d, sizeof(double)), 0));

  callBackData_t hostFnData = {0};
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsUsingStreamCapture";
  cudaHostFn_t fn = myHostNodeCallback;
  /*
  DPCT1007:75: Migration of cudaLaunchHostFunc is not supported.
  */
  checkCudaErrors(cudaLaunchHostFunc(stream1, fn, &hostFnData));
  /*
  DPCT1027:76: The call to cudaStreamEndCapture was replaced with 0 because SYCL
  currently does not support capture operations on queues.
  */
  checkCudaErrors(0);

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  /*
  DPCT1007:77: Migration of cudaGraphGetNodes is not supported.
  */
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created using stream capture API = %zu\n",
         numNodes);

  cudaGraphExec_t graphExec;
  /*
  DPCT1007:78: Migration of cudaGraphInstantiate is not supported.
  */
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  /*
  DPCT1007:79: Migration of cudaGraphClone is not supported.
  */
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  /*
  DPCT1007:80: Migration of cudaGraphInstantiate is not supported.
  */
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    /*
    DPCT1007:81: Migration of cudaGraphLaunch is not supported.
    */
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }

  /*
  DPCT1003:82: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((streamForGraph->wait(), 0));

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    /*
    DPCT1007:83: Migration of cudaGraphLaunch is not supported.
    */
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }

  /*
  DPCT1003:84: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((streamForGraph->wait(), 0));

  /*
  DPCT1007:85: Migration of cudaGraphExecDestroy is not supported.
  */
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  /*
  DPCT1007:86: Migration of cudaGraphExecDestroy is not supported.
  */
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  /*
  DPCT1007:87: Migration of cudaGraphDestroy is not supported.
  */
  checkCudaErrors(cudaGraphDestroy(graph));
  /*
  DPCT1007:88: Migration of cudaGraphDestroy is not supported.
  */
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  /*
  DPCT1003:89: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_current_device().destroy_queue(stream1), 0));
  /*
  DPCT1003:90: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_current_device().destroy_queue(stream2), 0));
  /*
  DPCT1003:91: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (dpct::get_current_device().destroy_queue(streamForGraph), 0));
}

int main(int argc, char **argv) {
  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  // This will pick the best possible CUDA capable device
  int devID = findCudaDevice(argc, (const char **)argv);

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;

  /*
  DPCT1003:92: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (inputVec_h = sycl::malloc_host<float>(size, dpct::get_default_queue()),
       0));
  /*
  DPCT1003:93: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (inputVec_d = sycl::malloc_device<float>(size, dpct::get_default_queue()),
       0));
  /*
  DPCT1003:94: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((outputVec_d = sycl::malloc_device<double>(
                       maxBlocks, dpct::get_default_queue()),
                   0));
  /*
  DPCT1003:95: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((
      result_d = sycl::malloc_device<double>(1, dpct::get_default_queue()), 0));

  init_input(inputVec_h, size);

  cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                   maxBlocks);
  cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d,
                               size, maxBlocks);

  /*
  DPCT1003:96: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(inputVec_d, dpct::get_default_queue()), 0));
  /*
  DPCT1003:97: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(outputVec_d, dpct::get_default_queue()), 0));
  /*
  DPCT1003:98: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(result_d, dpct::get_default_queue()), 0));
  /*
  DPCT1003:99: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(inputVec_h, dpct::get_default_queue()), 0));
  return EXIT_SUCCESS;
}
