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
 * This sample evaluates fair call price for a
 * given set of European options using Monte Carlo approach.
 * See supplied whitepaper for more explanations.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_functions.h>  // Helper functions (utilities, parsing, timing)
#include <helper_cuda.h>  // helper functions (cuda error checking and initialization)
#include <multithreading.h>

#include "MonteCarlo_common.h"
#include <chrono>

int *pArgc = NULL;
char **pArgv = NULL;

#ifdef WIN32
#define strcasecmp _strcmpi
#endif

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

/// Utility function to tweak problem size for small GPUs
int adjustProblemSize(int GPU_N, int default_nOptions) {
  int nOptions = default_nOptions;

  // select problem size
  for (int i = 0; i < GPU_N; i++) {
    dpct::device_info deviceProp;
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::dev_mgr::instance().get_device(i).get_device_info(deviceProp)));

    int cudaCores = _ConvertSMVer2Cores(deviceProp.get_major_version(),
                                        deviceProp.get_minor_version()) *
                    deviceProp.get_max_compute_units();

    if (cudaCores <= 32) {
      nOptions = (nOptions < cudaCores / 2 ? nOptions : cudaCores / 2);
    }
  }

  return nOptions;
}

int adjustGridSize(int GPUIndex, int defaultGridSize) {
  dpct::device_info deviceProp;
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dev_mgr::instance().get_device(GPUIndex).get_device_info(
          deviceProp)));
  int maxGridSize = deviceProp.get_max_compute_units() * 40;
  return ((defaultGridSize > maxGridSize) ? maxGridSize : defaultGridSize);
}

///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(TOptionValue &callValue, TOptionData optionData,
                              float *h_Random, int pathN);

// Black-Scholes formula for call options
extern "C" void BlackScholesCall(float &CallResult, TOptionData optionData);

////////////////////////////////////////////////////////////////////////////////
// GPU-driving host thread
////////////////////////////////////////////////////////////////////////////////
// Timer
StopWatchInterface **hTimer = NULL;

static CUT_THREADPROC solverThread(TOptionPlan *plan) {
  // Init GPU

  checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan->device)));

  dpct::device_info deviceProp;
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                       .get_device(plan->device)
                                       .get_device_info(deviceProp)));

  // Start the timer
  sdkStartTimer(&hTimer[plan->device]);

  // Allocate intermediate memory for MC integrator and initialize
  // RNG states
  initMonteCarloGPU(plan);

  // Main computation
  MonteCarloGPU(plan);

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

  // Stop the timer
  sdkStopTimer(&hTimer[plan->device]);

  // Shut down this GPU
  closeMonteCarloGPU(plan);

  dpct::get_default_queue().wait();

  printf("solverThread() finished - GPU Device %d: %s\n", plan->device,
         deviceProp.get_name());

  CUT_THREADEND;
}

static void multiSolver(TOptionPlan *plan, int nPlans) {
  // allocate and initialize an array of stream handles
  dpct::queue_ptr *streams =
      (dpct::queue_ptr *)malloc(nPlans * sizeof(dpct::queue_ptr));
  dpct::event_ptr *events =
      (dpct::event_ptr *)malloc(nPlans * sizeof(dpct::event_ptr));
  std::chrono::time_point<std::chrono::steady_clock> events_ct1_i;

  for (int i = 0; i < nPlans; i++) {

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan[i].device)));
    checkCudaErrors(DPCT_CHECK_ERROR(
        (streams[i]) = dpct::get_current_device().create_queue()));
    checkCudaErrors(DPCT_CHECK_ERROR(events[i] = new sycl::event()));
  }

  // Init Each GPU
  // In CUDA 4.0 we can call cudaSetDevice multiple times to target each device
  // Set the device desired, then perform initializations on that device

  for (int i = 0; i < nPlans; i++) {
    // set the target device to perform initialization on

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan[i].device)));

    dpct::device_info deviceProp;
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                         .get_device(plan[i].device)
                                         .get_device_info(deviceProp)));

    // Allocate intermediate memory for MC integrator
    // and initialize RNG state
    initMonteCarloGPU(&plan[i]);
  }

  for (int i = 0; i < nPlans; i++) {

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan[i].device)));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  }

  // Start the timer
  sdkResetTimer(&hTimer[0]);
  sdkStartTimer(&hTimer[0]);

  for (int i = 0; i < nPlans; i++) {

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan[i].device)));

    // Main computations
    MonteCarloGPU(&plan[i], streams[i]);

    events_ct1_i = std::chrono::steady_clock::now();
    checkCudaErrors(
        DPCT_CHECK_ERROR(*events[i] = streams[i]->ext_oneapi_submit_barrier()));
  }

  for (int i = 0; i < nPlans; i++) {

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan[i].device)));
    events[i]->wait_and_throw();
  }

  // Stop the timer
  sdkStopTimer(&hTimer[0]);

  for (int i = 0; i < nPlans; i++) {

    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(plan[i].device)));
    closeMonteCarloGPU(&plan[i]);
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(streams[i])));
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(events[i])));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
#define DO_CPU
#undef DO_CPU

#define PRINT_RESULTS
#undef PRINT_RESULTS

void usage() {
  printf("--method=[threaded,streamed] --scaling=[strong,weak] [--help]\n");
  printf("Method=threaded: 1 CPU thread for each GPU     [default]\n");
  printf(
      "       streamed: 1 CPU thread handles all GPUs (requires CUDA 4.0 or "
      "newer)\n");
  printf("Scaling=strong : constant problem size\n");
  printf(
      "        weak   : problem size scales with number of available GPUs "
      "[default]\n");
}

int main(int argc, char **argv) {
  char *multiMethodChoice = NULL;
  char *scalingChoice = NULL;
  bool use_threads = true;
  bool bqatest = false;
  bool strongScaling = false;

  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", argv[0]);

  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    bqatest = true;
  }

  getCmdLineArgumentString(argc, (const char **)argv, "method",
                           &multiMethodChoice);
  getCmdLineArgumentString(argc, (const char **)argv, "scaling",
                           &scalingChoice);

  if (checkCmdLineFlag(argc, (const char **)argv, "h") ||
      checkCmdLineFlag(argc, (const char **)argv, "help")) {
    usage();
    exit(EXIT_SUCCESS);
  }

  if (multiMethodChoice == NULL) {
    use_threads = false;
  } else {
    if (!strcasecmp(multiMethodChoice, "threaded")) {
      use_threads = true;
    } else {
      use_threads = false;
    }
  }

  if (use_threads == false) {
    printf("Using single CPU thread for multiple GPUs\n");
  }

  if (scalingChoice == NULL) {
    strongScaling = false;
  } else {
    if (!strcasecmp(scalingChoice, "strong")) {
      strongScaling = true;
    } else {
      strongScaling = false;
    }
  }

  // GPU number present in the system
  int GPU_N;
  checkCudaErrors(
      DPCT_CHECK_ERROR(GPU_N = dpct::dev_mgr::instance().device_count()));
  int nOptions = 8 * 1024;

  nOptions = adjustProblemSize(GPU_N, nOptions);

  // select problem size
  int scale = (strongScaling) ? 1 : GPU_N;
  int OPT_N = nOptions * scale;
  int PATH_N = 262144;

  // initialize the timers
  hTimer = new StopWatchInterface *[GPU_N];

  for (int i = 0; i < GPU_N; i++) {
    sdkCreateTimer(&hTimer[i]);
    sdkResetTimer(&hTimer[i]);
  }

  // Input data array
  TOptionData *optionData = new TOptionData[OPT_N];
  // Final GPU MC results
  TOptionValue *callValueGPU = new TOptionValue[OPT_N];
  //"Theoretical" call values by Black-Scholes formula
  float *callValueBS = new float[OPT_N];
  // Solver config
  TOptionPlan *optionSolver = new TOptionPlan[GPU_N];
  // OS thread ID
  CUTThread *threadID = new CUTThread[GPU_N];

  int gpuBase, gpuIndex;
  int i;

  float time;

  double delta, ref, sumDelta, sumRef, sumReserve;

  printf("MonteCarloMultiGPU\n");
  printf("==================\n");
  printf("Parallelization method  = %s\n",
         use_threads ? "threaded" : "streamed");
  printf("Problem scaling         = %s\n", strongScaling ? "strong" : "weak");
  printf("Number of GPUs          = %d\n", GPU_N);
  printf("Total number of options = %d\n", OPT_N);
  printf("Number of paths         = %d\n", PATH_N);

  printf("main(): generating input data...\n");
  srand(123);

  for (i = 0; i < OPT_N; i++) {
    optionData[i].S = randFloat(5.0f, 50.0f);
    optionData[i].X = randFloat(10.0f, 25.0f);
    optionData[i].T = randFloat(1.0f, 5.0f);
    optionData[i].R = 0.06f;
    optionData[i].V = 0.10f;
    callValueGPU[i].Expected = -1.0f;
    callValueGPU[i].Confidence = -1.0f;
  }

  printf("main(): starting %i host threads...\n", GPU_N);

  // Get option count for each GPU
  for (i = 0; i < GPU_N; i++) {
    optionSolver[i].optionCount = OPT_N / GPU_N;
  }

  // Take into account cases with "odd" option counts
  for (i = 0; i < (OPT_N % GPU_N); i++) {
    optionSolver[i].optionCount++;
  }

  // Assign GPU option ranges
  gpuBase = 0;

  for (i = 0; i < GPU_N; i++) {
    optionSolver[i].device = i;
    optionSolver[i].optionData = optionData + gpuBase;
    optionSolver[i].callValue = callValueGPU + gpuBase;
    optionSolver[i].pathN = PATH_N;
    optionSolver[i].gridSize =
        adjustGridSize(optionSolver[i].device, optionSolver[i].optionCount);
    gpuBase += optionSolver[i].optionCount;
  }

  if (use_threads || bqatest) {
    // Start CPU thread for each GPU
    for (gpuIndex = 0; gpuIndex < GPU_N; gpuIndex++) {
      threadID[gpuIndex] = cutStartThread((CUT_THREADROUTINE)solverThread,
                                          &optionSolver[gpuIndex]);
    }

    printf("main(): waiting for GPU results...\n");
    cutWaitForThreads(threadID, GPU_N);

    printf("main(): GPU statistics, threaded\n");

    for (i = 0; i < GPU_N; i++) {
      dpct::device_info deviceProp;
      checkCudaErrors(DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                           .get_device(optionSolver[i].device)
                                           .get_device_info(deviceProp)));
      printf("GPU Device #%i: %s\n", optionSolver[i].device,
             deviceProp.get_name());
      printf("Options         : %i\n", optionSolver[i].optionCount);
      printf("Simulation paths: %i\n", optionSolver[i].pathN);
      time = sdkGetTimerValue(&hTimer[i]);
      printf("Total time (ms.): %f\n", time);
      printf("Options per sec.: %f\n", OPT_N / (time * 0.001));
    }

    printf("main(): comparing Monte Carlo and Black-Scholes results...\n");
    sumDelta = 0;
    sumRef = 0;
    sumReserve = 0;

    for (i = 0; i < OPT_N; i++) {
      BlackScholesCall(callValueBS[i], optionData[i]);
      delta = fabs(callValueBS[i] - callValueGPU[i].Expected);
      ref = callValueBS[i];
      sumDelta += delta;
      sumRef += fabs(ref);

      if (delta > 1e-6) {
        sumReserve += callValueGPU[i].Confidence / delta;
      }

#ifdef PRINT_RESULTS
      printf("BS: %f; delta: %E\n", callValueBS[i], delta);
#endif
    }

    sumReserve /= OPT_N;
  }

  if (!use_threads || bqatest) {
    multiSolver(optionSolver, GPU_N);

    printf("main(): GPU statistics, streamed\n");

    for (i = 0; i < GPU_N; i++) {
      dpct::device_info deviceProp;
      checkCudaErrors(DPCT_CHECK_ERROR(dpct::dev_mgr::instance()
                                           .get_device(optionSolver[i].device)
                                           .get_device_info(deviceProp)));
      printf("GPU Device #%i: %s\n", optionSolver[i].device,
             deviceProp.get_name());
      printf("Options         : %i\n", optionSolver[i].optionCount);
      printf("Simulation paths: %i\n", optionSolver[i].pathN);
    }

    time = sdkGetTimerValue(&hTimer[0]);
    printf("\nTotal time (ms.): %f\n", time);
    printf("\tNote: This is elapsed time for all to compute.\n");
    printf("Options per sec.: %f\n", OPT_N / (time * 0.001));

    printf("main(): comparing Monte Carlo and Black-Scholes results...\n");
    sumDelta = 0;
    sumRef = 0;
    sumReserve = 0;

    for (i = 0; i < OPT_N; i++) {
      BlackScholesCall(callValueBS[i], optionData[i]);
      delta = fabs(callValueBS[i] - callValueGPU[i].Expected);
      ref = callValueBS[i];
      sumDelta += delta;
      sumRef += fabs(ref);

      if (delta > 1e-6) {
        sumReserve += callValueGPU[i].Confidence / delta;
      }

#ifdef PRINT_RESULTS
      printf("BS: %f; delta: %E\n", callValueBS[i], delta);
#endif
    }

    sumReserve /= OPT_N;
  }

#ifdef DO_CPU
  printf("main(): running CPU MonteCarlo...\n");
  TOptionValue callValueCPU;
  sumDelta = 0;
  sumRef = 0;

  for (i = 0; i < OPT_N; i++) {
    MonteCarloCPU(callValueCPU, optionData[i], NULL, PATH_N);
    delta = fabs(callValueCPU.Expected - callValueGPU[i].Expected);
    ref = callValueCPU.Expected;
    sumDelta += delta;
    sumRef += fabs(ref);
    printf("Exp : %f | %f\t", callValueCPU.Expected, callValueGPU[i].Expected);
    printf("Conf: %f | %f\n", callValueCPU.Confidence,
           callValueGPU[i].Confidence);
  }

  printf("L1 norm: %E\n", sumDelta / sumRef);
#endif

  printf("Shutting down...\n");

  for (int i = 0; i < GPU_N; i++) {
    sdkStartTimer(&hTimer[i]);
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(i)));
  }

  delete[] optionSolver;
  delete[] callValueBS;
  delete[] callValueGPU;
  delete[] optionData;
  delete[] threadID;
  delete[] hTimer;

  printf("Test Summary...\n");
  printf("L1 norm        : %E\n", sumDelta / sumRef);
  printf("Average reserve: %f\n", sumReserve);
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");
  printf(sumReserve > 1.0f ? "Test passed\n" : "Test failed!\n");
  exit(sumReserve > 1.0f ? EXIT_SUCCESS : EXIT_FAILURE);
}
