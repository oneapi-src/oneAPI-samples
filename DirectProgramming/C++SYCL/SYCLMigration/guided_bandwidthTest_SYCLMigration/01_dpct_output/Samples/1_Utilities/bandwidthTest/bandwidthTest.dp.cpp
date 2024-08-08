#define DPCT_COMPAT_RT_VERSION 12020
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
 * This is a simple test program to measure the memcopy bandwidth of the GPU.
 * It can measure device to device copy bandwidth, host to device copy bandwidth
 * for pageable and pinned memory, and device to host copy bandwidth for
 * pageable and pinned memory.
 *
 * Usage:
 * ./bandwidthTest [option]...
 */

// CUDA runtime
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// includes
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

#include <cassert>
#include <iostream>
#include <memory>

static const char *sSDKsample = "CUDA Bandwidth Test";

// defines, project
#define MEMCOPY_ITERATIONS 100
#define DEFAULT_SIZE (32 * (1e6))      // 32 M
#define DEFAULT_INCREMENT (4 * (1e6))  // 4 M
#define CACHE_CLEAR_SIZE (16 * (1e6))  // 16 M

// shmoo mode defines
#define SHMOO_MEMSIZE_MAX (64 * (1e6))       // 64 M
#define SHMOO_MEMSIZE_START (1e3)            // 1 KB
#define SHMOO_INCREMENT_1KB (1e3)            // 1 KB
#define SHMOO_INCREMENT_2KB (2 * 1e3)        // 2 KB
#define SHMOO_INCREMENT_10KB (10 * (1e3))    // 10KB
#define SHMOO_INCREMENT_100KB (100 * (1e3))  // 100 KB
#define SHMOO_INCREMENT_1MB (1e6)            // 1 MB
#define SHMOO_INCREMENT_2MB (2 * 1e6)        // 2 MB
#define SHMOO_INCREMENT_4MB (4 * 1e6)        // 4 MB
#define SHMOO_LIMIT_20KB (20 * (1e3))        // 20 KB
#define SHMOO_LIMIT_50KB (50 * (1e3))        // 50 KB
#define SHMOO_LIMIT_100KB (100 * (1e3))      // 100 KB
#define SHMOO_LIMIT_1MB (1e6)                // 1 MB
#define SHMOO_LIMIT_16MB (16 * 1e6)          // 16 MB
#define SHMOO_LIMIT_32MB (32 * 1e6)          // 32 MB

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

// enums, project
enum testMode { QUICK_MODE, RANGE_MODE, SHMOO_MODE };
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };

const char *sMemoryCopyKind[] = {"Device to Host", "Host to Device",
                                 "Device to Device", NULL};

const char *sMemoryMode[] = {"PINNED", "PAGEABLE", NULL};

// if true, use CPU based timing for everything
static bool bDontUseGPUTiming;

int *pArgc = NULL;
char **pArgv = NULL;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(const int argc, const char **argv);
void testBandwidth(unsigned int start, unsigned int end, unsigned int increment,
                   testMode mode, memcpyKind kind, printMode printmode,
                   memoryMode memMode, int startDevice, int endDevice, bool wc);
void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode,
                        memoryMode memMode, int startDevice, int endDevice,
                        bool wc);
void testBandwidthRange(unsigned int start, unsigned int end,
                        unsigned int increment, memcpyKind kind,
                        printMode printmode, memoryMode memMode,
                        int startDevice, int endDevice, bool wc);
void testBandwidthShmoo(memcpyKind kind, printMode printmode,
                        memoryMode memMode, int startDevice, int endDevice,
                        bool wc);
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
                               bool wc);
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode,
                               bool wc);
float testDeviceToDeviceTransfer(unsigned int memSize);
void printResultsReadable(unsigned int *memSizes, double *bandwidths,
                          unsigned int count, memcpyKind kind,
                          memoryMode memMode, int iNumDevs, bool wc);
void printResultsCSV(unsigned int *memSizes, double *bandwidths,
                     unsigned int count, memcpyKind kind, memoryMode memMode,
                     int iNumDevs, bool wc);
void printHelp(void);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  flush_buf = (char *)malloc(FLUSH_SIZE);

  // set logfile name and start logs
  printf("[%s] - Starting...\n", sSDKsample);

  int iRetVal = runTest(argc, (const char **)argv);

  if (iRetVal < 0) {
    /*
    DPCT1093:17: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(0)));
  }

  // finish
  printf("%s\n", (iRetVal == 0) ? "Result = PASS" : "Result = FAIL");

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n");

  free(flush_buf);

  exit((iRetVal == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////
// Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int runTest(const int argc, const char **argv) try {
  int start = DEFAULT_SIZE;
  int end = DEFAULT_SIZE;
  int startDevice = 0;
  int endDevice = 0;
  int increment = DEFAULT_INCREMENT;
  testMode mode = QUICK_MODE;
  bool htod = false;
  bool dtoh = false;
  bool dtod = false;
  bool wc = false;
  char *modeStr;
  char *device = NULL;
  printMode printmode = USER_READABLE;
  char *memModeStr = NULL;
  memoryMode memMode = PINNED;

  // process command line args
  if (checkCmdLineFlag(argc, argv, "help")) {
    printHelp();
    return 0;
  }

  if (checkCmdLineFlag(argc, argv, "csv")) {
    printmode = CSV;
  }

  if (getCmdLineArgumentString(argc, argv, "memory", &memModeStr)) {
    if (strcmp(memModeStr, "pageable") == 0) {
      memMode = PAGEABLE;
    } else if (strcmp(memModeStr, "pinned") == 0) {
      memMode = PINNED;
    } else {
      printf("Invalid memory mode - valid modes are pageable or pinned\n");
      printf("See --help for more information\n");
      return -1000;
    }
  } else {
    // default - pinned memory
    memMode = PINNED;
  }

  if (getCmdLineArgumentString(argc, argv, "device", &device)) {
    int deviceCount;
    dpct::err0 error_id = DPCT_CHECK_ERROR(
        deviceCount = dpct::dev_mgr::instance().device_count());

    if (deviceCount == 0) {
      printf("!!!!!No devices found!!!!!\n");
      return -2000;
    }

    if (strcmp(device, "all") == 0) {
      printf(
          "\n!!!!!Cumulative Bandwidth to be computed from all the devices "
          "!!!!!!\n\n");
      startDevice = 0;
      endDevice = deviceCount - 1;
    } else {
      startDevice = endDevice = atoi(device);

      if (startDevice >= deviceCount || startDevice < 0) {
        printf(
            "\n!!!!!Invalid GPU number %d given hence default gpu %d will be "
            "used !!!!!\n",
            startDevice, 0);
        startDevice = endDevice = 0;
      }
    }
  }

  printf("Running on...\n\n");

  for (int currentDevice = startDevice; currentDevice <= endDevice;
       currentDevice++) {
    dpct::device_info deviceProp;
    dpct::err0 error_id = DPCT_CHECK_ERROR(dpct::get_device_info(
        deviceProp, dpct::dev_mgr::instance().get_device(currentDevice)));

    if (error_id == 0) {
      printf(" Device %d: %s\n", currentDevice, deviceProp.get_name());

      /*
      DPCT1035:19: All SYCL devices can be used by the host to submit tasks. You
      may need to adjust this code.
      */
      if (false) {
        fprintf(stderr,
                "Error: device is running in <Compute Mode Prohibited>, no "
                "threads can use ::cudaSetDevice().\n");
        /*
        DPCT1093:20: The "currentDevice" device may be not the one intended for
        use. Adjust the selected device if needed.
        */
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(currentDevice)));

        exit(EXIT_FAILURE);
      }
    } else {
      printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error_id,
             /*
             DPCT1009:21: SYCL uses exceptions to report errors and does not use
             the error codes. The call was replaced by a placeholder string. You
             need to rewrite this code.
             */
             "<Placeholder string>");
      /*
      DPCT1093:22: The "currentDevice" device may be not the one intended for
      use. Adjust the selected device if needed.
      */
      checkCudaErrors(DPCT_CHECK_ERROR(dpct::select_device(currentDevice)));

      exit(EXIT_FAILURE);
    }
  }

  if (getCmdLineArgumentString(argc, argv, "mode", &modeStr)) {
    // figure out the mode
    if (strcmp(modeStr, "quick") == 0) {
      printf(" Quick Mode\n\n");
      mode = QUICK_MODE;
    } else if (strcmp(modeStr, "shmoo") == 0) {
      printf(" Shmoo Mode\n\n");
      mode = SHMOO_MODE;
    } else if (strcmp(modeStr, "range") == 0) {
      printf(" Range Mode\n\n");
      mode = RANGE_MODE;
    } else {
      printf("Invalid mode - valid modes are quick, range, or shmoo\n");
      printf("See --help for more information\n");
      return -3000;
    }
  } else {
    // default mode - quick
    printf(" Quick Mode\n\n");
    mode = QUICK_MODE;
  }

  if (checkCmdLineFlag(argc, argv, "htod")) {
    htod = true;
  }

  if (checkCmdLineFlag(argc, argv, "dtoh")) {
    dtoh = true;
  }

  if (checkCmdLineFlag(argc, argv, "dtod")) {
    dtod = true;
  }

#if DPCT_COMPAT_RT_VERSION >= 2020

  if (checkCmdLineFlag(argc, argv, "wc")) {
    wc = true;
  }

#endif

  if (checkCmdLineFlag(argc, argv, "cputiming")) {
    bDontUseGPUTiming = true;
  }

  if (!htod && !dtoh && !dtod) {
    // default:  All
    htod = true;
    dtoh = true;
    dtod = true;
  }

  if (RANGE_MODE == mode) {
    if (checkCmdLineFlag(argc, (const char **)argv, "start")) {
      start = getCmdLineArgumentInt(argc, argv, "start");

      if (start <= 0) {
        printf("Illegal argument - start must be greater than zero\n");
        return -4000;
      }
    } else {
      printf("Must specify a starting size in range mode\n");
      printf("See --help for more information\n");
      return -5000;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "end")) {
      end = getCmdLineArgumentInt(argc, argv, "end");

      if (end <= 0) {
        printf("Illegal argument - end must be greater than zero\n");
        return -6000;
      }

      if (start > end) {
        printf("Illegal argument - start is greater than end\n");
        return -7000;
      }
    } else {
      printf("Must specify an end size in range mode.\n");
      printf("See --help for more information\n");
      return -8000;
    }

    if (checkCmdLineFlag(argc, argv, "increment")) {
      increment = getCmdLineArgumentInt(argc, argv, "increment");

      if (increment <= 0) {
        printf("Illegal argument - increment must be greater than zero\n");
        return -9000;
      }
    } else {
      printf("Must specify an increment in user mode\n");
      printf("See --help for more information\n");
      return -10000;
    }
  }

  if (htod) {
    testBandwidth((unsigned int)start, (unsigned int)end,
                  (unsigned int)increment, mode, HOST_TO_DEVICE, printmode,
                  memMode, startDevice, endDevice, wc);
  }

  if (dtoh) {
    testBandwidth((unsigned int)start, (unsigned int)end,
                  (unsigned int)increment, mode, DEVICE_TO_HOST, printmode,
                  memMode, startDevice, endDevice, wc);
  }

  if (dtod) {
    testBandwidth((unsigned int)start, (unsigned int)end,
                  (unsigned int)increment, mode, DEVICE_TO_DEVICE, printmode,
                  memMode, startDevice, endDevice, wc);
  }

  // Ensure that we reset all CUDA Devices in question
  for (int nDevice = startDevice; nDevice <= endDevice; nDevice++) {
    /*
    DPCT1093:23: The "nDevice" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(nDevice);
  }

  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

///////////////////////////////////////////////////////////////////////////////
//  Run a bandwidth test
///////////////////////////////////////////////////////////////////////////////
void testBandwidth(unsigned int start, unsigned int end, unsigned int increment,
                   testMode mode, memcpyKind kind, printMode printmode,
                   memoryMode memMode, int startDevice, int endDevice,
                   bool wc) {
  switch (mode) {
    case QUICK_MODE:
      testBandwidthQuick(DEFAULT_SIZE, kind, printmode, memMode, startDevice,
                         endDevice, wc);
      break;

    case RANGE_MODE:
      testBandwidthRange(start, end, increment, kind, printmode, memMode,
                         startDevice, endDevice, wc);
      break;

    case SHMOO_MODE:
      testBandwidthShmoo(kind, printmode, memMode, startDevice, endDevice, wc);
      break;

    default:
      break;
  }
}

//////////////////////////////////////////////////////////////////////
//  Run a quick mode bandwidth test
//////////////////////////////////////////////////////////////////////
void testBandwidthQuick(unsigned int size, memcpyKind kind, printMode printmode,
                        memoryMode memMode, int startDevice, int endDevice,
                        bool wc) {
  testBandwidthRange(size, size, DEFAULT_INCREMENT, kind, printmode, memMode,
                     startDevice, endDevice, wc);
}

///////////////////////////////////////////////////////////////////////
//  Run a range mode bandwidth test
//////////////////////////////////////////////////////////////////////
void testBandwidthRange(unsigned int start, unsigned int end,
                        unsigned int increment, memcpyKind kind,
                        printMode printmode, memoryMode memMode,
                        int startDevice, int endDevice, bool wc) {
  // count the number of copies we're going to run
  unsigned int count = 1 + ((end - start) / increment);

  unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
  double *bandwidths = (double *)malloc(count * sizeof(double));

  // Before calculating the cumulative bandwidth, initialize bandwidths array to
  // NULL
  for (unsigned int i = 0; i < count; i++) {
    bandwidths[i] = 0.0;
  }

  // Use the device asked by the user
  for (int currentDevice = startDevice; currentDevice <= endDevice;
       currentDevice++) {
    /*
    DPCT1093:24: The "currentDevice" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(currentDevice);

    // run each of the copies
    for (unsigned int i = 0; i < count; i++) {
      memSizes[i] = start + i * increment;

      switch (kind) {
        case DEVICE_TO_HOST:
          bandwidths[i] += testDeviceToHostTransfer(memSizes[i], memMode, wc);
          break;

        case HOST_TO_DEVICE:
          bandwidths[i] += testHostToDeviceTransfer(memSizes[i], memMode, wc);
          break;

        case DEVICE_TO_DEVICE:
          bandwidths[i] += testDeviceToDeviceTransfer(memSizes[i]);
          break;
      }
    }
  }  // Complete the bandwidth computation on all the devices

  // print results
  if (printmode == CSV) {
    printResultsCSV(memSizes, bandwidths, count, kind, memMode,
                    (1 + endDevice - startDevice), wc);
  } else {
    printResultsReadable(memSizes, bandwidths, count, kind, memMode,
                         (1 + endDevice - startDevice), wc);
  }

  // clean up
  free(memSizes);
  free(bandwidths);
}

//////////////////////////////////////////////////////////////////////////////
// Intense shmoo mode - covers a large range of values with varying increments
//////////////////////////////////////////////////////////////////////////////
void testBandwidthShmoo(memcpyKind kind, printMode printmode,
                        memoryMode memMode, int startDevice, int endDevice,
                        bool wc) {
  // count the number of copies to make
  unsigned int count =
      1 + (SHMOO_LIMIT_20KB / SHMOO_INCREMENT_1KB) +
      ((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB) +
      ((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB) +
      ((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB) +
      ((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB) +
      ((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB) +
      ((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

  unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
  double *bandwidths = (double *)malloc(count * sizeof(double));

  // Before calculating the cumulative bandwidth, initialize bandwidths array to
  // NULL
  for (unsigned int i = 0; i < count; i++) {
    bandwidths[i] = 0.0;
  }

  // Use the device asked by the user
  for (int currentDevice = startDevice; currentDevice <= endDevice;
       currentDevice++) {
    /*
    DPCT1093:25: The "currentDevice" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(currentDevice);
    // Run the shmoo
    int iteration = 0;
    unsigned int memSize = 0;

    while (memSize <= SHMOO_MEMSIZE_MAX) {
      if (memSize < SHMOO_LIMIT_20KB) {
        memSize += SHMOO_INCREMENT_1KB;
      } else if (memSize < SHMOO_LIMIT_50KB) {
        memSize += SHMOO_INCREMENT_2KB;
      } else if (memSize < SHMOO_LIMIT_100KB) {
        memSize += SHMOO_INCREMENT_10KB;
      } else if (memSize < SHMOO_LIMIT_1MB) {
        memSize += SHMOO_INCREMENT_100KB;
      } else if (memSize < SHMOO_LIMIT_16MB) {
        memSize += SHMOO_INCREMENT_1MB;
      } else if (memSize < SHMOO_LIMIT_32MB) {
        memSize += SHMOO_INCREMENT_2MB;
      } else {
        memSize += SHMOO_INCREMENT_4MB;
      }

      memSizes[iteration] = memSize;

      switch (kind) {
        case DEVICE_TO_HOST:
          bandwidths[iteration] +=
              testDeviceToHostTransfer(memSizes[iteration], memMode, wc);
          break;

        case HOST_TO_DEVICE:
          bandwidths[iteration] +=
              testHostToDeviceTransfer(memSizes[iteration], memMode, wc);
          break;

        case DEVICE_TO_DEVICE:
          bandwidths[iteration] +=
              testDeviceToDeviceTransfer(memSizes[iteration]);
          break;
      }

      iteration++;
      printf(".");
      fflush(0);
    }
  }  // Complete the bandwidth computation on all the devices

  // print results
  printf("\n");

  if (CSV == printmode) {
    printResultsCSV(memSizes, bandwidths, count, kind, memMode,
                    (1 + endDevice - startDevice), wc);
  } else {
    printResultsReadable(memSizes, bandwidths, count, kind, memMode,
                         (1 + endDevice - startDevice), wc);
  }

  // clean up
  free(memSizes);
  free(bandwidths);
}

///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode,
                               bool wc) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  unsigned char *h_idata = NULL;
  unsigned char *h_odata = NULL;
  dpct::event_ptr start, stop;

  sdkCreateTimer(&timer);
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

  // allocate host memory
  if (PINNED == memMode) {
  // pinned memory mode - use special function to get OS-pinned memory
#if DPCT_COMPAT_RT_VERSION >= 2020
    checkCudaErrors(
        DPCT_CHECK_ERROR(h_idata = (unsigned char *)sycl::malloc_host(
                             memSize, dpct::get_in_order_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(h_odata = (unsigned char *)sycl::malloc_host(
                             memSize, dpct::get_in_order_queue())));
#else
    checkCudaErrors(cudaMallocHost((void **)&h_idata, memSize));
    checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
  } else {
    // pageable memory mode - use malloc
    h_idata = (unsigned char *)malloc(memSize);
    h_odata = (unsigned char *)malloc(memSize);

    if (h_idata == 0 || h_odata == 0) {
      fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
      exit(EXIT_FAILURE);
    }
  }

  // initialize the memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_idata[i] = (unsigned char)(i & 0xff);
  }

  // allocate device memory
  unsigned char *d_idata;
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_idata = (unsigned char *)sycl::malloc_device(
                           memSize, dpct::get_in_order_queue())));

  // initialize the device memory
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_idata, h_idata, memSize).wait()));

  // copy data from GPU to Host
  if (PINNED == memMode) {
    if (bDontUseGPUTiming) sdkStartTimer(&timer);
    /*
    DPCT1024:26: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::sync_barrier(start, &dpct::get_in_order_queue())));
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue().memcpy(h_odata, d_idata, memSize)));
    }
    /*
    DPCT1024:27: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::sync_barrier(stop, &dpct::get_in_order_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    checkCudaErrors(DPCT_CHECK_ERROR(
        elapsedTimeInMs = (stop->get_profiling_info<
                               sycl::info::event_profiling::command_end>() -
                           start->get_profiling_info<
                               sycl::info::event_profiling::command_start>()) /
                          1000000.0f));
    if (bDontUseGPUTiming) {
      sdkStopTimer(&timer);
      elapsedTimeInMs = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
    }
  } else {
    elapsedTimeInMs = 0;
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
      sdkStartTimer(&timer);
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue().memcpy(h_odata, d_idata, memSize).wait()));
      sdkStopTimer(&timer);
      elapsedTimeInMs += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      memset(flush_buf, i, FLUSH_SIZE);
    }
  }

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;
  // clean up memory
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
  sdkDeleteTimer(&timer);

  if (PINNED == memMode) {
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(h_idata, dpct::get_in_order_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(h_odata, dpct::get_in_order_queue())));
  } else {
    free(h_idata);
    free(h_odata);
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_idata, dpct::get_in_order_queue())));

  return bandwidthInGBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode,
                               bool wc) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  dpct::event_ptr start, stop;
  sdkCreateTimer(&timer);
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

  // allocate host memory
  unsigned char *h_odata = NULL;

  if (PINNED == memMode) {
#if DPCT_COMPAT_RT_VERSION >= 2020
    // pinned memory mode - use special function to get OS-pinned memory
    checkCudaErrors(
        DPCT_CHECK_ERROR(h_odata = (unsigned char *)sycl::malloc_host(
                             memSize, dpct::get_in_order_queue())));
#else
    // pinned memory mode - use special function to get OS-pinned memory
    checkCudaErrors(cudaMallocHost((void **)&h_odata, memSize));
#endif
  } else {
    // pageable memory mode - use malloc
    h_odata = (unsigned char *)malloc(memSize);

    if (h_odata == 0) {
      fprintf(stderr, "Not enough memory available on host to run test!\n");
      exit(EXIT_FAILURE);
    }
  }

  unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
  unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

  if (h_cacheClear1 == 0 || h_cacheClear2 == 0) {
    fprintf(stderr, "Not enough memory available on host to run test!\n");
    exit(EXIT_FAILURE);
  }

  // initialize the memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_odata[i] = (unsigned char)(i & 0xff);
  }

  for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
    h_cacheClear1[i] = (unsigned char)(i & 0xff);
    h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
  }

  // allocate device memory
  unsigned char *d_idata;
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_idata = (unsigned char *)sycl::malloc_device(
                           memSize, dpct::get_in_order_queue())));

  // copy host memory to device memory
  if (PINNED == memMode) {
    if (bDontUseGPUTiming) sdkStartTimer(&timer);
    /*
    DPCT1024:28: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::sync_barrier(start, &dpct::get_in_order_queue())));
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue().memcpy(d_idata, h_odata, memSize)));
    }
    /*
    DPCT1024:29: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::sync_barrier(stop, &dpct::get_in_order_queue())));
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    checkCudaErrors(DPCT_CHECK_ERROR(
        elapsedTimeInMs = (stop->get_profiling_info<
                               sycl::info::event_profiling::command_end>() -
                           start->get_profiling_info<
                               sycl::info::event_profiling::command_start>()) /
                          1000000.0f));
    if (bDontUseGPUTiming) {
      sdkStopTimer(&timer);
      elapsedTimeInMs = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
    }
  } else {
    elapsedTimeInMs = 0;
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
      sdkStartTimer(&timer);
      checkCudaErrors(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue().memcpy(d_idata, h_odata, memSize).wait()));
      sdkStopTimer(&timer);
      elapsedTimeInMs += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      memset(flush_buf, i, FLUSH_SIZE);
    }
  }

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;
  // clean up memory
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
  sdkDeleteTimer(&timer);

  if (PINNED == memMode) {
    checkCudaErrors(
        DPCT_CHECK_ERROR(sycl::free(h_odata, dpct::get_in_order_queue())));
  } else {
    free(h_odata);
  }

  free(h_cacheClear1);
  free(h_cacheClear2);
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_idata, dpct::get_in_order_queue())));

  return bandwidthInGBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float testDeviceToDeviceTransfer(unsigned int memSize) {
  StopWatchInterface *timer = NULL;
  float elapsedTimeInMs = 0.0f;
  float bandwidthInGBs = 0.0f;
  dpct::event_ptr start, stop;

  sdkCreateTimer(&timer);
  checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
  checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

  // allocate host memory
  unsigned char *h_idata = (unsigned char *)malloc(memSize);

  if (h_idata == 0) {
    fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
    exit(EXIT_FAILURE);
  }

  // initialize the host memory
  for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
    h_idata[i] = (unsigned char)(i & 0xff);
  }

  // allocate device memory
  unsigned char *d_idata;
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_idata = (unsigned char *)sycl::malloc_device(
                           memSize, dpct::get_in_order_queue())));
  unsigned char *d_odata;
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_odata = (unsigned char *)sycl::malloc_device(
                           memSize, dpct::get_in_order_queue())));

  // initialize memory
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_idata, h_idata, memSize).wait()));

  // run the memcopy
  sdkStartTimer(&timer);
  /*
  DPCT1024:30: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::sync_barrier(start, &dpct::get_in_order_queue())));

  for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(d_odata, d_idata, memSize)));
  }

  /*
  DPCT1024:31: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::sync_barrier(stop, &dpct::get_in_order_queue())));

  // Since device to device memory copies are non-blocking,
  // cudaDeviceSynchronize() is required in order to get
  // proper timing.
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

  // get the total elapsed time in ms
  sdkStopTimer(&timer);
  checkCudaErrors(DPCT_CHECK_ERROR(
      elapsedTimeInMs = (stop->get_profiling_info<
                             sycl::info::event_profiling::command_end>() -
                         start->get_profiling_info<
                             sycl::info::event_profiling::command_start>()) /
                        1000000.0f));

  if (bDontUseGPUTiming) {
    elapsedTimeInMs = sdkGetTimerValue(&timer);
  }

  // calculate bandwidth in GB/s
  double time_s = elapsedTimeInMs / 1e3;
  bandwidthInGBs = (2.0f * memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
  bandwidthInGBs = bandwidthInGBs / time_s;

  // clean up memory
  sdkDeleteTimer(&timer);
  free(h_idata);
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(stop)));
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_idata, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_odata, dpct::get_in_order_queue())));

  return bandwidthInGBs;
}

/////////////////////////////////////////////////////////
// print results in an easily read format
////////////////////////////////////////////////////////
void printResultsReadable(unsigned int *memSizes, double *bandwidths,
                          unsigned int count, memcpyKind kind,
                          memoryMode memMode, int iNumDevs, bool wc) {
  printf(" %s Bandwidth, %i Device(s)\n", sMemoryCopyKind[kind], iNumDevs);
  printf(" %s Memory Transfers\n", sMemoryMode[memMode]);

  if (wc) {
    printf(" Write-Combined Memory Writes are Enabled");
  }

  printf("   Transfer Size (Bytes)\tBandwidth(GB/s)\n");
  unsigned int i;

  for (i = 0; i < (count - 1); i++) {
    printf("   %u\t\t\t%s%.1f\n", memSizes[i],
           (memSizes[i] < 10000) ? "\t" : "", bandwidths[i]);
  }

  printf("   %u\t\t\t%s%.1f\n\n", memSizes[i],
         (memSizes[i] < 10000) ? "\t" : "", bandwidths[i]);
}

///////////////////////////////////////////////////////////////////////////
// print results in a database format
///////////////////////////////////////////////////////////////////////////
void printResultsCSV(unsigned int *memSizes, double *bandwidths,
                     unsigned int count, memcpyKind kind, memoryMode memMode,
                     int iNumDevs, bool wc) {
  std::string sConfig;

  // log config information
  if (kind == DEVICE_TO_DEVICE) {
    sConfig += "D2D";
  } else {
    if (kind == DEVICE_TO_HOST) {
      sConfig += "D2H";
    } else if (kind == HOST_TO_DEVICE) {
      sConfig += "H2D";
    }

    if (memMode == PAGEABLE) {
      sConfig += "-Paged";
    } else if (memMode == PINNED) {
      sConfig += "-Pinned";

      if (wc) {
        sConfig += "-WriteCombined";
      }
    }
  }

  unsigned int i;
  double dSeconds = 0.0;

  for (i = 0; i < count; i++) {
    dSeconds = (double)memSizes[i] / (bandwidths[i] * (double)(1e9));
    printf(
        "bandwidthTest-%s, Bandwidth = %.1f GB/s, Time = %.5f s, Size = %u "
        "bytes, NumDevsUsed = %d\n",
        sConfig.c_str(), bandwidths[i], dSeconds, memSizes[i], iNumDevs);
  }
}

///////////////////////////////////////////////////////////////////////////
// Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void) {
  printf("Usage:  bandwidthTest [OPTION]...\n");
  printf(
      "Test the bandwidth for device to host, host to device, and device to "
      "device transfers\n");
  printf("\n");
  printf(
      "Example:  measure the bandwidth of device to host pinned memory copies "
      "in the range 1024 Bytes to 102400 Bytes in 1024 Byte increments\n");
  printf(
      "./bandwidthTest --memory=pinned --mode=range --start=1024 --end=102400 "
      "--increment=1024 --dtoh\n");

  printf("\n");
  printf("Options:\n");
  printf("--help\tDisplay this help menu\n");
  printf("--csv\tPrint results as a CSV\n");
  printf("--device=[deviceno]\tSpecify the device device to be used\n");
  printf("  all - compute cumulative bandwidth on all the devices\n");
  printf("  0,1,2,...,n - Specify any particular device to be used\n");
  printf("--memory=[MEMMODE]\tSpecify which memory mode to use\n");
  printf("  pageable - pageable memory\n");
  printf("  pinned   - non-pageable system memory\n");
  printf("--mode=[MODE]\tSpecify the mode to use\n");
  printf("  quick - performs a quick measurement\n");
  printf("  range - measures a user-specified range of values\n");
  printf("  shmoo - performs an intense shmoo of a large range of values\n");

  printf("--htod\tMeasure host to device transfers\n");
  printf("--dtoh\tMeasure device to host transfers\n");
  printf("--dtod\tMeasure device to device transfers\n");
#if DPCT_COMPAT_RT_VERSION >= 2020
  printf("--wc\tAllocate pinned memory as write-combined\n");
#endif
  printf("--cputiming\tForce CPU-based timing always\n");

  printf("Range mode options\n");
  printf("--start=[SIZE]\tStarting transfer size in bytes\n");
  printf("--end=[SIZE]\tEnding transfer size in bytes\n");
  printf("--increment=[SIZE]\tIncrement size in bytes\n");
}
