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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/random>

#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>
#include <chrono>

template <typename T, bool floatKeys> bool testSort(int argc, char **argv) try {
  int cmdVal;
  int keybits = 32;

  unsigned int numElements = 1048576;
  bool keysOnly = checkCmdLineFlag(argc, (const char **)argv, "keysonly");
  bool quiet = checkCmdLineFlag(argc, (const char **)argv, "quiet");

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    numElements = cmdVal;

    if (cmdVal < 0) {
      printf("Error: elements must be > 0, elements=%d is invalid\n", cmdVal);
      exit(EXIT_SUCCESS);
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "keybits")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "keybits");
    keybits = cmdVal;

    if (keybits <= 0) {
      printf("Error: keybits must be > 0, keybits=%d is invalid\n", keybits);
      exit(EXIT_SUCCESS);
    }
  }

  unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;

  if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    numIterations = cmdVal;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Command line:\nradixSortThrust [-option]\n");
    printf("Valid options:\n");
    printf("-n=<N>        : number of elements to sort\n");
    printf("-keybits=bits : keybits must be > 0\n");
    printf(
        "-keysonly     : only sort an array of keys (default sorts key-value "
        "pairs)\n");
    printf(
        "-float        : use 32-bit float keys (default is 32-bit unsigned "
        "int)\n");
    printf(
        "-quiet        : Output only the number of elements and the time to "
        "sort\n");
    printf("-help         : Output a help message\n");
    exit(EXIT_SUCCESS);
  }

  if (!quiet)
    printf("\nSorting %d %d-bit %s keys %s\n\n", numElements, keybits,
           floatKeys ? "float" : "unsigned int",
           keysOnly ? "(only)" : "and values");

  int deviceID = -1;
  deviceID = dpct::dev_mgr::instance().current_device_id();

  if (0 == deviceID) {
    dpct::device_info devprop;
    dpct::dev_mgr::instance().get_device(deviceID).get_device_info(devprop);
    unsigned int totalMem = (keysOnly ? 2 : 4) * numElements * sizeof(T);

    if (devprop.get_global_mem_size() < totalMem) {
      printf("Error: insufficient amount of memory to sort %d elements.\n",
             numElements);
      printf("%d bytes needed, %d bytes available\n", (int)totalMem,
             (int)devprop.get_global_mem_size());
      exit(EXIT_SUCCESS);
    }
  }

  std::vector<T> h_keys(numElements);
  std::vector<T> h_keysSorted(numElements);
  std::vector<unsigned int> h_values;

  if (!keysOnly) h_values = std::vector<unsigned int>(numElements);

  // Fill up with some random data
  oneapi::dpl::minstd_rand rng(clock());

  if (floatKeys) {
    oneapi::dpl::uniform_real_distribution<float> u01(0, 1);

    for (int i = 0; i < (int)numElements; i++) h_keys[i] = u01(rng);
  } else {
    oneapi::dpl::uniform_int_distribution<unsigned int> u(0, UINT_MAX);

    for (int i = 0; i < (int)numElements; i++) h_keys[i] = u(rng);
  }

  if (!keysOnly)
      dpct::iota(oneapi::dpl::execution::seq, h_values.begin(), h_values.end());

  // Copy data onto the GPU
  dpct::device_vector<T> d_keys;
  dpct::device_vector<unsigned int> d_values;

  // run multiple iterations to compute an average sort time
  dpct::event_ptr start_event, stop_event;
  std::chrono::time_point<std::chrono::steady_clock> start_event_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_event_ct1;
  start_event = new sycl::event();
  stop_event = new sycl::event();

  float totalTime = 0;

  for (unsigned int i = 0; i < numIterations; i++) {
    // reset data before sort
    d_keys = h_keys;

    if (!keysOnly) d_values = h_values;

    start_event_ct1 = std::chrono::steady_clock::now();

    if (keysOnly)
      oneapi::dpl::sort(
          oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
          d_keys.begin(), d_keys.end());
    else
      dpct::sort(
          oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
          d_keys.begin(), d_keys.end(), d_values.begin());

    stop_event_ct1 = std::chrono::steady_clock::now();

    float time = 0;
    DPCT_CHECK_ERROR((time = std::chrono::duration<float, std::milli>(
                                     stop_event_ct1 - start_event_ct1)
                                     .count()));
    totalTime += time;
  }

  totalTime /= (1.0e3f * numIterations);
  printf(
      "radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = "
      "%u elements\n",
      1.0e-6f * numElements / totalTime, totalTime, numElements);

  // Get results back to host for correctness checking
  std::copy(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      d_keys.begin(), d_keys.end(), h_keysSorted.begin());

  if (!keysOnly)
    std::copy(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        d_values.begin(), d_values.end(), h_values.begin());

  // Check results
  bool bTestResult = oneapi::dpl::is_sorted(
      oneapi::dpl::execution::seq, h_keysSorted.begin(), h_keysSorted.end());

  dpct::destroy_event(start_event);
  dpct::destroy_event(stop_event);

  if (!bTestResult && !quiet) {
    return false;
  }

  return bTestResult;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char **argv) {
  time_t start, end;
  time(&start);
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  bool bTestResult = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "float"))
    bTestResult = testSort<float, true>(argc, argv);
  else
    bTestResult = testSort<unsigned int, false>(argc, argv);

  printf(bTestResult ? "Test passed\n" : "Test failed!\n");
  time(&end);
  double time_taken = double(end - start);
    std::cout << "Time taken by program is : " << std::fixed
        << time_taken << std::setprecision(5);
    std::cout << " sec " << std::endl;
}