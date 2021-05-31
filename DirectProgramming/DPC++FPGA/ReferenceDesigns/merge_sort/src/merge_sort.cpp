#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "MergeSort.hpp"

using namespace sycl;
using namespace std::chrono;

// determines whether we will use USM host or device allocations to move data
// between host and the device.
#if defined(USM_HOST_ALLOCATIONS)
constexpr bool kUseUSMHostAllocation = true;
#else
constexpr bool kUseUSMHostAllocation = false;
#endif

// the number of merge units, must be a power of 2
#ifndef MERGE_UNITS
#define MERGE_UNITS 8
#endif
constexpr size_t kMergeUnits = MERGE_UNITS;
static_assert(kMergeUnits > 0);
static_assert(IsPow2(kMergeUnits));

// the width of the sort, must be a power of 2
#ifndef SORT_WIDTH
#define SORT_WIDTH 4
#endif
constexpr size_t kSortWidth = SORT_WIDTH;
static_assert(kSortWidth > 1);
static_assert(IsPow2(kSortWidth));

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in main
template <typename ValueT, typename IndexT, typename KernelPtrType>
double fpga_sort(queue &q, ValueT *in_vec, ValueT *out_vec, IndexT count);

template <typename T>
bool validate(T *val, T *ref, unsigned int count);
////////////////////////////////////////////////////////////////////////////////

// main
int main(int argc, char *argv[]) {
  // the type to sort, needs a compare function!
  using ValueT = int;

  // the type used to index in the sorter
  using IndexT = unsigned int;

  bool passed = true;
#ifdef FPGA_EMULATOR
  IndexT count = 64;
  int runs = 1;
#else
  IndexT count = 1 << 24;
  int runs = 17;
#endif
  int seed = 777;

  // get the size of the input as the first command line argument
  if (argc > 1) {
    count = atoi(argv[1]);
  }

  // get the number of runs as the second command line argument
  if (argc > 2) {
    runs = atoi(argv[2]);
  }

  // get the random number generator seed as the third command line argument
  if (argc > 3) {
    seed = atoi(argv[3]);
  }

  // enforce at least two runs
  runs = std::max((int)2, runs);

  // check args
  if (count <= kMergeUnits) {
    std::cerr << "ERROR: count must be greater than number of merge units\n";
    std::terminate();
  } else if (count > std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the index type (IndexT) does not have enough bits to "
              << "count to 'count'\n";
    std::terminate();
  } else if ((count % kSortWidth) != 0) {
    std::cerr << "ERROR: count must be a multiple of the sorter width\n";
    std::terminate();
  }

  // the device selector
#ifdef FPGA_EMULATOR
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // create the device queue
  queue q(selector, dpc_common::exception_handler);

  // make sure the device supports USM device allocations
  device d = q.get_device();
  if (!d.get_info<info::device::usm_device_allocations>()) {
    std::cerr << "ERROR: The selected device does not support USM device"
              << " allocations\n";
    std::terminate();
  }

  // make sure the device support USM host allocations if we chose to use them
  if (!d.get_info<info::device::usm_host_allocations>() &&
      kUseUSMHostAllocation) {
    std::cerr << "ERROR: The selected device does not support USM host"
              << " allocations\n";
    std::terminate();
  }

  // the input, output, and output reference data
  std::vector<ValueT> in_vec(count), out_vec(count), ref(count);

  // generate some random input data
  srand(seed);
  std::generate(in_vec.begin(), in_vec.end(), [] { return rand() % 100; });

  // copy the input to the output reference and compute the expected result
  std::copy(in_vec.begin(), in_vec.end(), ref.begin());
  std::sort(ref.begin(), ref.end());

  // allocate the input and output data either in USM host or device allocations
  ValueT *in, *out;
  if constexpr (kUseUSMHostAllocation) {
    // streaming data using USM host allocations
    if ((in = malloc_host<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in' using "
                << "malloc_host\n";
      std::terminate();
    }
    if ((out = malloc_host<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out' using "
                << "malloc_host\n";
      std::terminate();
    }

    // Copy the input to USM memory and reset the output.
    // This is NOT efficient since, in the case of USM host allocations,
    // we could have simply generated the input data into the host allocation
    // and avoided this copy. However, it makes the code cleaner to assume the
    // input is always in 'in_vec' and this portion of the code is not part of
    // the performance timing anyways.
    std::copy(in_vec.begin(), in_vec.end(), in);
    std::fill(out, out + count, ValueT(0));
  } else {
    // streaming data using device allocations
    if ((in = malloc_device<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in' using "
                << "malloc_device\n";
      std::terminate();
    }
    if ((out = malloc_device<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out' using "
                << "malloc_device\n";
      std::terminate();
    }

    // copy the input to the device memory and wait for the copy to finish
    q.memcpy(in, in_vec.data(), count * sizeof(ValueT)).wait();
  }

  // track timing information, in ms
  std::vector<double> time(runs);

  try {
    std::cout << "Running sort " << runs << " times for an "
              << "input size of " << count << " using " << kMergeUnits
              << " " << kSortWidth << "-way merge units\n";
    std::cout << "Streaming data from "
              << (kUseUSMHostAllocation ? "host" : "device") << " memory\n";

    // the pointer type for the kernel depends on whether we are streaming
    // data via USM host allocations or device memory
    using KernelPtrType =
        typename std::conditional_t<kUseUSMHostAllocation, host_ptr<ValueT>,
                                    device_ptr<ValueT>>;

    // run some sort iterations
    for (int i = 0; i < runs; i++) {
      // run the sort
      time[i] = fpga_sort<ValueT, IndexT, KernelPtrType>(q, in, out, count);

      // Copy output to out_vec. In the case where we are using USM host
      // allocations this is unnecessary since we could simply deference
      // 'out'. However, it makes the following code cleaner since the output
      // is always in 'out_vec' and this copy is not part of the performance
      // timing anyway.
      q.memcpy(out_vec.data(), out, count * sizeof(ValueT)).wait();

      // validate the output
      passed &= validate(out_vec.data(), ref.data(), count);
    }
  } catch (exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  // free the memory allocated with malloc_host or malloc_device
  sycl::free(in, q);
  sycl::free(out, q);

  // print the performance results
  if (passed) {
    // don't print performance when running in the emulator
#ifndef FPGA_EMULATOR
    double avg_time = std::accumulate(time.begin() + 1, time.end(), 0.0);
    avg_time /= (double)(runs - 1);

    IndexT input_count_m = count * 1e-6;

    std::cout << "Execution time: " << avg_time << " ms\n";
    std::cout << "Throughput: " << (input_count_m / (avg_time * 1e-3))
              << " Melements/s\n";
#endif

    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 0;
  }
}

// forward declare the kernel and pipe IDs to reduce name mangling
class InputKernelID;
class OuputKernelID;
class SortInPipeID;
class SortOutPipeID;

//
// perform the actual sort on the FPGA.
//
template <typename ValueT, typename IndexT, typename KernelPtrType>
double fpga_sort(queue &q, ValueT *in_ptr, ValueT *out_ptr, IndexT count) {
  // the input and output pipe for the sorter
  using SortInPipe =
      sycl::INTEL::pipe<SortInPipeID, sycl::vec<ValueT, kSortWidth>>;
  using SortOutPipe =
      sycl::INTEL::pipe<SortOutPipeID, sycl::vec<ValueT, kSortWidth>>;

  // the sorter must sort a power of 2, so round up the requested count
  // to the nearest power of 2; we will pad the input to make sure the
  // output is still correct
  IndexT sorter_count = RoundUpPow2(count);

  // This is the element we will pad the input with. In the case of this design,
  // we are sorting from smallest to largest and we want the last elements out
  // to be this element, so pad with MAX. If you are sorting from largest to
  // smallest, make this the minimum element. If you are sorting custom types
  // which are not supported by std::numeric_limits, then you will have to set
  // this padding element differently.
  const auto padding_element = std::numeric_limits<ValueT>::max();

  // We are sorting kSortWidth elements per cycle, so we will have 
  // sorter_count/kSortWidth pipe reads/writes from/to the sorter
  const IndexT total_pipe_accesses = sorter_count / kSortWidth;

  // launch the kernel that feeds data into the sorter
  auto producer_event = q.submit([&](handler &h) {
    h.single_task<InputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the input pointer and write it to the sorter's input pipe
      KernelPtrType in(in_ptr);
      for (IndexT i = 0; i < total_pipe_accesses; i++) {
        // read data from device memory
        bool in_range = i < sorter_count;

        // build the input pipe value
        sycl::vec<ValueT, kSortWidth> data;
        #pragma unroll
        for (unsigned char j = 0; j < kSortWidth; j++) {
          data[j] = in_range ? in[i * kSortWidth + j] : padding_element;
        }

        // write it into the sorter
        SortInPipe::write(data);
      }
    });
  });

  // launch the kernel that pulls data out of the sorter
  auto consumer_event = q.submit([&](handler &h) {
    h.single_task<OuputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the sorters output pipe and write to the output pointer
      KernelPtrType out(out_ptr);
      for (IndexT i = 0; i < total_pipe_accesses; i++) {
        // read data from the sorter
        auto data = SortOutPipe::read();

        // sorter_count is a multiple of kSortWidth
        bool in_range = i < sorter_count;

        // only write out to device memory if the index is in range
        if (in_range) {
          // write output to device memory
          #pragma unroll
          for (unsigned char j = 0; j < kSortWidth; j++) {
            out[i * kSortWidth + j] = data[j];
          }
        }
      }
    });
  });

  // allocate some memory for the merge sort to use as temporary storage
  ValueT *buf_0, *buf_1;
  if ((buf_0 = malloc_device<ValueT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_0'\n";
    std::terminate();
  }
  if ((buf_1 = malloc_device<ValueT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_1'\n";
    std::terminate();
  }

  // launch the merge sort kernels
  auto merge_sort_events =
      SubmitMergeSort<ValueT, IndexT, SortInPipe, SortOutPipe, kSortWidth,
                      kMergeUnits>(q, sorter_count, buf_0, buf_1);

  // wait for the producer and consumer to finish
  auto start = high_resolution_clock::now();
  producer_event.wait();
  consumer_event.wait();
  auto end = high_resolution_clock::now();

  // wait for the merge sort events to all finish
  for (auto &e : merge_sort_events) {
    e.wait();
  }

  // free the memory allocated for the merge sort temporary buffers
  sycl::free(buf_0, q);
  sycl::free(buf_1, q);

  // return the duration of the sort in milliseconds, excluding memory transfers
  duration<double, std::milli> diff = end - start;
  return diff.count();
}

//
// simple function to check if two regions of memory contain the same values
//
template <typename T>
bool validate(T *val, T *ref, unsigned int count) {
  for (unsigned int i = 0; i < count; i++) {
    if (val[i] != ref[i]) {
      std::cout << "ERROR: mismatch at entry " << i << "\n";
      std::cout << "\t" << val[i] << " != " << ref[i]
                << " (val[i] != ref[i])\n";
      return false;
    }
  }

  return true;
}
