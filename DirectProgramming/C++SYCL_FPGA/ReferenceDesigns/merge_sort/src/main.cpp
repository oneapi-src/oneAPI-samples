#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

#include "merge_sort.hpp"

// Included from DirectProgramming/C++SYCL_FPGA/include/
#include "constexpr_math.hpp"

using namespace sycl;
using namespace std::chrono;

// Determines whether we will use USM host or device allocations to move data
// between host and the device.
// This can be set on the command line by defining the preprocessor macro
// 'USM_HOST_ALLOCATIONS' using the flag: '-DUSM_HOST_ALLOCATIONS'
#if defined(USM_HOST_ALLOCATIONS)
constexpr bool kUseUSMHostAllocation = true;
#else
constexpr bool kUseUSMHostAllocation = false;
#endif

// The number of merge units, which must be a power of 2.
// This can be set by defining the preprocessor macro 'MERGE_UNITS'
// otherwise the default value below is used.
#ifndef MERGE_UNITS
#if defined(FPGA_SIMULATOR)
#define MERGE_UNITS 2
#else
#define MERGE_UNITS 8
#endif
#endif
constexpr size_t kMergeUnits = MERGE_UNITS;
static_assert(kMergeUnits > 0);
static_assert(fpga_tools::IsPow2(kMergeUnits));

// The width of the sort, which must be a power of 2
// This can be set by defining the preprocessor macro 'SORT_WIDTH'
// otherwise the default value below is used.
#ifndef SORT_WIDTH
#define SORT_WIDTH 4
#endif
constexpr size_t kSortWidth = SORT_WIDTH;
static_assert(kSortWidth >= 1);
static_assert(fpga_tools::IsPow2(kSortWidth));

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in this file by main()
template <typename ValueT, typename IndexT, typename KernelPtrType>
double FPGASort(queue &q, ValueT *in_vec, ValueT *out_vec, IndexT count);

template <typename T>
bool Validate(T *val, T *ref, unsigned int count);
////////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]) {
  // the type to sort, needs a compare function!
  using ValueT = int;

  // the type used to index in the sorter
  // below we do a runtime check to make sure this type has enough bits to
  // count all the elements to be sorted.
  using IndexT = unsigned int;

  /////////////////////////////////////////////////////////////
  // reading and validating the command line arguments
  // defaults
  bool passed = true;
#if defined(FPGA_EMULATOR)
  IndexT count = 128;
  int runs = 2;
#elif defined(FPGA_SIMULATOR)
  IndexT count = 16;
  int runs = 2;
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
  if (runs < 2) {
    std::cerr << "ERROR: 'runs' must be 2 or more\n";
    std::terminate();
  }

  // check args
  if (count <= kMergeUnits) {
    std::cerr << "ERROR: 'count' must be greater than number of merge units\n";
    std::terminate();
  } else if (count > std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the index type (IndexT) does not have enough bits to "
              << "count to 'count'\n";
    std::terminate();
  } else if ((count % kSortWidth) != 0) {
    std::cerr << "ERROR: 'count' must be a multiple of the sorter width\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////

  // the device selector
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  // create the device queue
  queue q(selector, fpga_tools::exception_handler);

  auto device = q.get_device();

  // make sure the device supports USM device allocations in BSP mode
#if defined(IS_BSP)
  if (!device.has(aspect::usm_device_allocations)) {
    std::cerr << "ERROR: The selected device does not support USM device"
              << " allocations\n";
    std::terminate();
  }
#endif

  // make sure the device support USM host allocations if we chose to use them
  if (!device.has(aspect::usm_host_allocations) &&
      kUseUSMHostAllocation) {
    std::cerr << "ERROR: The selected device does not support USM host"
              << " allocations\n";
    std::terminate();
  }

  std::cout << "Running on device: "
            << device.get_info<info::device::name>().c_str() 
            << std::endl;

  // the input, output, and reference data
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
    // using USM host allocations
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
    // the performance timing.
    std::copy(in_vec.begin(), in_vec.end(), in);
    std::fill(out, out + count, ValueT(0));
  } else {
    // using device allocations
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

    // the pointer type for the kernel depends on whether data is coming from
    // USM host or device allocations
    using KernelPtrType =
        typename std::conditional_t<kUseUSMHostAllocation, host_ptr<ValueT>,
                                    device_ptr<ValueT>>;

    // run the sort multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      // run the sort
      time[i] = FPGASort<ValueT, IndexT, KernelPtrType>(q, in, out, count);

      // Copy the output to 'out_vec'. In the case where we are using USM host
      // allocations this is unnecessary since we could simply deference
      // 'out'. However, it makes the following code cleaner since the output
      // is always in 'out_vec' and this copy is not part of the performance
      // timing.
      q.memcpy(out_vec.data(), out, count * sizeof(ValueT)).wait();

      // validate the output
      passed &= Validate(out_vec.data(), ref.data(), count);
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
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels in actual FPGA hardware
    double avg_time_ms =
      std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);

    IndexT input_count_mega = count * 1e-6;

    std::cout << "Execution time: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << (input_count_mega / (avg_time_ms * 1e-3))
              << " Melements/s\n";

    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

// forward declare the kernel and pipe IDs to reduce name mangling
class InputKernelID;
class OutputKernelID;
class SortInPipeID;
class SortOutPipeID;

//
// perform the actual sort on the FPGA.
//
template <typename ValueT, typename IndexT, typename KernelPtrType>
double FPGASort(queue &q, ValueT *in_ptr, ValueT *out_ptr, IndexT count) {
  // the input and output pipe for the sorter
  using SortInPipe =
      sycl::ext::intel::pipe<SortInPipeID, sycl::vec<ValueT, kSortWidth>>;
  using SortOutPipe =
      sycl::ext::intel::pipe<SortOutPipeID, sycl::vec<ValueT, kSortWidth>>;

  // the sorter must sort a power of 2, so round up the requested count
  // to the nearest power of 2; we will pad the input to make sure the
  // output is still correct
  const IndexT sorter_count = fpga_tools::RoundUpPow2(count);

  // allocate some memory for the merge sort to use as temporary storage
  ValueT *buf_0, *buf_1;
#if defined (IS_BSP)
  if ((buf_0 = malloc_device<ValueT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_0'\n";
    std::terminate();
  }
  if ((buf_1 = malloc_device<ValueT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_1'\n";
    std::terminate();
  }
#else
  if ((buf_0 = malloc_shared<ValueT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_0'\n";
    std::terminate();
  }
  if ((buf_1 = malloc_shared<ValueT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_1'\n";
    std::terminate();
  }
#endif

  // This is the element we will pad the input with. In the case of this design,
  // we are sorting from smallest to largest and we want the last elements out
  // to be this element, so pad with MAX. If you are sorting from largest to
  // smallest, make this the MIN element. If you are sorting custom types
  // which are not supported by std::numeric_limits, then you will have to set
  // this padding element differently.
  const auto padding_element = std::numeric_limits<ValueT>::max();

  // We are sorting kSortWidth elements per cycle, so we will have 
  // sorter_count/kSortWidth pipe reads/writes from/to the sorter
  const IndexT total_pipe_accesses = sorter_count / kSortWidth;

  // launch the kernel that provides data into the sorter
  auto input_kernel_event =
    q.single_task<InputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the input pointer and write it to the sorter's input pipe
      KernelPtrType in(in_ptr);

      for (IndexT i = 0; i < total_pipe_accesses; i++) {
        // read data from device memory
        bool in_range = i * kSortWidth < count;

        // build the input pipe data
        sycl::vec<ValueT, kSortWidth> data;
        #pragma unroll
        for (unsigned char j = 0; j < kSortWidth; j++) {
          data[j] = in_range ? in[i * kSortWidth + j] : padding_element;
        }

        // write it into the sorter
        SortInPipe::write(data);
      }
    });

  // launch the kernel that reads out data from the sorter
  auto output_kernel_event =
    q.single_task<OutputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the sorter's output pipe and write to the output pointer
      KernelPtrType out(out_ptr);
      
      for (IndexT i = 0; i < total_pipe_accesses; i++) {
        // read data from the sorter
        auto data = SortOutPipe::read();

        // sorter_count is a multiple of kSortWidth
        bool in_range = i * kSortWidth < count;

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

  // launch the merge sort kernels
  auto merge_sort_events =
      SubmitMergeSort<ValueT, IndexT, SortInPipe, SortOutPipe, kSortWidth,
                      kMergeUnits>(q, sorter_count, buf_0, buf_1);

  // wait for the input and output kernels to finish
  auto start = high_resolution_clock::now();
  input_kernel_event.wait();
  output_kernel_event.wait();
  auto end = high_resolution_clock::now();

  // wait for the merge sort kernels to finish
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
bool Validate(T *val, T *ref, unsigned int count) {
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
