#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include <utility> 

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
#ifdef FPGA_EMULATOR
#define MERGE_UNITS 4
#else
#define MERGE_UNITS 16
#endif
#endif
constexpr size_t kMergeUnits = MERGE_UNITS;

// sanity check on number of merge units
static_assert(kMergeUnits > 0);
static_assert(IsPow2(kMergeUnits));

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in main
template<typename ValueT, typename IndexT, typename KernelPtrType>
double fpga_sort(queue& q, ValueT *in_vec, ValueT *out_vec, IndexT count);

template<typename T>
bool validate(T *val, T *ref, unsigned int count);
////////////////////////////////////////////////////////////////////////////////

// main
int main(int argc, char* argv[]) {
  // type to sort, needs a compare function!
  using ValueT = int;

  // the type used to index in the sorter
  using IndexT = unsigned int;

  bool passed = true;

#ifdef FPGA_EMULATOR
  IndexT count = 32;
  int runs = 1;
#else
  IndexT count = 1 << 24;
  int runs = 17;
#endif

  // get the size of the input as the first command line argument
  if(argc > 1) {
    count = atoi(argv[1]);
  }

  // get the number of runs as the second command line argument
  if (argc > 2) {
    runs = atoi(argv[2]);
  }

  // enforce at least two runs
  runs = std::max((int)2, runs);
  
  // check args
  if (count <= kMergeUnits) {
    std::cerr << "ERROR: count must be greater than number of merge units\n";
    std::terminate();
  } else if (count > std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the index type does not have bits to count to "
              << "'count'\n";
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

  // the input and output data
  std::vector<ValueT> in_vec(count), out_vec(count), ref(count);

  // generate some random input data
  srand(time(NULL));
  std::generate(in_vec.begin(), in_vec.end(), [] { return rand() % 100; });

  // copy the input to the output reference and compute the expected result
  std::copy(in_vec.begin(), in_vec.end(), ref.begin());
  std::sort(ref.begin(), ref.end());

  // allocate the input and output data either in USM host or device allocations
  ValueT *in, *out;
  if constexpr(kUseUSMHostAllocation) {
    // streaming data using USM host allocations
    if ((in = malloc_host<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_host<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out'\n";
      std::terminate(); 
    }

    // Copy input to USM memory and reset the output
    // this is NOT efficient since, in the case of USM host allocations,
    // we could have simply generated the input data into the host allocation
    // and avoided this copy. However, it makes the code cleaner to assume the
    // input is always in 'in_vec' and this portion of the code is not part of
    // the performance timing anyways.
    std::copy(in_vec.begin(), in_vec.end(), in);
    std::fill(out, out + count, ValueT(0));
  } else {
    // streaming data using device allocations
    if ((in = malloc_device<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_device<ValueT>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out'\n";
      std::terminate(); 
    }

    // copy input to the device memory adn wait for the copy to finish
    q.memcpy(in, in_vec.data(), count*sizeof(ValueT)).wait();
  }

  // track timing information, in ms
  std::vector<double> time(runs);

  try {
    std::cout << "Running sort " << runs << " times for an "
              << "input size of " << count << " using "
              << kMergeUnits << " merge units\n";
    std::cout << "Streaming data from "
              << (kUseUSMHostAllocation ? "host" : "device") << " memory\n";
    
    // the pointer type for the kernel depends on whether we are streaming
    // data via USM host allocations or device memory
    using KernelPtrType = typename std::conditional_t<kUseUSMHostAllocation,
                                                      host_ptr<ValueT>,
                                                      device_ptr<ValueT>>;


    // run some sort iterations
    for (int i = 0; i < runs; i++) {
      // run the sort
      time[i] = fpga_sort<ValueT, IndexT, KernelPtrType>(q, in, out, count);

      // Copy output to out_vec. In the case where we are using USM host
      // allocations this is unnecessary since we could simply deference
      // 'out'. However, it makes the following code cleaner since output
      // is always in 'out_vec' and this copy is not part of the performance
      // timing anyway.
      q.memcpy(out_vec.data(), out, count*sizeof(ValueT)).wait();

      // validate the output
      passed &= validate(out_vec.data(), ref.data(), count);
    }
  } catch (exception const& e) {
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
    std::cout << "Throughput: "
              << (input_count_m / (avg_time * 1e-3)) << " Melements/s\n"; 
#endif

    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 0;
  }
}

// forward declare the kernel and pipe IDs to reduce name mangling
class ProducerKernelID;
class ConsumerKernelID;
class SortInPipeID;
class SortOutPipeID;

//
// perform the actual sort on the FPGA.
//
template<typename ValueT, typename IndexT, typename KernelPtrType>
double fpga_sort(queue& q, ValueT *in_ptr, ValueT* out_ptr, IndexT count) {
  // the input and output pipe for the sorter
  using SortInPipe = sycl::INTEL::pipe<SortInPipeID, ValueT>;
  using SortOutPipe = sycl::INTEL::pipe<SortOutPipeID, ValueT>;

  // the sorter must sort a power of 2, so round up the requested count
  // to the nearest power of 2; we will pad the input to make sure the
  // output is still correct
  IndexT sorter_count = RoundUpPow2(count);

  // This is the element we will pad the input with. In the case of this design,
  // we are sorting from smallest to largest and we want the last output
  // elements to be this element, so pad with MAX. If you are sorting from
  // largest to smallest, make this the minimum element. If you are sorting
  // custom types which are not supported by std::numeric_limits, then you will
  // have to set this padding element differently.
  const auto padding_element = std::numeric_limits<ValueT>::max();
 
  // launch the producer
  auto producer_event = q.submit([&](handler& h) {
    h.single_task<ProducerKernelID>([=] {
      // read from the input pointer and write it to the sorter's input pipe
      KernelPtrType in(in_ptr);
      for (unsigned int i = 0; i < sorter_count; i++) {
        auto data = (i < count) ? in[i] : padding_element;
        SortInPipe::write(data);
      }
    });
  });


  // launch the consumer
  auto consumer_event = q.submit([&](handler& h) {
    h.single_task<ConsumerKernelID>([=] {
      // read from the sorters output pipe and write to the output pointer
      KernelPtrType out(out_ptr);
      for (unsigned int i = 0; i < sorter_count; i++) {
        auto data = SortOutPipe::read();
        if (i < count) out[i] = data;
      }
    });
  });

  // allocate some memory for the merge sort to use as temporary storage
  ValueT* buf_0, *buf_1;
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
    SubmitMergeSort<ValueT, IndexT, SortInPipe, SortOutPipe, kMergeUnits>(q,
                                                                   sorter_count,
                                                                   buf_0, buf_1);

  // wait for the producer and consumer to finish
  auto start = high_resolution_clock::now();
  producer_event.wait();
  consumer_event.wait();
  auto end = high_resolution_clock::now();

  // wait for the merge sort events to all finish
  for (auto& e : merge_sort_events) {
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
template<typename T>
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
