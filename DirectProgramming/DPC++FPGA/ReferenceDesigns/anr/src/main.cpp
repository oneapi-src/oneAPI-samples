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

// the type to use for the pixel intensity values
using PixelT = unsigned char; // 8 bits

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in this file by main()
template <typename T>
void ParseFiles(std::string data_dir, std::vector<T>& in_pixels,
                std::vector<T>& ref_pixels, int& w, int& h);

template <typename T>
bool Validate(T *val, T *ref, unsigned int count);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  /////////////////////////////////////////////////////////////
  // reading and validating the command line arguments
  // defaults
  std::string data_dir = "../test_data";
  bool passed = true;
#ifdef FPGA_EMULATOR
  int runs = 2;
  int batches = 256;
#else
  int runs = 17;
  int batches = 1024;
#endif

  // get the input directory
  if (argc > 1) {
    data_dir = std::string(argv[1]);
  }

  // get the number of runs as the second command line argument
  if (argc > 2) {
    runs = atoi(argv[2]);
  }
  
  // get the number of batches as the third command line argument
  if (argc > 3) {
    batches = atoi(argv[3]);
  }

  // enforce at least two runs
  if (runs < 2) {
    std::cerr << "ERROR: 'runs' must be 2 or more\n";
    std::terminate();
  }

  // enforce at least one batch
  if (batches < 1) {
    std::cerr << "ERROR: 'batches' must be atleast 1\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////

  // the device selector
#ifdef FPGA_EMULATOR
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // create the device queue
  queue q(selector, dpc_common::exception_handler);

  // make sure the device supports USM device allocations
  auto d = q.get_device();
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

  // parse the input
  int w, h, pixel_count;
  std::vector<PixelT> in_pixels, ref_pixels;
  ParseFiles(data_dir, in_pixels, ref_pixels, w, h);
  pixel_count = w*h;

  // create the output (initialize to all 0s)
  std::vector<PixelT> out_pixels(in_pixels.size(), 0);

  // allocate the input and output data either in USM host or device allocations
  PixelT *in, *out;
  if constexpr (kUseUSMHostAllocation) {
    // using USM host allocations
    if ((in = malloc_host<PixelT>(pixel_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in' using "
                << "malloc_host\n";
      std::terminate();
    }
    if ((out = malloc_host<PixelT>(pixel_count, q)) == nullptr) {
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
    std::copy(in_pixels.begin(), in_pixels.end(), in);
    std::fill(out, out + pixel_count, PixelT(0));
  } else {
    // using device allocations
    if ((in = malloc_device<PixelT>(pixel_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in' using "
                << "malloc_device\n";
      std::terminate();
    }
    if ((out = malloc_device<PixelT>(pixel_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out' using "
                << "malloc_device\n";
      std::terminate();
    }

    // copy the input to the device memory and wait for the copy to finish
    q.memcpy(in, in_pixels.data(), pixel_count * sizeof(PixelT)).wait();
  }

  // track timing information, in ms
  std::vector<double> time(runs);

  try {
    // the pointer type for the kernel depends on whether data is coming from
    // USM host or device allocations
    using KernelPtrType = typename std::conditional_t<kUseUSMHostAllocation,
                                                      host_ptr<PixelT>,
                                                      device_ptr<PixelT>>;

    // run the sort multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      // run ANR
      //time[i] = fpga_sort<PixelT, IndexT, KernelPtrType>(q, in, out, count);

      // Copy the output to 'out_vec'. In the case where we are using USM host
      // allocations this is unnecessary since we could simply deference
      // 'out'. However, it makes the following code cleaner since the output
      // is always in 'out_vec' and this copy is not part of the performance
      // timing.
      q.memcpy(out_pixels.data(), out, pixel_count * sizeof(PixelT)).wait();

      // validate the output
      passed &= Validate(out_pixels.data(), ref_pixels.data(), pixel_count);
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

    size_t input_count_mega = pixel_count * sizeof(PixelT) * 1e-6;

    std::cout << "Execution time: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << (input_count_mega / (avg_time_ms * 1e-3))
              << " MB/s\n";

    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

/*
//
// perform the actual sort on the FPGA.
//
template <typename PixelT, typename IndexT, typename KernelPtrType>
double fpga_sort(queue &q, PixelT *in_ptr, PixelT *out_ptr, IndexT count) {
  // the input and output pipe for the sorter
  using SortInPipe =
      sycl::INTEL::pipe<SortInPipeID, sycl::vec<PixelT, kSortWidth>>;
  using SortOutPipe =
      sycl::INTEL::pipe<SortOutPipeID, sycl::vec<PixelT, kSortWidth>>;

  // the sorter must sort a power of 2, so round up the requested count
  // to the nearest power of 2; we will pad the input to make sure the
  // output is still correct
  const IndexT sorter_count = impu::math::RoundUpPow2(count);

  // allocate some memory for the merge sort to use as temporary storage
  PixelT *buf_0, *buf_1;
  if ((buf_0 = malloc_device<PixelT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_0'\n";
    std::terminate();
  }
  if ((buf_1 = malloc_device<PixelT>(sorter_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate memory for 'buf_1'\n";
    std::terminate();
  }

  // This is the element we will pad the input with. In the case of this design,
  // we are sorting from smallest to largest and we want the last elements out
  // to be this element, so pad with MAX. If you are sorting from largest to
  // smallest, make this the MIN element. If you are sorting custom types
  // which are not supported by std::numeric_limits, then you will have to set
  // this padding element differently.
  const auto padding_element = std::numeric_limits<PixelT>::max();

  // We are sorting kSortWidth elements per cycle, so we will have 
  // sorter_count/kSortWidth pipe reads/writes from/to the sorter
  const IndexT total_pipe_accesses = sorter_count / kSortWidth;

  // launch the kernel that provides data into the sorter
  auto input_kernel_event = q.submit([&](handler &h) {
    h.single_task<InputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the input pointer and write it to the sorter's input pipe
      KernelPtrType in(in_ptr);

      for (IndexT i = 0; i < total_pipe_accesses; i++) {
        // read data from device memory
        bool in_range = i < sorter_count;

        // build the input pipe data
        sycl::vec<PixelT, kSortWidth> data;
        #pragma unroll
        for (unsigned char j = 0; j < kSortWidth; j++) {
          data[j] = in_range ? in[i * kSortWidth + j] : padding_element;
        }

        // write it into the sorter
        SortInPipe::write(data);
      }
    });
  });

  // launch the kernel that reads out data from the sorter
  auto output_kernel_event = q.submit([&](handler &h) {
    h.single_task<OuputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the sorter's output pipe and write to the output pointer
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

  // launch the merge sort kernels
  auto merge_sort_events =
      SubmitMergeSort<PixelT, IndexT, SortInPipe, SortOutPipe, kSortWidth,
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
*/

//
// Function that parses an input file and returns the result
//
template <typename T>
void ParseFiles(std::string data_dir, std::vector<T>& in_pixels,
                std::vector<T>& ref_pixels, int& w, int& h) {
  // TODO
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
