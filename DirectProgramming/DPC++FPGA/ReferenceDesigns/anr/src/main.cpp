#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <sstream> 
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "anr.hpp"
#include "anr_params.hpp"
#include "data_bundle.hpp"

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

// The size of the filter can be changed at the command line
#ifndef FILTER_SIZE
#define FILTER_SIZE 9
#endif
constexpr int kFilterSize = FILTER_SIZE;
static_assert(kFilterSize > 0);
// TODO: static asserts on filter size

// The number of pixels per cycle
#ifndef PIXELS_PER_CYCLE
//#define PIXELS_PER_CYCLE 1
#define PIXELS_PER_CYCLE 2
#endif
constexpr int kPixelsPerCycle = PIXELS_PER_CYCLE;
static_assert(kPixelsPerCycle > 0);

// The maximum number of columns in the image
#ifndef MAX_COLS
#define MAX_COLS 4096
#endif
constexpr int kMaxCols = MAX_COLS;
static_assert(kMaxCols > 0);
static_assert(kMaxCols > kPixelsPerCycle);
static_assert((kMaxCols % kPixelsPerCycle) == 0);

// the type to use for the pixel intensity values and a temporary type
// which should have more bits than the pixel type to check overflow
using PixelT = unsigned char; // 8 bits
using TmpT = unsigned long long; // 64 bits
static_assert(std::is_unsigned_v<PixelT>);
static_assert(std::is_unsigned_v<TmpT>);
static_assert(sizeof(TmpT) > sizeof(PixelT));

// the type used for indexing the rows and columns of the image
using IndexT = short;
static_assert(std::is_integral_v<IndexT>);
static_assert(!std::is_unsigned_v<IndexT>);

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in this file by main()
void ParseFiles(std::string data_dir, std::vector<PixelT>& in_pixels,
                std::vector<PixelT>& ref_pixels, int& cols, int& rows,
                ANRParams& params);

void WriteOutputFile(std::string data_dir, std::vector<PixelT>& pixels,
                     int cols, int rows);

template <typename T, typename KernelPtrType>
double RunANR(queue &q, T *in_ptr, T *out_ptr,
              ANRParams params, int cols, int rows, int frames);


bool Validate(PixelT *val, PixelT *ref, unsigned int count, double thresh=0.0001);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  /////////////////////////////////////////////////////////////
  // reading and validating the command line arguments
  // defaults
  std::string data_dir = "../test_data";
  bool passed = true;
#ifdef FPGA_EMULATOR
  int runs = 2;
  int frames = 2;
#else
  int runs = 9;
  int frames = 8;
#endif

  // get the input directory
  if (argc > 1) {
    data_dir = std::string(argv[1]);
  }

  // get the number of runs as the second command line argument
  if (argc > 2) {
    runs = atoi(argv[2]);
  }
  
  // get the number of frames as the third command line argument
  if (argc > 3) {
    frames = atoi(argv[3]);
  }

  // enforce at least two runs
  if (runs < 2) {
    std::cerr << "ERROR: 'runs' must be 2 or more\n";
    std::terminate();
  }

  // enforce at least one batch
  if (frames < 1) {
    std::cerr << "ERROR: 'frames' must be atleast 1\n";
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
  int cols, rows, pixel_count;
  ANRParams params;
  std::vector<PixelT> in_pixels, ref_pixels;
  ParseFiles(data_dir, in_pixels, ref_pixels, cols, rows, params);
  pixel_count = cols * rows;

  //////////////////////////////////////////////////////////////////////////////
  // DEBUG
  /*
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%d ", in_pixels[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
  */
  /*
  //std::array<PixelT, kFilterSize> test_in = {127, 0, 199, 0, 231, 0, 185, 0, 221};
  //PixelT ref = 211, res = 0;
  std::array<PixelT, kFilterSize> test_in = {0, 0, 0, 0, 211, 0, 164, 0, 255};
  PixelT ref = 210, res = 0;
  if (!TestBilateralFilter<PixelT, kFilterSize>(test_in, params, ref, res)) {
    std::cerr << "ERROR: TestBilateralFilter failed ("
              << (TmpT)res << " != " << (TmpT)ref << ")\n";
    std::terminate();
  }
  */
  //////////////////////////////////////////////////////////////////////////////

  // check rows and cols
  if ((cols % kPixelsPerCycle) != 0) {
    std::cerr << "ERROR: The number of columns (cols) must be a multiple of "
              << "kPixelsPerCycle\n";
    std::terminate();
  }

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

  // print out some info
  std::cout << "Data directory:   " << data_dir << "\n";
  std::cout << "Runs:             " << runs << "\n";
  std::cout << "Columns:          " << cols << "\n";
  std::cout << "Rows:             " << rows << "\n";
  std::cout << "Frames:           " << frames << "\n";
  std::cout << "Filter Size:      " << kFilterSize << "\n";
  std::cout << "Pixels Per Cycle: " << kPixelsPerCycle << "\n";
  std::cout << "Maximum Columns:  " << kMaxCols << "\n";
  #ifdef DISABLE_GLOBAL_MEM
    std::cout << "Global memory access is disabled; not validating results\n";
  #endif
  std::cout << "\n";

  try {
    // the pointer type for the kernel depends on whether data is coming from
    // USM host or device allocations
    using KernelPtrType = typename std::conditional_t<kUseUSMHostAllocation,
                                                      host_ptr<PixelT>,
                                                      device_ptr<PixelT>>;

    // run the sort multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      // run ANR
      time[i] =
        RunANR<PixelT, KernelPtrType>(q, in, out, params, cols, rows, frames);

      // Copy the output to 'out_vec'. In the case where we are using USM host
      // allocations this is unnecessary since we could simply deference
      // 'out'. However, it makes the following code cleaner since the output
      // is always in 'out_vec' and this copy is not part of the performance
      // timing.
      q.memcpy(out_pixels.data(), out, pixel_count * sizeof(PixelT)).wait();

      // validate the output
      #ifndef DISABLE_GLOBAL_MEM
        passed &= Validate(out_pixels.data(), ref_pixels.data(), pixel_count);
        //passed &= Validate(out_pixels.data(), in_pixels.data(), pixel_count);
      #else
        passed = true;
      #endif
    }
  } catch (exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  // free the memory allocated with malloc_host or malloc_device
  sycl::free(in, q);
  sycl::free(out, q);

  #ifndef DISABLE_GLOBAL_MEM
    // write the output files
    WriteOutputFile(data_dir, out_pixels, cols, rows);
  #endif

  // print the performance results
  if (passed) {
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels in actual FPGA hardware
    double avg_time_ms =
      std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);

    size_t input_count_mega = pixel_count * frames * sizeof(PixelT) * 1e-6;

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

// declare kernel and pipe names globally to reduce name mangling
class ANRInPipeID;
class ANROutPipeID;
class InputKernelID;
class OutputKernelID;
class ANRKernelID;

//
// Run the ANR algorithm on the FPGA
//
template <typename PixelT, typename KernelPtrType>
double RunANR(queue &q, PixelT *in_ptr, PixelT *out_ptr,
              ANRParams params, int cols, int rows, int frames) {
  // the input and output pipe for the sorter
  using PipeType = DataBundle<PixelT, kPixelsPerCycle>;
  using ANRInPipe = sycl::INTEL::pipe<ANRInPipeID, PipeType>;
  using ANROutPipe = sycl::INTEL::pipe<ANROutPipeID, PipeType>;

  // the total number of pixels for a single frame
  const int frame_pixel_count = cols * rows;
  const int iterations = frame_pixel_count / kPixelsPerCycle;

  // launch the kernel that provides data into the ANR kernel
#ifndef DISABLE_GLOBAL_MEM
  auto input_kernel_event = q.submit([&](handler &h) {
    h.single_task<InputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the input pointer and write it to the sorter's input pipe
      KernelPtrType in(in_ptr);

      [[intel::loop_coalesce(2)]]
      for (int f = 0; f < frames; f++) {
        for (int i = 0; i < iterations; i++) {
          // read from memory and write into pipe
          // NOTE: assumes number of pixels divisible by kPixelsPerCycle
          // TODO: turn off caching here
          PipeType pipe_data;
          #pragma unroll
          for (int k = 0; k < kPixelsPerCycle; k++) {
            pipe_data[k] = in[i * kPixelsPerCycle + k];
          }
          ANRInPipe::write(pipe_data);
        }
      }
    });
  });
#else
  auto input_kernel_event = q.submit([&](handler &h) {
    h.single_task<InputKernelID>([=] {
      [[intel::loop_coalesce(2)]]
      for (int f = 0; f < frames; f++) {
        for (int i = 0; i < iterations; i++) {
          PipeType pipe_data;
          #pragma unroll
          for (int k = 0; k < kPixelsPerCycle; k++) {
            pipe_data[k] = (i * kPixelsPerCycle + k);
          }
          ANRInPipe::write(pipe_data);
        }
      }
    });
  });
#endif

#ifndef DISABLE_GLOBAL_MEM
  // launch the kernel that reads out data from the ANR kernel
  auto output_kernel_event = q.submit([&](handler &h) {
    h.single_task<OutputKernelID>([=]() [[intel::kernel_args_restrict]] {
      // read from the sorter's output pipe and write to the output pointer
      KernelPtrType out(out_ptr);
      
      [[intel::loop_coalesce(2)]]
      for (int f = 0; f < frames; f++) {
        for (int i = 0; i < iterations; i++) {
          // read from output pipe and write to memory
          // NOTE: assumes number of pixels divisible by kPixelsPerCycle
          auto pipe_data = ANROutPipe::read();
          #pragma unroll
          for (int k = 0; k < kPixelsPerCycle; k++) {
            out[i * kPixelsPerCycle + k] = pipe_data[k];
          }
        }
      }
    });
  });
#else
  // launch the kernel that reads out data from the ANR kernel
  auto output_kernel_event = q.submit([&](handler &h) {
    h.single_task<OutputKernelID>([=] {      
      [[intel::loop_coalesce(2)]]
      for (int f = 0; f < frames; f++) {
        for (int i = 0; i < iterations; i++) {
          (void)ANROutPipe::read();
        }
      }
    });
  });
#endif

  // launch ANR kernel
  auto anr_kernel_events =
    SubmitANRKernels<PixelT, IndexT, ANRInPipe, ANROutPipe, kFilterSize, kPixelsPerCycle>(q, params,
                                                                                          cols, rows,
                                                                                          frames);

  // wait for the input and output kernels to finish
  auto start = high_resolution_clock::now();
  input_kernel_event.wait();
  output_kernel_event.wait();
  auto end = high_resolution_clock::now();

  // wait for the ANR kernels to finish
  for (auto &e : anr_kernel_events) {
    e.wait();
  }

  // return the duration in milliseconds, excluding memory transfers
  duration<double, std::milli> diff = end - start;
  return diff.count();
}

//
// Helper to parse data files
//
void ParseDataFile(std::string filename, std::vector<PixelT>& pixels,
                   int& cols, int& rows) {
  // create the file stream to parse
  std::ifstream is(filename);

  // get header and data
  std::string header_str, data_str;
  if (!std::getline(is, header_str)) {
    std::cerr << "ERROR: failed to get header line from " << filename << "\n";
    std::terminate();
  }
  if (!std::getline(is, data_str)) {
    std::cerr << "ERROR: failed to get data line from " << filename << "\n";
    std::terminate();
  }

  // first two elements are the width and height
  std::stringstream header_ss(header_str);
  header_ss >> cols >> rows;

  // expecting to parse cols*rows pixels
  pixels.resize(cols*rows);

  // parse all of the pixels
  std::stringstream data_ss(data_str);
  for (int i = 0; i < cols*rows; i++) {
    // parse using 64 bit integer
    TmpT x;

    // parse the pixel value
    if (!(data_ss >> x)) {
      std::cerr << "ERROR: ran out of pixels when parsing " << filename << "\n";
      std::terminate();
    }

    // check for parsing failure
    if (data_ss.fail()) {
      std::cerr << "ERROR: failed to parse pixel in " << filename << "\n";
      std::terminate();
    }

    // check if the parsed value fits in given type 'PixelT'
    if (x > static_cast<TmpT>(std::numeric_limits<PixelT>::max())) {
      std::cerr << "ERROR: value (" << x
                << ") is too big to store in pixel type 'T'\n";
      std::terminate();
    }
    if (x < static_cast<TmpT>(std::numeric_limits<PixelT>::min())) {
      std::cerr << "ERROR: value (" << x
                << ") is too small to store in pixel type 'T'\n";
      std::terminate();
    }

    // set the value
    pixels[i] = static_cast<PixelT>(x);
  }
}

//
// Function that parses an input file and returns the result
//
void ParseFiles(std::string data_dir, std::vector<PixelT>& in_pixels,
                std::vector<PixelT>& ref_pixels, int& cols, int& rows,
                ANRParams& params) {
  // parse the files
  int noisy_w, noisy_h;
  ParseDataFile(data_dir + "/input_noisy.data", in_pixels, noisy_w, noisy_h);
  int ref_w, ref_h;
  ParseDataFile(data_dir + "/output_ref.data", ref_pixels, ref_w, ref_h);

  // ensure dimensions match
  if (noisy_w != ref_w) {
    std::cerr << "noisy input and reference widths do not match "
              << noisy_w << " != " << ref_w << "\n";
    std::terminate();
  }
  if (noisy_h != ref_h) {
    std::cerr << "noisy input and reference heights do not match "
              << noisy_h << " != " << ref_h << "\n";
    std::terminate();
  }

  // set width and height
  cols = ref_w;
  rows = ref_h;

  // parse config parameters file
  params = ANRParams::FromFile(data_dir + "/param_config.data");

  // ensure the parsed filter size matches the compile time constant
  if (params.filter_size != kFilterSize) {
    std::cerr << "ERROR: the filter size parsed from " << data_dir
              << "/param_config.data (" << params.filter_size
              << ") does not match the compile time constant filter size "
              << "(kFilterSize = " << kFilterSize << ")\n";
    std::terminate();
  }
}

//
// Function to write the output to a file
//
void WriteOutputFile(std::string data_dir, std::vector<PixelT>& pixels,
                     int cols, int rows) {
  std::ofstream of(data_dir + "/output.data");

  // write the size
  of << cols << " " << rows << "\n";

  // write the pixels 
  for (auto& p : pixels) {
    of << static_cast<TmpT>(p) << " ";
  }
}

//
// Validate the output pixels using normalized root-mean-squared-error (RMSE)
//
bool Validate(PixelT *val, PixelT *ref, unsigned int count, double thresh) {
  // compute the MSE
  double nrmse = 0.0;
  for (unsigned int i = 0; i < count; i++) {
    auto diff = (val[i] - ref[i]);
    if(diff != 0) nrmse += diff * diff;
  }

  // compute the RMSE
  nrmse = std::sqrt(nrmse / count);

  // normalize
  nrmse /=
    (std::numeric_limits<PixelT>::max() - std::numeric_limits<PixelT>::min());

  // check MSE
  if (nrmse >= thresh) {
    std::cerr << "ERROR: Normalized root-mean-squared-error is too high: "
              << nrmse << "\n";
    return false;
  } else {
    //std::cout << "NRMSE = " << nrmse << "\n";
    return true;
  }
}
